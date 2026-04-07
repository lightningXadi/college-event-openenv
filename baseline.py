from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from environment import Action, CollegeEventEnv


# ---------------------------------------------------------------------------
# OpenAI-powered agent
# ---------------------------------------------------------------------------

def _get_openai_action(client, task_description: str, obs: dict, history: List[dict]) -> dict:
    """Ask GPT to decide the next action based on current observation."""

    system_prompt = """You are an AI agent controlling a college event registration system.

Your goal is to complete the given task by choosing actions one at a time.

Available actions (respond with ONLY valid JSON, nothing else):
1. View events:         {"type": "view_events"}
2. View registrations:  {"type": "view_registrations"}
3. Register student:    {"type": "register", "student_id": "stu_001", "event_id": "evt_orientation_101"}
4. Cancel registration: {"type": "cancel", "student_id": "stu_001", "event_id": "evt_orientation_101"}

Rules:
- Always start by viewing events to understand what is available.
- Never register the same student for the same event twice.
- Respond ONLY with a JSON object. No explanation, no markdown, no extra text.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task_description}\n\nCurrent observation:\n{json.dumps(obs, indent=2)}\n\nAction history so far:\n{json.dumps(history, indent=2)}\n\nWhat is the next action? Respond with JSON only."}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model wraps in ```json
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_baseline_episode(env: CollegeEventEnv, task_id: str) -> Dict[str, Any]:
    """
    Runs the baseline agent against the environment.
    Uses OpenAI GPT if OPENAI_API_KEY is set, otherwise falls back to deterministic plan.
    Used by the FastAPI /baseline endpoint.
    """

    api_key = os.environ.get("OPENAI_API_KEY", "")

    if api_key:
        return _run_openai_episode(env, task_id, api_key)
    else:
        return _run_deterministic_episode(env, task_id)


# ---------------------------------------------------------------------------
# OpenAI episode
# ---------------------------------------------------------------------------

def _run_openai_episode(env: CollegeEventEnv, task_id: str, api_key: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        # openai package not installed, fall back
        return _run_deterministic_episode(env, task_id)

    client = OpenAI(api_key=api_key)

    from tasks import tasks_by_id
    task_spec = tasks_by_id().get(task_id)
    task_description = task_spec.description if task_spec else task_id

    obs = env.reset(task_id)
    rewards: List[float] = []
    errors: List[str] = []
    history: List[dict] = []
    max_steps = task_spec.max_steps if task_spec else 10

    for step in range(max_steps):
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

        try:
            action_dict = _get_openai_action(client, task_description, obs_dict, history)
            action = Action(**action_dict)
        except Exception as e:
            errors.append(f"Step {step} action parse error: {e}")
            break

        history.append(action_dict)
        res = env.step(action)
        rewards.append(res.reward.value)
        obs = res.observation

        if res.observation.errors:
            errors.extend(res.observation.errors)
        if res.done:
            break

    from environment import _read_json
    from grader import grade_from_task_id

    result = grade_from_task_id(
        task_id,
        events=_read_json("events.json"),
        registrations=_read_json("registrations.json"),
    )

    return {
        "task_id": task_id,
        "episode_id": obs.episode_id,
        "agent": "openai/gpt-4o-mini",
        "final_score": float(result["score"]),
        "reward_sum": float(sum(rewards)),
        "num_steps": int(env.state().step_count),
        "errors_seen": errors[:10],
        "checks": result["checks"],
    }


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _plan_for_task(task_id: str) -> List[Action]:
    if task_id == "college_event_task_1":
        return [
            Action(type="view_events"),
            Action(type="register", student_id="stu_001", event_id="evt_orientation_101"),
            Action(type="view_registrations"),
        ]

    if task_id == "college_event_task_2":
        return [
            Action(type="view_events"),
            Action(type="register", student_id="stu_001", event_id="evt_ai_workshop"),
            Action(type="register", student_id="stu_002", event_id="evt_ai_workshop"),
            Action(type="register", student_id="stu_002", event_id="evt_ai_workshop"),  # duplicate demo
            Action(type="view_registrations"),
        ]

    # Task 3 (hard)
    return [
        Action(type="view_events"),
        Action(type="register", student_id="stu_001", event_id="evt_orientation_101"),
        Action(type="register", student_id="stu_002", event_id="evt_ai_workshop"),
        Action(type="cancel", student_id="stu_001", event_id="evt_orientation_101"),
        Action(type="register", student_id="stu_001", event_id="evt_ai_workshop"),
        Action(type="view_registrations"),
    ]


def _run_deterministic_episode(env: CollegeEventEnv, task_id: str) -> Dict[str, Any]:
    obs = env.reset(task_id)
    rewards: List[float] = []
    errors: List[str] = []

    for a in _plan_for_task(task_id):
        res = env.step(a)
        rewards.append(res.reward.value)
        if res.observation.errors:
            errors.extend(res.observation.errors)
        if res.done:
            break

    from environment import _read_json
    from grader import grade_from_task_id

    result = grade_from_task_id(
        task_id,
        events=_read_json("events.json"),
        registrations=_read_json("registrations.json"),
    )

    return {
        "task_id": task_id,
        "episode_id": obs.episode_id,
        "agent": "deterministic",
        "final_score": float(result["score"]),
        "reward_sum": float(sum(rewards)),
        "num_steps": int(env.state().step_count),
        "errors_seen": errors[:10],
        "checks": result["checks"],
    }


# ---------------------------------------------------------------------------
# HTTP runner (for running baseline.py directly from terminal)
# ---------------------------------------------------------------------------

def run_baseline_over_http(
    *,
    base_url: str = "http://localhost:8000",
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    tasks_to_run: List[str]
    if task_id:
        tasks_to_run = [task_id]
    else:
        tasks_to_run = [t["id"] for t in requests.get(f"{base_url}/tasks", timeout=10).json()]

    results: List[Dict[str, Any]] = []
    for tid in tasks_to_run:
        r = requests.post(f"{base_url}/env/reset", json={"task_id": tid}, timeout=10)
        r.raise_for_status()
        episode_id = r.json()["episode_id"]

        rewards: List[float] = []
        for a in _plan_for_task(tid):
            rr = requests.post(f"{base_url}/env/step", json=a.model_dump(), timeout=10)
            rr.raise_for_status()
            payload = rr.json()
            rewards.append(float(payload["reward"]["value"]))
            if payload["done"]:
                break

        gr = requests.post(f"{base_url}/grader", json={"task_id": tid}, timeout=10)
        gr.raise_for_status()
        score = float(gr.json()["score"])

        results.append({
            "task_id": tid,
            "episode_id": episode_id,
            "final_score": score,
            "reward_sum": float(sum(rewards)),
        })

    return {"base_url": base_url, "results": results}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run the baseline agent over HTTP.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--task-id", default=None)
    args = p.parse_args()

    out = run_baseline_over_http(base_url=args.base_url, task_id=args.task_id)
    print(json.dumps(out, indent=2))
