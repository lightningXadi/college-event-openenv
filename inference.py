"""
inference.py  –  OpenEnv compatibility shim
--------------------------------------------
- Exposes the FastAPI `app` for uvicorn (inference:app)
- Prints [START]/[STEP]/[END] structured output via LiteLLM proxy
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from app import app, env
from environment import Action, Observation, StepResult, _read_json
from grader import grade_from_task_id
from tasks import get_tasks


# ---------------------------------------------------------------------------
# Flat OpenEnv-required routes
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "college_event_task_1"


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None) -> Observation:
    task_id = (req.task_id if req and req.task_id else None) or "college_event_task_1"
    try:
        return env.reset(task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown task_id")


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    return env.step(action)


@app.get("/state", response_model=Observation)
def state() -> Observation:
    return env.state()


# ---------------------------------------------------------------------------
# LLM call — tries multiple URL patterns, wraps all errors
# ---------------------------------------------------------------------------

def _chat_completion(url: str, api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_llm_action(api_key: str, api_base: str, task_description: str, obs: dict, history: list) -> dict:
    system_prompt = (
        "You are an AI agent controlling a college event registration system.\n\n"
        "Your goal is to complete the given task by choosing actions one at a time.\n\n"
        "Available actions (respond with ONLY valid JSON, nothing else):\n"
        '1. View events:         {"type": "view_events"}\n'
        '2. View registrations:  {"type": "view_registrations"}\n'
        '3. Register student:    {"type": "register", "student_id": "stu_001", "event_id": "evt_orientation_101"}\n'
        '4. Cancel registration: {"type": "cancel", "student_id": "stu_001", "event_id": "evt_orientation_101"}\n\n'
        "Rules:\n"
        "- Always start by viewing events.\n"
        "- Never register the same student for the same event twice.\n"
        "- Respond ONLY with a JSON object. No explanation, no markdown, no extra text."
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task: {task_description}\n\n"
                    f"Current observation:\n{json.dumps(obs, indent=2)}\n\n"
                    f"Action history so far:\n{json.dumps(history, indent=2)}\n\n"
                    "What is the next action? Respond with JSON only."
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 150,
    }

    base = api_base.rstrip("/")

    # Try multiple URL patterns in case the proxy uses a different format
    candidate_urls = []
    if base.endswith("/v1"):
        candidate_urls.append(base + "/chat/completions")
        candidate_urls.append(base[: -len("/v1")] + "/chat/completions")
    else:
        candidate_urls.append(base + "/v1/chat/completions")
        candidate_urls.append(base + "/chat/completions")

    last_error = None
    for url in candidate_urls:
        try:
            data = _chat_completion(url, api_key, payload)
            raw = data["choices"][0]["message"]["content"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            last_error = e
            continue

    # All URLs failed — raise the last error so validator sees it clearly
    raise RuntimeError(f"LLM proxy unreachable. Last error: {last_error}. Tried: {candidate_urls}")


# ---------------------------------------------------------------------------
# Structured stdout output — only runs when validator injects credentials
# ---------------------------------------------------------------------------

def _run_structured_output() -> None:
    api_key = os.environ.get("API_KEY", "").strip()
    api_base = os.environ.get("API_BASE_URL", "").strip()

    if not api_key or not api_base:
        return

    _env = env.__class__()

    for task in get_tasks():
        task_id = task.id
        print(f"[START] task={task_id}", flush=True)

        obs = _env.reset(task_id)
        step_num = 0
        history: list = []

        for _ in range(task.max_steps):
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
            action_dict = _get_llm_action(api_key, api_base, task.description, obs_dict, history)
            action = Action(**action_dict)
            history.append(action_dict)

            res = _env.step(action)
            step_num += 1
            print(f"[STEP] step={step_num} reward={res.reward.value}", flush=True)
            obs = res.observation
            if res.done:
                break

        result = grade_from_task_id(
            task_id,
            events=_read_json("events.json"),
            registrations=_read_json("registrations.json"),
        )
        print(f"[END] task={task_id} score={result['score']} steps={step_num}", flush=True)


_run_structured_output()
