"""
inference.py  –  OpenEnv compatibility shim
--------------------------------------------
- Exposes the FastAPI `app` for uvicorn (inference:app)
- Prints [START]/[STEP]/[END] structured output via LiteLLM proxy
  (only when API_KEY and API_BASE_URL are injected by the validator)
"""

from __future__ import annotations

import json
import os
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
# LLM agent using injected proxy credentials
# ---------------------------------------------------------------------------

def _get_llm_action(client, task_description: str, obs: dict, history: list) -> dict:
    system_prompt = (
        "You are an AI agent controlling a college event registration system.\n\n"
        "Your goal is to complete the given task by choosing actions one at a time.\n\n"
        "Available actions (respond with ONLY valid JSON, nothing else):\n"
        '1. View events:         {"type": "view_events"}\n'
        '2. View registrations:  {"type": "view_registrations"}\n'
        '3. Register student:    {"type": "register", "student_id": "stu_001", "event_id": "evt_orientation_101"}\n'
        '4. Cancel registration: {"type": "cancel", "student_id": "stu_001", "event_id": "evt_orientation_101"}\n\n'
        "Rules:\n"
        "- Always start by viewing events to understand what is available.\n"
        "- Never register the same student for the same event twice.\n"
        "- Respond ONLY with a JSON object. No explanation, no markdown, no extra text."
    )
    messages = [
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
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Structured stdout output — only runs when validator injects credentials
# ---------------------------------------------------------------------------

def _run_structured_output() -> None:
    api_key = os.environ.get("API_KEY", "").strip()
    api_base = os.environ.get("API_BASE_URL", "").strip()

    # Skip silently during normal HF Space startup (no credentials injected yet).
    # The validator will re-run this module with credentials present.
    if not api_key or not api_base:
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_base)

    _env = env.__class__()

    for task in get_tasks():
        task_id = task.id
        print(f"[START] task={task_id}", flush=True)

        obs = _env.reset(task_id)
        step_num = 0
        history: list = []

        for _ in range(task.max_steps):
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
            action_dict = _get_llm_action(client, task.description, obs_dict, history)
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
