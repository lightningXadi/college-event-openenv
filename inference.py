"""
inference.py  –  OpenEnv compatibility shim
--------------------------------------------
- Exposes the FastAPI `app` for uvicorn (inference:app)
- Prints [START]/[STEP]/[END] structured output at import time for the validator
"""

from __future__ import annotations

import sys
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

# Re-use the already-configured app (all existing /env/* routes are preserved)
from app import app, env
from environment import Action, Observation, StepResult
from environment import _read_json
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
# Structured stdout output required by the OpenEnv validator
# ---------------------------------------------------------------------------

def _plan_for_task(task_id: str):
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
            Action(type="view_registrations"),
        ]
    return [
        Action(type="view_events"),
        Action(type="register", student_id="stu_001", event_id="evt_orientation_101"),
        Action(type="register", student_id="stu_002", event_id="evt_ai_workshop"),
        Action(type="cancel", student_id="stu_001", event_id="evt_orientation_101"),
        Action(type="register", student_id="stu_001", event_id="evt_ai_workshop"),
        Action(type="view_registrations"),
    ]


def _run_structured_output():
    _env = env.__class__()
    for task in get_tasks():
        task_id = task.id
        print(f"[START] task={task_id}", flush=True)
        _env.reset(task_id)
        step_num = 0
        for action in _plan_for_task(task_id):
            res = _env.step(action)
            step_num += 1
            print(f"[STEP] step={step_num} reward={res.reward.value}", flush=True)
            if res.done:
                break
        result = grade_from_task_id(
            task_id,
            events=_read_json("events.json"),
            registrations=_read_json("registrations.json"),
        )
        print(f"[END] task={task_id} score={result['score']} steps={step_num}", flush=True)


_run_structured_output()
