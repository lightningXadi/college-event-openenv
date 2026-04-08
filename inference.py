"""
inference.py  –  OpenEnv compatibility shim
--------------------------------------------
The OpenEnv validator expects the following top-level routes:
  POST /reset          -> resets the environment
  POST /step           -> takes an action
  GET  /state          -> returns current observation
  GET  /tasks          -> lists available tasks
  POST /grader         -> grades the current episode
  POST /baseline       -> runs the deterministic baseline
  GET  /health         -> health-check

This file re-exports the same FastAPI `app` from app.py and adds
the flat aliases that OpenEnv's structural checker requires.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

# Re-use the already-configured app (all existing /env/* routes are preserved)
from app import app, env
from environment import Action, Observation, StepResult


class ResetRequest(BaseModel):
    task_id: Optional[str] = "college_event_task_1"


# ---------------------------------------------------------------------------
# Flat OpenEnv-required routes
# ---------------------------------------------------------------------------


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None) -> Observation:
    """OpenEnv structural check: POST /reset"""
    task_id = (req.task_id if req and req.task_id else None) or "college_event_task_1"
    try:
        return env.reset(task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown task_id")


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """OpenEnv structural check: POST /step"""
    return env.step(action)


@app.get("/state", response_model=Observation)
def state() -> Observation:
    """OpenEnv structural check: GET /state"""
    return env.state()
