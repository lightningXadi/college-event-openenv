from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from environment import Action, CollegeEventEnv, Observation, StepResult
from grader import grade_from_task_id
from tasks import TaskSpec, get_tasks, tasks_by_id
from fastapi.openapi.docs import get_swagger_ui_html


app = FastAPI(
    title="College Event Registration API",
    description="OpenEnv-compatible environment for college event management.",
    version="1.0.0",
    docs_url=None,        # disable default docs
    redoc_url=None
)


env = CollegeEventEnv()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"


class TaskInfo(BaseModel):
    id: str
    title: str
    difficulty: str
    description: str
    max_steps: int


class ResetRequest(BaseModel):
    task_id: str


class GraderRequest(BaseModel):
    task_id: str
    # If not provided, grader evaluates the current environment JSON files.
    # (This keeps the demo beginner-friendly and reproducible.)
    events: Optional[List[Dict[str, Any]]] = None
    registrations: Optional[List[Dict[str, Any]]] = None


class GraderResponse(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    checks: List[Dict[str, Any]]


class BaselineRequest(BaseModel):
    task_id: Optional[str] = None


class BaselineResponse(BaseModel):
    results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html>

<head>
<title>College Event Registration</title>

<style>

body{
    margin:0;
    font-family: 'Segoe UI', Arial, sans-serif;
    background: linear-gradient(135deg,#eef2ff,#dbeafe);
    color:#333;
}

/* container */

.container{
    max-width:1100px;
    margin:auto;
    padding:60px 20px;
    text-align:center;
}

/* header */

h1{
    font-size:48px;
    color:#4f46e5;
    margin-bottom:10px;
}

.subtitle{
    font-size:18px;
    color:#555;
    margin-bottom:50px;
}

/* main card */

.card{
    background:white;
    padding:40px;
    border-radius:14px;
    box-shadow:0 12px 30px rgba(0,0,0,0.08);
}

/* buttons */

.buttons{
    margin-top:30px;
}

a{
    text-decoration:none;
    margin:10px;
    padding:14px 26px;
    border-radius:8px;
    font-weight:bold;
    background:#4f46e5;
    color:white;
    transition:0.25s;
    display:inline-block;
}

a:hover{
    transform:translateY(-2px);
    box-shadow:0 8px 16px rgba(0,0,0,0.15);
}

.secondary{
    background:#10b981;
}

/* features section */

.features{
    margin-top:60px;
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
    gap:20px;
}

.feature{
    background:white;
    padding:25px;
    border-radius:12px;
    box-shadow:0 8px 20px rgba(0,0,0,0.06);
}

.feature h3{
    color:#4f46e5;
}

/* footer */

footer{
    margin-top:60px;
    font-size:14px;
    color:#666;
}

</style>
</head>


<body>

<div class="container">

<div class="card">

<h1>College Event Registration</h1>

<p class="subtitle">
OpenEnv compatible AI environment for managing event registrations,
capacity limits, and cancellations.
</p>

<div class="buttons">
<a href="/docs">API Documentation</a>
<a href="/tasks" class="secondary">View Tasks</a>
</div>

</div>


<div class="features">

<div class="feature">
<h3>Event Management</h3>
<p>View and manage multiple college events with capacity limits.</p>
</div>

<div class="feature">
<h3>Student Registration</h3>
<p>Register students for events while preventing duplicates.</p>
</div>

<div class="feature">
<h3>AI Environment</h3>
<p>Built for OpenEnv agents with rewards, tasks, and grading.</p>
</div>

</div>


<footer>
OpenEnv Hackathon Project • Built with FastAPI
</footer>

</div>

</body>
</html>
"""


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    out: List[TaskInfo] = []
    for t in get_tasks():
        out.append(
            TaskInfo(
                id=t.id,
                title=t.title,
                difficulty=t.difficulty,
                description=t.description,
                max_steps=t.max_steps,
            )
        )
    return out


# --- OpenEnv core API over HTTP (useful for agent/baseline scripts) --------


@app.post("/env/reset", response_model=Observation)
def env_reset(req: ResetRequest) -> Observation:
    try:
        return env.reset(req.task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown task_id")


@app.get("/env/state", response_model=Observation)
def env_state() -> Observation:
    return env.state()


@app.post("/env/step", response_model=StepResult)
def env_step(action: Action) -> StepResult:
    return env.step(action)


# --- Grader ---------------------------------------------------------------


@app.post("/grader", response_model=GraderResponse)
def grader(req: GraderRequest) -> GraderResponse:
    tasks: Dict[str, TaskSpec] = tasks_by_id()
    if req.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    if req.events is None or req.registrations is None:
        # Evaluate the current episode data stored in JSON files.
        from environment import _read_json  # local import to keep API surface small

        events = _read_json("events.json")
        registrations = _read_json("registrations.json")
    else:
        events = req.events
        registrations = req.registrations

    try:
        result = grade_from_task_id(req.task_id, events=events, registrations=registrations)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    return GraderResponse(task_id=req.task_id, score=float(result["score"]), checks=result["checks"])


# --- Baseline -------------------------------------------------------------


@app.post("/baseline", response_model=BaselineResponse)
def baseline(req: BaselineRequest) -> BaselineResponse:
    """
    Runs a simple, deterministic baseline (no external model calls).
    If task_id is omitted, runs all tasks.
    """

    from baseline import run_baseline_episode

    tasks = get_tasks()
    if req.task_id is not None:
        by = tasks_by_id()
        if req.task_id not in by:
            raise HTTPException(status_code=404, detail="Unknown task_id")
        tasks = [by[req.task_id]]

    results: List[Dict[str, Any]] = []
    for t in tasks:
        results.append(run_baseline_episode(env, t.id))
    return BaselineResponse(results=results)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()
from fastapi.responses import HTMLResponse

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    swagger = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="College Event API Explorer"
    ).body.decode()

    return HTMLResponse(f"""
    <html>
    <head>
        <style>
            .banner {{
                background:#4f46e5;
                color:white;
                padding:15px;
                font-size:20px;
                text-align:center;
                font-weight:bold;
            }}
        </style>
    </head>
    <body>
        <div class="banner">
        College Event Registration API • OpenEnv Hackathon
        </div>
        {swagger}
    </body>
    </html>
    """)

