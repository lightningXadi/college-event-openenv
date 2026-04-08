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
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>College Event Registration</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">

<style>
  :root {
    --bg:       #f4f0eb;
    --surface:  #fffef9;
    --ink:      #1a1108;
    --ink-muted:#5a5040;
    --accent:   #ff6154;
    --accent2:  #23c4a4;
    --accent3:  #f9c846;
    --border:   #e0d8cc;
    --shadow:   0 2px 0 var(--border);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
    overflow-x: hidden;
    position: relative;
  }

  /* ── Background Animation ── */
  .bg-canvas {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
  }

  .blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.28;
    animation: drift linear infinite;
  }

  .blob-1 {
    width: 520px; height: 520px;
    background: var(--accent);
    top: -180px; left: -140px;
    animation-duration: 22s;
  }
  .blob-2 {
    width: 400px; height: 400px;
    background: var(--accent2);
    bottom: -120px; right: -100px;
    animation-duration: 28s;
    animation-direction: reverse;
  }
  .blob-3 {
    width: 300px; height: 300px;
    background: var(--accent3);
    top: 40%; left: 55%;
    animation-duration: 18s;
    animation-delay: -9s;
  }

  @keyframes drift {
    0%   { transform: translate(0, 0)    rotate(0deg); }
    33%  { transform: translate(40px, 30px) rotate(120deg); }
    66%  { transform: translate(-30px, 50px) rotate(240deg); }
    100% { transform: translate(0, 0)    rotate(360deg); }
  }

  /* Grid dots overlay */
  .bg-grid {
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
      radial-gradient(circle, rgba(26,17,8,0.10) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
  }

  /* ── Layout ── */
  .page {
    position: relative;
    z-index: 1;
    max-width: 1080px;
    margin: 0 auto;
    padding: 0 24px 80px;
  }

  /* ── Nav ── */
  nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 24px 0 16px;
    border-bottom: 2px solid var(--ink);
  }

  .nav-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 22px;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .nav-logo .dot {
    width: 10px; height: 10px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
  }

  .nav-links {
    display: flex;
    gap: 12px;
  }

  .nav-links a {
    font-size: 14px;
    font-weight: 500;
    color: var(--ink);
    text-decoration: none;
    padding: 8px 18px;
    border-radius: 6px;
    border: 2px solid var(--ink);
    background: var(--surface);
    box-shadow: var(--shadow);
    transition: transform 0.12s, box-shadow 0.12s, background 0.12s;
  }

  .nav-links a:hover {
    background: var(--ink);
    color: var(--bg);
    transform: translateY(-2px);
    box-shadow: 0 4px 0 rgba(26,17,8,0.25);
  }

  .nav-links a.filled {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }

  .nav-links a.filled:hover {
    background: #e84e42;
    border-color: #e84e42;
    color: #fff;
  }

  /* ── Hero ── */
  .hero {
    padding: 72px 0 56px;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 32px;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--accent);
    border: 2px solid var(--accent);
    border-radius: 100px;
    padding: 5px 14px;
    margin-bottom: 20px;
  }

  .hero-badge::before {
    content: '';
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
  }

  h1 {
    font-family: 'Instrument Serif', serif;
    font-size: clamp(48px, 6vw, 80px);
    line-height: 1.05;
    letter-spacing: -1.5px;
    color: var(--ink);
    max-width: 680px;
  }

  h1 em {
    font-style: italic;
    color: var(--accent);
  }

  .hero-sub {
    font-size: 17px;
    color: var(--ink-muted);
    line-height: 1.6;
    max-width: 520px;
    margin-top: 20px;
  }

  .hero-cta {
    display: flex;
    gap: 12px;
    margin-top: 36px;
    flex-wrap: wrap;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 14px 28px;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    font-weight: 600;
    text-decoration: none;
    border: 2px solid var(--ink);
    cursor: pointer;
    transition: transform 0.12s, box-shadow 0.12s;
    box-shadow: 3px 3px 0 var(--ink);
  }

  .btn:hover {
    transform: translate(-2px, -2px);
    box-shadow: 5px 5px 0 var(--ink);
  }

  .btn-primary {
    background: var(--ink);
    color: var(--bg);
  }

  .btn-secondary {
    background: var(--surface);
    color: var(--ink);
  }

  /* ── Stats strip ── */
  .stats {
    display: flex;
    gap: 1px;
    background: var(--ink);
    border: 2px solid var(--ink);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 72px;
  }

  .stat {
    flex: 1;
    background: var(--surface);
    padding: 28px 24px;
    text-align: center;
  }

  .stat-num {
    font-family: 'Instrument Serif', serif;
    font-size: 36px;
    color: var(--ink);
    line-height: 1;
  }

  .stat-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--ink-muted);
    margin-top: 6px;
  }

  /* ── Feature cards ── */
  .section-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: var(--ink-muted);
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 72px;
  }

  .feature-card {
    background: var(--surface);
    border: 2px solid var(--ink);
    border-radius: 12px;
    padding: 32px 28px;
    box-shadow: 4px 4px 0 var(--ink);
    transition: transform 0.14s, box-shadow 0.14s;
  }

  .feature-card:hover {
    transform: translate(-3px, -3px);
    box-shadow: 7px 7px 0 var(--ink);
  }

  .feature-card:nth-child(2) { background: var(--ink); color: #f4f0eb; }
  .feature-card:nth-child(2) .fc-title { color: var(--accent3); }
  .feature-card:nth-child(2) .fc-body  { color: #b8a99a; }

  .fc-icon {
    font-size: 28px;
    margin-bottom: 16px;
    display: block;
  }

  .fc-title {
    font-family: 'Instrument Serif', serif;
    font-size: 22px;
    color: var(--ink);
    margin-bottom: 10px;
    line-height: 1.2;
  }

  .fc-body {
    font-size: 14px;
    color: var(--ink-muted);
    line-height: 1.65;
  }

  /* ── API section ── */
  .api-section {
    background: var(--ink);
    border-radius: 16px;
    padding: 48px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    align-items: center;
    margin-bottom: 72px;
  }

  .api-title {
    font-family: 'Instrument Serif', serif;
    font-size: 36px;
    color: #f4f0eb;
    line-height: 1.1;
    margin-bottom: 14px;
  }

  .api-desc {
    font-size: 15px;
    color: #9a8a7a;
    line-height: 1.7;
    margin-bottom: 24px;
  }

  .endpoint-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .endpoint {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    font-size: 13px;
  }

  .method {
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border-radius: 4px;
    min-width: 44px;
    text-align: center;
  }

  .method.get  { background: #23c4a4; color: #0a2e28; }
  .method.post { background: #f9c846; color: #2a200a; }

  .endpoint-path {
    font-family: 'DM Mono', 'Courier New', monospace;
    color: #e0d8cc;
    font-size: 13px;
  }

  .endpoint-note {
    margin-left: auto;
    font-size: 11px;
    color: #6a5a4a;
  }

  /* ── Footer ── */
  footer {
    border-top: 2px solid var(--ink);
    padding-top: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    color: var(--ink-muted);
  }

  footer strong {
    color: var(--ink);
  }

  .footer-pill {
    background: var(--ink);
    color: var(--bg);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
  }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .hero { grid-template-columns: 1fr; }
    .features { grid-template-columns: 1fr; }
    .api-section { grid-template-columns: 1fr; }
    .stats { flex-direction: column; gap: 0; }
    .nav-links a span { display: none; }
  }
</style>
</head>

<body>

<!-- Animated background -->
<div class="bg-canvas">
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="blob blob-3"></div>
</div>
<div class="bg-grid"></div>

<div class="page">

  <!-- Nav -->
  <nav>
    <div class="nav-logo">
      <span class="dot"></span>
      EventEnv
    </div>
    <div class="nav-links">
      <a href="/docs">API <span>Docs</span></a>
      <a href="/tasks" class="filled">View Tasks →</a>
    </div>
  </nav>

  <!-- Hero -->
  <section class="hero">
    <div>
      <div class="hero-badge">OpenEnv Hackathon</div>
      <h1>College Events,<br><em>Intelligently</em><br>Managed.</h1>
      <p class="hero-sub">
        An OpenEnv-compatible AI environment for event registration,
        capacity enforcement, and cancellation — built for agents
        that reason, plan, and act.
      </p>
      <div class="hero-cta">
        <a href="/docs" class="btn btn-primary">Explore API →</a>
        <a href="/tasks" class="btn btn-secondary">View Tasks</a>
      </div>
    </div>
  </section>

  <!-- Stats -->
  <div class="stats">
    <div class="stat">
      <div class="stat-num">REST</div>
      <div class="stat-label">API Interface</div>
    </div>
    <div class="stat">
      <div class="stat-num">FastAPI</div>
      <div class="stat-label">Framework</div>
    </div>
    <div class="stat">
      <div class="stat-num">Graded</div>
      <div class="stat-label">Task System</div>
    </div>
    <div class="stat">
      <div class="stat-num">AI</div>
      <div class="stat-label">Agent Ready</div>
    </div>
  </div>

  <!-- Features -->
  <div class="section-label">What it does</div>
  <div class="features">
    <div class="feature-card">
      <span class="fc-icon">🎓</span>
      <div class="fc-title">Event Management</div>
      <p class="fc-body">Create and manage multiple college events with configurable capacity limits and real-time availability tracking.</p>
    </div>
    <div class="feature-card">
      <span class="fc-icon">🧠</span>
      <div class="fc-title">AI Environment</div>
      <p class="fc-body">Fully OpenEnv-compatible with reward signals, task specifications, and a grader for evaluating agent performance.</p>
    </div>
    <div class="feature-card">
      <span class="fc-icon">📋</span>
      <div class="fc-title">Student Registration</div>
      <p class="fc-body">Register students, prevent duplicates, handle cancellations, and enforce waitlists — all through a clean HTTP API.</p>
    </div>
  </div>

  <!-- API section -->
  <div class="api-section">
    <div>
      <div class="api-title">Clean API.<br>Zero friction.</div>
      <p class="api-desc">
        Every endpoint is designed for agents and humans alike.
        Reset the environment, take actions, and get scored — all over HTTP.
      </p>
      <a href="/docs" class="btn btn-secondary" style="border-color:#4a3a2a; color:#f4f0eb; background: rgba(255,255,255,0.08); box-shadow: 3px 3px 0 rgba(255,255,255,0.15);">
        Open API Explorer →
      </a>
    </div>
    <div class="endpoint-list">
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="endpoint-path">/tasks</span>
        <span class="endpoint-note">List all tasks</span>
      </div>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="endpoint-path">/env/reset</span>
        <span class="endpoint-note">Start episode</span>
      </div>
      <div class="endpoint">
        <span class="method get">GET</span>
        <span class="endpoint-path">/env/state</span>
        <span class="endpoint-note">Current obs.</span>
      </div>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="endpoint-path">/env/step</span>
        <span class="endpoint-note">Take action</span>
      </div>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="endpoint-path">/grader</span>
        <span class="endpoint-note">Score episode</span>
      </div>
      <div class="endpoint">
        <span class="method post">POST</span>
        <span class="endpoint-path">/baseline</span>
        <span class="endpoint-note">Run baseline</span>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <footer>
    <div>Built with <strong>FastAPI</strong> &amp; <strong>OpenEnv</strong></div>
    <div class="footer-pill">Hackathon Project 2025</div>
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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body {{ margin: 0; font-family: 'DM Sans', sans-serif; }}
            .banner {{
                background: #1a1108;
                color: #f4f0eb;
                padding: 16px 32px;
                font-size: 16px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 16px;
                border-bottom: 3px solid #ff6154;
            }}
            .banner-logo {{
                font-family: 'Instrument Serif', serif;
                font-size: 20px;
                color: #f4f0eb;
            }}
            .banner-dot {{
                width: 9px; height: 9px;
                background: #ff6154;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }}
            .banner-tag {{
                margin-left: auto;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #9a8a7a;
                background: rgba(255,255,255,0.06);
                padding: 5px 12px;
                border-radius: 100px;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .banner a {{
                color: #23c4a4;
                text-decoration: none;
                font-size: 13px;
            }}
            .banner a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="banner">
            <span class="banner-logo"><span class="banner-dot"></span>EventEnv</span>
            API Explorer
            <a href="/">← Back to Home</a>
            <span class="banner-tag">OpenEnv Hackathon</span>
        </div>
        {swagger}
    </body>
    </html>
    """)
