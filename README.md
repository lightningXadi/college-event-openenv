# College Event Registration System (OpenEnv-Compatible)

A simple **OpenEnv-compatible environment** for a **college event registration system** built using **FastAPI**.

The environment simulates common operations in event management such as viewing events, registering students, preventing duplicate registrations, cancelling registrations, and handling capacity limits.

This project exposes **OpenEnv-style APIs** so agents can interact with the environment using `reset`, `step`, and `state`.

---

# Architecture

The system consists of a FastAPI-based environment that exposes OpenEnv-style APIs.

```
Agent / Client
      │
      ▼
FastAPI Server (app.py)
      │
      ▼
Environment Logic (environment.py)
      │
      ▼
JSON Data Storage
(events.json, students.json, registrations.json)
```

---

# Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic
- OpenAPI / Swagger UI
- Docker (containerization)

# Features

- View available college events
- Register students for events
- Prevent duplicate registrations
- Cancel registrations
- Enforce event capacity limits
- Built-in grading system
- Deterministic baseline agent
- Interactive API documentation with Swagger UI

---

# Folder Structure

```
project/
├── app.py
├── environment.py
├── tasks.py
├── grader.py
├── baseline.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── data/
    ├── events.json
    ├── students.json
    └── registrations.json
```

### File Description

**app.py**  
Main FastAPI application that exposes API endpoints.

**environment.py**  
Core environment logic that manages events, registrations, and rewards.

**tasks.py**  
Defines the tasks used for evaluation.

**grader.py**  
Grades the final environment state and returns a score.

**baseline.py**  
Runs a deterministic baseline agent that solves the tasks.

**openenv.yaml**  
OpenEnv environment specification.

**data/**  
Stores environment data:

- `events.json` → event details and capacity  
- `students.json` → student list  
- `registrations.json` → active registrations  

---

# Installation

### Windows

```
cd project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```
cd project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# Run the Server

Start the FastAPI server:

```
uvicorn app:app --reload --port 8000
```

Open in browser:

API Docs  
```
http://localhost:8000/docs
```

Tasks  
```
http://localhost:8000/tasks
```

Health Check  
```
http://localhost:8000/health
```

---

# OpenEnv API

### Reset Environment

```
POST /env/reset
```

Example request:

```json
{
  "task_id": "college_event_task_1"
}
```

---

### Perform Action

```
POST /env/step
```

Example:

```json
{
  "type": "register",
  "student_id": "stu_001",
  "event_id": "evt_orientation_101"
}
```

---

### Get Environment State

```
GET /env/state
```

Returns current events, registrations, and environment messages.

---

# Tasks

The environment contains **three tasks with increasing difficulty**.

### Task 1 (Easy)

Register **stu_001** for **evt_orientation_101**

### Task 2 (Medium)

Register **stu_001** and **stu_002** for **evt_ai_workshop** while preventing duplicate registrations.

### Task 3 (Hard)

Manage multiple events, cancellations, and event capacity constraints.

---

# Grading

Evaluate a task using:

```
POST /grader
```

Example:

```
curl -X POST http://localhost:8000/grader \
-H "Content-Type: application/json" \
-d "{\"task_id\":\"college_event_task_1\"}"
```

The grader returns a **score between 0.0 and 1.0**.

---

# Run Baseline Agent

Run the baseline agent:

```
python baseline.py --base-url http://127.0.0.1:8000
```

This automatically solves all tasks and prints the results.

---

# Docker

Build Docker image:

```
docker build -t college-event-openenv .
```

Run container:

```
docker run -p 8000:8000 college-event-openenv
```

---

# License

MIT License

### Health Check

GET /health

Returns the health status of the environment.

Example:
http://localhost:8000/health


# Example Agent Workflow

1. Reset environment

POST /env/reset
{
  "task_id": "college_event_task_1"
}

2. View available events

POST /env/step
{
  "type": "view_events"
}

3. Register student

POST /env/step
{
  "type": "register",
  "student_id": "stu_001",
  "event_id": "evt_orientation_101"
}
