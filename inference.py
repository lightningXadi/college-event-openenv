"""
inference.py  –  OpenEnv compatibility shim
--------------------------------------------
Runs all tasks through the deterministic baseline agent and prints
the required [START] / [STEP] / [END] structured output blocks to stdout
so the OpenEnv validator can parse results.

Expected output format:
  [START] task=<task_id>
  [STEP] step=<n> reward=<value>
  [END] task=<task_id> score=<value> steps=<n>
"""

from __future__ import annotations

import sys

from environment import CollegeEventEnv, Action, _read_json
from grader import grade_from_task_id
from tasks import get_tasks


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

    # Task 3 (hard)
    return [
        Action(type="view_events"),
        Action(type="register", student_id="stu_001", event_id="evt_orientation_101"),
        Action(type="register", student_id="stu_002", event_id="evt_ai_workshop"),
        Action(type="cancel", student_id="stu_001", event_id="evt_orientation_101"),
        Action(type="register", student_id="stu_001", event_id="evt_ai_workshop"),
        Action(type="view_registrations"),
    ]


def run_all_tasks():
    env = CollegeEventEnv()
    tasks = get_tasks()

    for task in tasks:
        task_id = task.id

        print(f"[START] task={task_id}", flush=True)

        obs = env.reset(task_id)
        step_num = 0

        for action in _plan_for_task(task_id):
            res = env.step(action)
            step_num += 1
            reward_value = res.reward.value
            print(f"[STEP] step={step_num} reward={reward_value}", flush=True)
            if res.done:
                break

        result = grade_from_task_id(
            task_id,
            events=_read_json("events.json"),
            registrations=_read_json("registrations.json"),
        )
        score = result["score"]

        print(f"[END] task={task_id} score={score} steps={step_num}", flush=True)


if __name__ == "__main__":
    run_all_tasks()
