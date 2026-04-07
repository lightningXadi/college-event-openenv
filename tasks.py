from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class TaskSpec:
    """
    Defines a task (episode) for the environment.

    Each task contains:
    - initial_data: what JSON files should look like after reset()
    - goal: what must be true at the end to get full credit in the grader
    - max_steps: a safety cap to prevent infinite loops
    """

    id: str
    title: str
    difficulty: Difficulty
    description: str
    max_steps: int
    initial_data: Dict[str, Any]
    goal: Dict[str, Any]


def get_tasks() -> List[TaskSpec]:
    """
    Three tasks of increasing difficulty:

    - Task 1 (Easy): Register a student for an event.
    - Task 2 (Medium): Register multiple students and prevent duplicates.
    - Task 3 (Hard): Multiple events + registration + cancellation + capacity constraints.
    """

    base_events = [
        {
            "id": "evt_orientation_101",
            "title": "Freshers Orientation 101",
            "location": "Main Auditorium",
            "starts_at": "2026-09-01T10:00:00",
            "capacity": 2,
        },
        {
            "id": "evt_ai_workshop",
            "title": "AI Club: Intro Workshop",
            "location": "CS Lab 2",
            "starts_at": "2026-09-03T16:00:00",
            "capacity": 3,
        },
        {
            "id": "evt_career_fair",
            "title": "Campus Career Fair",
            "location": "Sports Complex",
            "starts_at": "2026-09-10T09:00:00",
            "capacity": 4,
        },
    ]

    base_students = [
        {"id": "stu_001", "name": "Aisha Khan", "email": "aisha.khan@college.edu"},
        {"id": "stu_002", "name": "Ben Carter", "email": "ben.carter@college.edu"},
        {"id": "stu_003", "name": "Chen Wei", "email": "chen.wei@college.edu"},
    ]

    # --------------------------- Task 1 (Easy)
    task1 = TaskSpec(
        id="college_event_task_1",
        title="Task 1 (Easy): Register one student",
        difficulty="easy",
        description=(
            "Register student stu_001 for event evt_orientation_101. "
            "The agent should first view events, then register."
        ),
        max_steps=8,
        initial_data={
            "events": base_events,
            "students": base_students,
            "registrations": [],
        },
        goal={
            "must_have_registrations": [{"student_id": "stu_001", "event_id": "evt_orientation_101"}],
            "must_not_have_duplicates": True,
            "must_respect_capacity": True,
        },
    )

    # --------------------------- Task 2 (Medium)
    task2 = TaskSpec(
        id="college_event_task_2",
        title="Task 2 (Medium): Register multiple + prevent duplicates",
        difficulty="medium",
        description=(
            "Register stu_001 and stu_002 for evt_ai_workshop. "
            "A duplicate registration attempt must be rejected (and penalized)."
        ),
        max_steps=10,
        initial_data={
            "events": base_events,
            "students": base_students,
            "registrations": [],
        },
        goal={
            "must_have_registrations": [
                {"student_id": "stu_001", "event_id": "evt_ai_workshop"},
                {"student_id": "stu_002", "event_id": "evt_ai_workshop"},
            ],
            "must_not_have_duplicates": True,
            "must_respect_capacity": True,
        },
    )

    # --------------------------- Task 3 (Hard)
    # Make capacity tight to force correctness around cancellations/capacity.
    hard_events = [
        {**base_events[0], "capacity": 1},  # orientation has only 1 seat
        {**base_events[1], "capacity": 2},  # workshop has only 2 seats
        base_events[2],
    ]
    task3 = TaskSpec(
        id="college_event_task_3",
        title="Task 3 (Hard): Multiple events + cancellations + capacity",
        difficulty="hard",
        description=(
            "Correctly manage multiple events and registrations: "
            "1) Register stu_001 for evt_orientation_101. "
            "2) Register stu_002 for evt_ai_workshop. "
            "3) Cancel stu_001 from evt_orientation_101. "
            "4) Register stu_001 for evt_ai_workshop (capacity is tight)."
        ),
        max_steps=14,
        initial_data={
            "events": hard_events,
            "students": base_students,
            "registrations": [],
        },
        goal={
            "must_have_registrations": [
                {"student_id": "stu_002", "event_id": "evt_ai_workshop"},
                {"student_id": "stu_001", "event_id": "evt_ai_workshop"},
            ],
            "must_have_cancellations": [{"student_id": "stu_001", "event_id": "evt_orientation_101"}],
            "must_not_have_registrations": [{"student_id": "stu_001", "event_id": "evt_orientation_101"}],
            "must_not_have_duplicates": True,
            "must_respect_capacity": True,
        },
    )

    return [task1, task2, task3]


def tasks_by_id() -> Dict[str, TaskSpec]:
    return {t.id: t for t in get_tasks()}

