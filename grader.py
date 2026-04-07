from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tasks import TaskSpec, tasks_by_id


def _active_pairs(registrations: List[Dict[str, Any]]) -> set[Tuple[str, str]]:
    return {(r["student_id"], r["event_id"]) for r in registrations if r.get("status") == "registered"}


def _cancelled_pairs(registrations: List[Dict[str, Any]]) -> set[Tuple[str, str]]:
    return {(r["student_id"], r["event_id"]) for r in registrations if r.get("status") == "cancelled"}


def grade_task(task: TaskSpec, *, events: List[Dict[str, Any]], registrations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a score in [0.0, 1.0] plus a beginner-friendly breakdown.

    The grader is intentionally tolerant:
    - it awards partial credit for meeting parts of the goal
    - it enforces hard constraints like "no duplicates" and "respect capacity"
    """

    goal = task.goal
    active = _active_pairs(registrations)
    cancelled = _cancelled_pairs(registrations)

    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, weight: float, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "weight": weight, "detail": detail})

    # --- Required registrations (primary objective)
    required_regs = goal.get("must_have_registrations", [])
    if required_regs:
        per = 0.6 / float(len(required_regs))
        for req in required_regs:
            pair = (req["student_id"], req["event_id"])
            check(
                name=f"has_registration:{pair[0]}->{pair[1]}",
                ok=pair in active,
                weight=per,
                detail="Student is actively registered for the event.",
            )
    else:
        check(name="has_required_registrations", ok=True, weight=0.6, detail="No required registrations in goal.")

    # --- Required cancellations (only used in hard task)
    required_cancels = goal.get("must_have_cancellations", [])
    if required_cancels:
        per = 0.2 / float(len(required_cancels))
        for req in required_cancels:
            pair = (req["student_id"], req["event_id"])
            check(
                name=f"has_cancellation:{pair[0]}->{pair[1]}",
                ok=pair in cancelled,
                weight=per,
                detail="A cancellation record exists for the student and event.",
            )
    else:
        check(name="required_cancellations", ok=True, weight=0.2, detail="No cancellation required for this task.")

    # --- Must NOT have registrations (avoid incorrect leftover registrations)
    forbidden = goal.get("must_not_have_registrations", [])
    if forbidden:
        per = 0.1 / float(len(forbidden))
        for req in forbidden:
            pair = (req["student_id"], req["event_id"])
            check(
                name=f"must_not_have_registration:{pair[0]}->{pair[1]}",
                ok=pair not in active,
                weight=per,
                detail="Student should not be actively registered for the event.",
            )
    else:
        check(name="forbidden_registrations", ok=True, weight=0.1, detail="No forbidden registrations in goal.")

    # --- Safety constraints: no duplicates + capacity
    # These are "hard constraints" but still graded with a small weight.
    # If violated, the score can still be non-zero but will be reduced.
    # Duplicates
    dup_ok = True
    counts: Dict[Tuple[str, str], int] = {}
    for r in registrations:
        if r.get("status") == "registered":
            k = (r["student_id"], r["event_id"])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] > 1:
                dup_ok = False
                break
    if goal.get("must_not_have_duplicates", False):
        check(
            name="no_duplicate_registrations",
            ok=dup_ok,
            weight=0.05,
            detail="No student should be registered twice for the same event.",
        )
    else:
        check(name="no_duplicate_registrations", ok=True, weight=0.05, detail="Duplicates not checked in this task.")

    # Capacity
    cap_by_id = {e["id"]: int(e["capacity"]) for e in events}
    taken_by_id: Dict[str, int] = {}
    for r in registrations:
        if r.get("status") == "registered":
            taken_by_id[r["event_id"]] = taken_by_id.get(r["event_id"], 0) + 1
    cap_ok = all(taken_by_id.get(eid, 0) <= cap for eid, cap in cap_by_id.items())

    if goal.get("must_respect_capacity", False):
        check(
            name="respect_capacity",
            ok=cap_ok,
            weight=0.05,
            detail="Active registrations must not exceed event capacities.",
        )
    else:
        check(name="respect_capacity", ok=True, weight=0.05, detail="Capacity not checked in this task.")

    # Score
    score = 0.0
    total_weight = sum(c["weight"] for c in checks) or 1.0
    for c in checks:
        if c["ok"]:
            score += c["weight"]
    score = max(0.0, min(1.0, score / total_weight))

    return {"task_id": task.id, "score": score, "checks": checks}


def grade_from_task_id(task_id: str, *, events: List[Dict[str, Any]], registrations: List[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = tasks_by_id()
    if task_id not in tasks:
        raise ValueError("Unknown task_id")
    return grade_task(tasks[task_id], events=events, registrations=registrations)

