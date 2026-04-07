from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from tasks import TaskSpec, tasks_by_id


# ---------------------------------------------------------------------------
# Pydantic models (Observation / Action / Reward)
# ---------------------------------------------------------------------------

ActionType = Literal[
    "view_events",
    "view_event",
    "view_registrations",
    "register",
    "cancel",
]


class Action(BaseModel):
    """
    A structured action an agent can take.

    Notes:
    - For register/cancel you must provide both student_id and event_id.
    - For view_event you must provide event_id.
    """

    type: ActionType
    student_id: Optional[str] = None
    event_id: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str


class EventView(BaseModel):
    id: str
    title: str
    location: str
    starts_at: str
    capacity: int
    seats_taken: int
    seats_left: int


class RegistrationView(BaseModel):
    student_id: str
    event_id: str
    status: Literal["registered", "cancelled"]
    created_at: str


class Observation(BaseModel):
    """
    What the agent observes after reset() and each step().
    """

    episode_id: str
    task_id: str
    task_title: str
    step_count: int
    max_steps: int
    done: bool
    last_action: Optional[Action] = None
    messages: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    events: List[EventView] = Field(default_factory=list)
    registrations: List[RegistrationView] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Local JSON storage helpers
# ---------------------------------------------------------------------------


def _data_dir() -> str:
    # data/ lives next to this file
    return os.path.join(os.path.dirname(__file__), "data")


def _read_json(filename: str) -> Any:
    path = os.path.join(_data_dir(), filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(filename: str, data: Any) -> None:
    path = os.path.join(_data_dir(), filename)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _now_iso() -> str:
    # ISO-ish but short; good enough for hackathon storage/logs
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeCounters:
    invalid_actions: int = 0
    repeat_actions: int = 0


class CollegeEventEnv:
    """
    OpenEnv-compatible environment for a College Event Registration System.

    Core API:
    - reset(task_id) -> Observation
    - step(action)   -> StepResult
    - state()        -> Observation  (current state snapshot)
    """

    def __init__(self) -> None:
        self._episode_id: str = "not_started"
        self._task: Optional[TaskSpec] = None
        self._step_count: int = 0
        self._done: bool = True
        self._last_action_key: Optional[str] = None
        self._counters = _EpisodeCounters()

    # ----------------------- OpenEnv API

    def reset(self, task_id: str) -> Observation:
        tasks = tasks_by_id()
        if task_id not in tasks:
            raise ValueError("Unknown task_id")

        self._task = tasks[task_id]
        self._episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        self._step_count = 0
        self._done = False
        self._last_action_key = None
        self._counters = _EpisodeCounters()

        # Re-initialize the JSON storage for a clean, reproducible episode.
        init = self._task.initial_data
        _write_json_atomic("events.json", init["events"])
        _write_json_atomic("students.json", init["students"])
        _write_json_atomic("registrations.json", init["registrations"])

        obs = self._observe(messages=["Environment reset. Use view_events to see what’s available."])
        return obs

    def state(self) -> Observation:
        if self._task is None:
            # Provide a stable "not started" state
            return Observation(
                episode_id=self._episode_id,
                task_id="",
                task_title="",
                step_count=self._step_count,
                max_steps=0,
                done=True,
                messages=["No active episode. Call reset(task_id)."],
                events=[],
                registrations=[],
            )
        return self._observe()

    def step(self, action: Action) -> StepResult:
        if self._task is None or self._done:
            obs = self.state()
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, reason="Episode is done or not started. Call reset()."),
                done=True,
                info={"error": "done_or_not_started"},
            )

        self._step_count += 1

        if self._step_count > self._task.max_steps:
            self._done = True
            obs = self._observe(last_action=action, errors=["Max steps exceeded. Episode ended."])
            return StepResult(
                observation=obs,
                reward=Reward(value=0.0, reason="Max steps exceeded."),
                done=True,
                info={"terminated": "max_steps"},
            )

        # Loop prevention: repeated identical action key is penalized.
        action_key = f"{action.type}:{action.student_id or ''}:{action.event_id or ''}"
        if self._last_action_key == action_key:
            self._counters.repeat_actions += 1
        else:
            self._counters.repeat_actions = 0
        self._last_action_key = action_key

        try:
            messages, errors = self._apply_action(action)
        except ValueError as e:
            self._counters.invalid_actions += 1
            errors = [str(e)]
            messages = []

        # If too many invalid actions, end the episode (anti-destructive safeguard).
        if self._counters.invalid_actions >= 4:
            self._done = True
            errors = list(errors) + ["Too many invalid actions. Episode ended."]

        # Reward is shaped by task progress + safety penalties, always normalized to [0, 1].
        reward_value, reward_reason = self._compute_reward(action, errors)

        # End episode early if goal is already satisfied.
        if not self._done and self._task is not None:
            if self._is_goal_satisfied()[0]:
                self._done = True
                messages = list(messages) + ["Task goal satisfied. Episode complete."]

        obs = self._observe(last_action=action, messages=messages, errors=errors)
        return StepResult(
            observation=obs,
            reward=Reward(value=reward_value, reason=reward_reason),
            done=self._done,
            info={
                "invalid_actions": self._counters.invalid_actions,
                "repeat_actions": self._counters.repeat_actions,
            },
        )

    # ----------------------- Internals

    def _observe(
        self,
        *,
        last_action: Optional[Action] = None,
        messages: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
    ) -> Observation:
        assert self._task is not None
        events = self._list_events_view()
        regs = self._list_registrations_view()
        return Observation(
            episode_id=self._episode_id,
            task_id=self._task.id,
            task_title=self._task.title,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            done=self._done,
            last_action=last_action,
            messages=messages or [],
            errors=errors or [],
            events=events,
            registrations=regs,
        )

    def _list_events_view(self) -> List[EventView]:
        events = _read_json("events.json")
        regs = _read_json("registrations.json")

        # Seats taken only counts active registrations.
        taken_by_event: Dict[str, int] = {}
        for r in regs:
            if r.get("status") == "registered":
                taken_by_event[r["event_id"]] = taken_by_event.get(r["event_id"], 0) + 1

        out: List[EventView] = []
        for e in events:
            seats_taken = int(taken_by_event.get(e["id"], 0))
            cap = int(e["capacity"])
            out.append(
                EventView(
                    id=e["id"],
                    title=e["title"],
                    location=e["location"],
                    starts_at=e["starts_at"],
                    capacity=cap,
                    seats_taken=seats_taken,
                    seats_left=max(0, cap - seats_taken),
                )
            )
        return out

    def _list_registrations_view(self) -> List[RegistrationView]:
        regs = _read_json("registrations.json")
        out: List[RegistrationView] = []
        for r in regs:
            out.append(
                RegistrationView(
                    student_id=r["student_id"],
                    event_id=r["event_id"],
                    status=r["status"],
                    created_at=r["created_at"],
                )
            )
        return out

    def _apply_action(self, action: Action) -> Tuple[List[str], List[str]]:
        if action.type == "view_events":
            return (["Listed available events."], [])

        if action.type == "view_event":
            if not action.event_id:
                raise ValueError("view_event requires event_id")
            events = _read_json("events.json")
            if not any(e["id"] == action.event_id for e in events):
                raise ValueError("Invalid event_id")
            return ([f"Viewed event {action.event_id}."], [])

        if action.type == "view_registrations":
            return (["Listed registrations."], [])

        if action.type in ("register", "cancel"):
            if not action.student_id or not action.event_id:
                raise ValueError(f"{action.type} requires student_id and event_id")

            events = _read_json("events.json")
            students = _read_json("students.json")
            regs = _read_json("registrations.json")

            if not any(e["id"] == action.event_id for e in events):
                raise ValueError("Invalid event_id")
            if not any(s["id"] == action.student_id for s in students):
                raise ValueError("Invalid student_id")

            # Normalize to "active registration exists?"
            def is_active(r: Dict[str, Any]) -> bool:
                return (
                    r.get("student_id") == action.student_id
                    and r.get("event_id") == action.event_id
                    and r.get("status") == "registered"
                )

            if action.type == "register":
                if any(is_active(r) for r in regs):
                    raise ValueError("Duplicate registration is not allowed")

                # Capacity check.
                event = next(e for e in events if e["id"] == action.event_id)
                capacity = int(event["capacity"])
                taken = sum(1 for r in regs if r.get("event_id") == action.event_id and r.get("status") == "registered")
                if taken >= capacity:
                    raise ValueError("Event is full (capacity reached)")

                regs.append(
                    {
                        "student_id": action.student_id,
                        "event_id": action.event_id,
                        "status": "registered",
                        "created_at": _now_iso(),
                    }
                )
                _write_json_atomic("registrations.json", regs)
                return ([f"Registered {action.student_id} for {action.event_id}."], [])

            # cancel
            if not any(is_active(r) for r in regs):
                raise ValueError("Cannot cancel: no active registration found")

            # Mark the active registration as cancelled (append-only log is clearer for beginners).
            # We keep historical rows; active seat count only includes status == "registered".
            for r in regs:
                if is_active(r):
                    r["status"] = "cancelled"
                    break
            _write_json_atomic("registrations.json", regs)
            return ([f"Cancelled registration: {action.student_id} from {action.event_id}."], [])

        raise ValueError("Unknown action type")

    def _compute_reward(self, action: Action, errors: List[str]) -> Tuple[float, str]:
        """
        Reward is always within [0.0, 1.0].

        High-level shaping:
        - Valid helpful actions get partial credit.
        - Invalid actions get 0.0 and also trigger termination after repeated misuse.
        - Repeating the same action is discouraged via a small decay.
        - Completing the goal gives 1.0.
        """

        if self._task is None:
            return (0.0, "No active task.")

        if errors:
            return (0.0, "Invalid action: " + "; ".join(errors))

        goal_ok, goal_note = self._is_goal_satisfied()
        if goal_ok:
            return (1.0, "Goal satisfied: " + goal_note)

        # Small penalty for repeats (loop prevention), but never below 0.
        repeat_decay = 0.08 * min(5, self._counters.repeat_actions)

        # Give reasonable intermediate rewards to guide agents.
        if action.type in ("view_events", "view_event", "view_registrations"):
            base = 0.15
            return (max(0.0, base - repeat_decay), "Useful inspection action.")

        if action.type == "register":
            base = 0.55
            return (max(0.0, base - repeat_decay), "Successful registration.")

        if action.type == "cancel":
            base = 0.45
            return (max(0.0, base - repeat_decay), "Successful cancellation.")

        return (max(0.0, 0.1 - repeat_decay), "Action applied.")

    def _is_goal_satisfied(self) -> Tuple[bool, str]:
        assert self._task is not None
        goal = self._task.goal
        regs = _read_json("registrations.json")
        events = _read_json("events.json")

        active_pairs = {(r["student_id"], r["event_id"]) for r in regs if r.get("status") == "registered"}
        cancelled_pairs = {(r["student_id"], r["event_id"]) for r in regs if r.get("status") == "cancelled"}

        # Must have registrations
        for pair in goal.get("must_have_registrations", []):
            if (pair["student_id"], pair["event_id"]) not in active_pairs:
                return (False, "Missing required registration(s).")

        # Must have cancellations
        for pair in goal.get("must_have_cancellations", []):
            if (pair["student_id"], pair["event_id"]) not in cancelled_pairs:
                return (False, "Missing required cancellation(s).")

        # Must not have registrations
        for pair in goal.get("must_not_have_registrations", []):
            if (pair["student_id"], pair["event_id"]) in active_pairs:
                return (False, "Has a registration that should not exist.")

        # No duplicates (same student/event registered more than once)
        if goal.get("must_not_have_duplicates", False):
            counts: Dict[Tuple[str, str], int] = {}
            for r in regs:
                if r.get("status") == "registered":
                    k = (r["student_id"], r["event_id"])
                    counts[k] = counts.get(k, 0) + 1
                    if counts[k] > 1:
                        return (False, "Duplicate active registrations detected.")

        # Respect capacity for every event
        if goal.get("must_respect_capacity", False):
            cap_by_id = {e["id"]: int(e["capacity"]) for e in events}
            taken_by_id: Dict[str, int] = {}
            for r in regs:
                if r.get("status") == "registered":
                    taken_by_id[r["event_id"]] = taken_by_id.get(r["event_id"], 0) + 1
            for eid, taken in taken_by_id.items():
                if taken > cap_by_id.get(eid, 0):
                    return (False, "Capacity constraint violated.")

        return (True, "All goal constraints satisfied.")

