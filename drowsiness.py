"""Detect eye closure that persists for a configured duration."""

import time
from dataclasses import dataclass

from config import CLOSURE_PERSISTENCE_SECONDS, DROWSINESS_DURATION_SECONDS


@dataclass(frozen=True)
class DrowsinessResult:
    closure_state: str
    elapsed_seconds: float
    drowsiness_triggered: bool


class DrowsinessDetector:
    """Track sustained closure using a monotonic clock.

    Both calibrated eyes must be classified CLOSED. The persistence interval
    filters brief classifications; an unavailable eye resets the measurement
    rather than treating it as closed.
    """

    def __init__(
        self,
        duration_threshold=DROWSINESS_DURATION_SECONDS,
        persistence_interval=CLOSURE_PERSISTENCE_SECONDS,
        clock=time.monotonic,
    ):
        if duration_threshold <= 0:
            raise ValueError("duration_threshold must be greater than zero")
        if persistence_interval < 0:
            raise ValueError("persistence_interval cannot be negative")

        self._duration_threshold = duration_threshold
        self._persistence_interval = persistence_interval
        self._clock = clock
        self._candidate_started_at = None
        self._closure_started_at = None

    def update(self, eye_result, face_detected):
        """Update closure state from an EyeDetectionResult and face presence."""
        if not face_detected:
            return self._reset("MEASUREMENT UNAVAILABLE")

        left_status = eye_result.left.status
        right_status = eye_result.right.status
        measurements_are_valid = (
            eye_result.calibrated
            and eye_result.left.ear is not None
            and eye_result.right.ear is not None
            and left_status in ("OPEN", "CLOSED")
            and right_status in ("OPEN", "CLOSED")
        )
        if not measurements_are_valid:
            return self._reset("MEASUREMENT UNAVAILABLE")

        if left_status != "CLOSED" or right_status != "CLOSED":
            return self._reset("EYES OPEN")

        now = self._clock()
        if self._candidate_started_at is None:
            self._candidate_started_at = now

        candidate_duration = now - self._candidate_started_at
        if (
            self._closure_started_at is None
            and candidate_duration < self._persistence_interval
        ):
            return DrowsinessResult(
                closure_state="VERIFYING CLOSURE",
                elapsed_seconds=candidate_duration,
                drowsiness_triggered=False,
            )

        if self._closure_started_at is None:
            # Count from the first closed reading after closure is confirmed.
            self._closure_started_at = self._candidate_started_at

        elapsed = now - self._closure_started_at
        return DrowsinessResult(
            closure_state="EYES CLOSED",
            elapsed_seconds=elapsed,
            drowsiness_triggered=elapsed >= self._duration_threshold,
        )

    def _reset(self, state):
        self._candidate_started_at = None
        self._closure_started_at = None
        return DrowsinessResult(
            closure_state=state,
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
