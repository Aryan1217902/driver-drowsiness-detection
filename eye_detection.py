"""Eye Aspect Ratio calculation, calibration, and classification."""

from collections import deque
from dataclasses import dataclass
import threading

import numpy as np

from config import (
    CALIBRATION_SECONDS,
    FALLBACK_EAR_THRESHOLD,
    MIN_CALIBRATION_SAMPLES,
    SMOOTHING_WINDOW_SIZE,
    THRESHOLD_BASELINE_RATIO,
)

# These six-point subsets belong to FACEMESH_LEFT_EYE and FACEMESH_RIGHT_EYE.
LEFT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
EYES = ("left", "right")


@dataclass(frozen=True)
class EyeMeasurement:
    ear: float | None
    threshold: float | None
    status: str
    raw_ear: float | None = None
    smoothed_ear: float | None = None


@dataclass(frozen=True)
class EyeDetectionResult:
    left: EyeMeasurement
    right: EyeMeasurement
    calibration_status: str
    calibrated: bool


def calculate_ear(face_landmarks, eye_indices, frame_width, frame_height):
    """Calculate EAR, or return None when the eye landmarks are unreliable."""
    try:
        points = np.array(
            [
                (
                    face_landmarks.landmark[index].x * frame_width,
                    face_landmarks.landmark[index].y * frame_height,
                )
                for index in eye_indices
            ],
            dtype=np.float64,
        )
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None

    # EAR is the sum of two vertical eye distances divided by twice
    # the horizontal corner-to-corner distance.
    vertical_distance_1 = np.linalg.norm(points[1] - points[5])
    vertical_distance_2 = np.linalg.norm(points[2] - points[4])
    horizontal_distance = np.linalg.norm(points[0] - points[3])

    if (
        not np.all(np.isfinite(points))
        or not np.isfinite(horizontal_distance)
        or horizontal_distance <= 1e-6
    ):
        return None

    ear = (vertical_distance_1 + vertical_distance_2) / (
        2.0 * horizontal_distance
    )
    return float(ear) if np.isfinite(ear) and ear > 0 else None


class EyeDetector:
    """Maintain per-eye calibration and smoothed OPEN/CLOSED status."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples = {eye: [] for eye in EYES}
        self._thresholds = {eye: None for eye in EYES}
        self._baselines = {eye: None for eye in EYES}
        self._history = {
            eye: deque(maxlen=SMOOTHING_WINDOW_SIZE) for eye in EYES
        }
        self._used_fallback = {eye: False for eye in EYES}
        self._calibrated = False

    def reset_calibration(self):
        """Reset calibration samples, thresholds, and history to start a new cycle."""
        with self._lock:
            self._samples = {eye: [] for eye in EYES}
            self._thresholds = {eye: None for eye in EYES}
            self._baselines = {eye: None for eye in EYES}
            self._history = {
                eye: deque(maxlen=SMOOTHING_WINDOW_SIZE) for eye in EYES
            }
            self._used_fallback = {eye: False for eye in EYES}
            self._calibrated = False

    def baselines(self):
        """Return the calibrated baseline EAR for each eye, or None if uncalibrated/fallback."""
        with self._lock:
            return dict(self._baselines)

    def process(
        self,
        face_landmarks,
        frame_width,
        frame_height,
        elapsed_seconds,
    ):
        """Update calibration and return measurements for the current frame."""
        indices = {
            "left": LEFT_EYE_INDICES,
            "right": RIGHT_EYE_INDICES,
        }
        current_ears = {
            eye: (
                calculate_ear(
                    face_landmarks,
                    indices[eye],
                    frame_width,
                    frame_height,
                )
                if face_landmarks is not None
                else None
            )
            for eye in EYES
        }

        with self._lock:
            if not self._calibrated and face_landmarks is not None:
                for eye in EYES:
                    ear = current_ears[eye]
                    if ear is not None:
                        self._samples[eye].append(ear)

            if not self._calibrated and elapsed_seconds >= CALIBRATION_SECONDS:
                self._finish_calibration()

            if face_landmarks is None:
                for history in self._history.values():
                    history.clear()

            measurements = {}
            for eye in EYES:
                ear = current_ears[eye]

                if not self._calibrated:
                    measurements[eye] = EyeMeasurement(
                        ear=ear,
                        threshold=None,
                        status="CALIBRATING",
                        raw_ear=ear,
                    )
                elif ear is None:
                    self._history[eye].clear()
                    measurements[eye] = EyeMeasurement(
                        ear=None,
                        threshold=self._thresholds[eye],
                        status="UNAVAILABLE",
                        raw_ear=None,
                        smoothed_ear=None,
                    )
                else:
                    # The median of a short rolling window reduces brief spikes.
                    history = self._history[eye]
                    history.append(ear)
                    smoothed_ear = float(np.median(history))
                    status = (
                        "CLOSED"
                        if smoothed_ear < self._thresholds[eye]
                        else "OPEN"
                    )
                    measurements[eye] = EyeMeasurement(
                        ear=smoothed_ear,
                        threshold=self._thresholds[eye],
                        status=status,
                        raw_ear=ear,
                        smoothed_ear=smoothed_ear,
                    )

            return EyeDetectionResult(
                left=measurements["left"],
                right=measurements["right"],
                calibration_status=self._calibration_status(),
                calibrated=self._calibrated,
            )

    def _finish_calibration(self):
        for eye in EYES:
            samples = self._samples[eye]
            if len(samples) >= MIN_CALIBRATION_SAMPLES:
                # Median baseline resists occasional landmark noise.
                baseline = float(np.median(samples))
                self._baselines[eye] = baseline
                self._thresholds[eye] = baseline * THRESHOLD_BASELINE_RATIO
            else:
                self._baselines[eye] = None
                self._thresholds[eye] = FALLBACK_EAR_THRESHOLD
                self._used_fallback[eye] = True
        self._calibrated = True

    def _calibration_status(self):
        if not self._calibrated:
            return "CALIBRATING"
        eyes_using_fallback = [
            eye.upper() for eye in EYES if self._used_fallback[eye]
        ]
        if eyes_using_fallback:
            return (
                "CALIBRATION INCOMPLETE - fallback for "
                + ", ".join(eyes_using_fallback)
            )
        return "CALIBRATION COMPLETE"

    def sample_counts(self):
        """Return the number of valid calibration samples for each eye."""
        with self._lock:
            return {eye: len(self._samples[eye]) for eye in EYES}
