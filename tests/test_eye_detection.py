import math
import unittest
from types import SimpleNamespace

import mediapipe as mp

from config import (
    FALLBACK_EAR_THRESHOLD,
    MIN_CALIBRATION_SAMPLES,
    SMOOTHING_WINDOW_SIZE,
)
from eye_detection import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EyeDetector,
    calculate_ear,
)


def make_face(left_ear=0.3, right_ear=0.3):
    """Create synthetic landmarks with known open-eye EAR values."""
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(468)]

    for indices, ear in (
        (LEFT_EYE_INDICES, left_ear),
        (RIGHT_EYE_INDICES, right_ear),
    ):
        vertical_gap = ear * 10
        points = (
            (10, 10),
            (14, 10 - vertical_gap / 2),
            (16, 10 - vertical_gap / 2),
            (20, 10),
            (16, 10 + vertical_gap / 2),
            (14, 10 + vertical_gap / 2),
        )
        for index, (x, y) in zip(indices, points):
            landmarks[index] = SimpleNamespace(x=x / 100, y=y / 100)

    return SimpleNamespace(landmark=landmarks)


class CalculateEarTests(unittest.TestCase):
    def test_calculates_expected_ear(self):
        face = make_face(left_ear=0.3)

        ear = calculate_ear(face, LEFT_EYE_INDICES, 100, 100)

        self.assertIsNotNone(ear)
        self.assertTrue(math.isclose(ear, 0.3, rel_tol=1e-9))

    def test_returns_none_when_eye_corner_distance_is_zero(self):
        face = make_face()
        for index in LEFT_EYE_INDICES:
            face.landmark[index] = SimpleNamespace(x=0.5, y=0.5)

        self.assertIsNone(calculate_ear(face, LEFT_EYE_INDICES, 100, 100))

    def test_returns_none_when_eye_landmarks_are_missing(self):
        face = SimpleNamespace(landmark=[SimpleNamespace(x=0.1, y=0.1)])

        self.assertIsNone(calculate_ear(face, LEFT_EYE_INDICES, 100, 100))

    def test_returns_none_when_eye_landmark_coordinates_are_invalid(self):
        face = make_face()
        face.landmark[LEFT_EYE_INDICES[0]] = SimpleNamespace(x=None, y=0.1)

        self.assertIsNone(calculate_ear(face, LEFT_EYE_INDICES, 100, 100))

    def test_returns_none_when_eye_landmark_coordinates_are_non_finite(self):
        face = make_face()
        face.landmark[LEFT_EYE_INDICES[0]] = SimpleNamespace(x=float("nan"), y=0.1)

        self.assertIsNone(calculate_ear(face, LEFT_EYE_INDICES, 100, 100))

    def test_returns_none_when_eye_has_no_vertical_aperture(self):
        face = make_face(left_ear=0)

        self.assertIsNone(calculate_ear(face, LEFT_EYE_INDICES, 100, 100))

    def test_selected_indices_belong_to_media_pipe_eye_contours(self):
        face_mesh = mp.solutions.face_mesh
        left_contour = {
            index for edge in face_mesh.FACEMESH_LEFT_EYE for index in edge
        }
        right_contour = {
            index for edge in face_mesh.FACEMESH_RIGHT_EYE for index in edge
        }

        self.assertTrue(set(LEFT_EYE_INDICES).issubset(left_contour))
        self.assertTrue(set(RIGHT_EYE_INDICES).issubset(right_contour))


class EyeDetectorTests(unittest.TestCase):
    def test_calibrates_separate_eye_thresholds(self):
        detector = EyeDetector()
        face = make_face(left_ear=0.4, right_ear=0.3)

        for _ in range(MIN_CALIBRATION_SAMPLES):
            detector.process(face, 100, 100, elapsed_seconds=0)
        result = detector.process(face, 100, 100, elapsed_seconds=5)

        self.assertTrue(result.calibrated)
        self.assertTrue(math.isclose(result.left.threshold, 0.3, rel_tol=1e-9))
        self.assertTrue(math.isclose(result.right.threshold, 0.225, rel_tol=1e-9))
        self.assertEqual(result.left.status, "OPEN")
        self.assertEqual(result.right.status, "OPEN")

    def test_uses_fallback_if_calibration_has_too_few_samples(self):
        detector = EyeDetector()

        result = detector.process(None, 100, 100, elapsed_seconds=5)

        self.assertTrue(result.calibrated)
        self.assertEqual(result.left.threshold, FALLBACK_EAR_THRESHOLD)
        self.assertEqual(result.right.threshold, FALLBACK_EAR_THRESHOLD)
        self.assertEqual(result.left.status, "UNAVAILABLE")
        self.assertIn("fallback", result.calibration_status)

    def test_calibrates_each_eye_independently_when_one_measurement_is_invalid(self):
        detector = EyeDetector()
        face = make_face()
        for _ in range(MIN_CALIBRATION_SAMPLES):
            face.landmark[RIGHT_EYE_INDICES[0]] = SimpleNamespace(
                x=float("nan"), y=0.1
            )
            detector.process(face, 100, 100, elapsed_seconds=0)

        result = detector.process(face, 100, 100, elapsed_seconds=5)

        self.assertEqual(
            detector.sample_counts()["left"],
            MIN_CALIBRATION_SAMPLES + 1,
        )
        self.assertEqual(detector.sample_counts()["right"], 0)
        self.assertIsNotNone(result.left.threshold)
        self.assertEqual(result.right.threshold, FALLBACK_EAR_THRESHOLD)
        self.assertEqual(result.right.status, "UNAVAILABLE")
        self.assertIn("RIGHT", result.calibration_status)

    def test_calibration_stops_collecting_samples_after_completion(self):
        detector = EyeDetector()
        face = make_face()
        for _ in range(MIN_CALIBRATION_SAMPLES):
            detector.process(face, 100, 100, elapsed_seconds=0)
        detector.process(face, 100, 100, elapsed_seconds=5)
        counts_at_calibration = detector.sample_counts()

        detector.process(face, 100, 100, elapsed_seconds=6)

        self.assertEqual(detector.sample_counts(), counts_at_calibration)

    def test_rolling_median_rejects_brief_low_ear_spike(self):
        detector = EyeDetector()
        open_face = make_face(left_ear=0.4, right_ear=0.4)
        for _ in range(MIN_CALIBRATION_SAMPLES):
            detector.process(open_face, 100, 100, elapsed_seconds=0)
        detector.process(open_face, 100, 100, elapsed_seconds=5)
        for _ in range(SMOOTHING_WINDOW_SIZE - 1):
            detector.process(open_face, 100, 100, elapsed_seconds=6)

        result = detector.process(make_face(0.05, 0.4), 100, 100, 6)

        self.assertEqual(result.left.status, "OPEN")
        self.assertEqual(result.right.status, "OPEN")

        for _ in range(SMOOTHING_WINDOW_SIZE):
            result = detector.process(make_face(0.05, 0.05), 100, 100, 7)
        self.assertEqual(result.left.status, "CLOSED")
        self.assertEqual(result.right.status, "CLOSED")

    def test_missing_face_is_unavailable_not_closed(self):
        detector = EyeDetector()
        open_face = make_face()
        for _ in range(MIN_CALIBRATION_SAMPLES):
            detector.process(open_face, 100, 100, elapsed_seconds=0)
        detector.process(open_face, 100, 100, elapsed_seconds=5)

        result = detector.process(None, 100, 100, elapsed_seconds=6)

        self.assertIsNone(result.left.ear)
        self.assertIsNone(result.right.ear)
        self.assertEqual(result.left.status, "UNAVAILABLE")
        self.assertEqual(result.right.status, "UNAVAILABLE")

    def test_missing_eye_landmarks_are_unavailable_not_closed(self):
        detector = EyeDetector()
        open_face = make_face()
        for _ in range(MIN_CALIBRATION_SAMPLES):
            detector.process(open_face, 100, 100, elapsed_seconds=0)
        detector.process(open_face, 100, 100, elapsed_seconds=5)

        result = detector.process(
            SimpleNamespace(landmark=[]),
            100,
            100,
            elapsed_seconds=6,
        )

        self.assertIsNone(result.left.ear)
        self.assertIsNone(result.right.ear)
        self.assertEqual(result.left.status, "UNAVAILABLE")
        self.assertEqual(result.right.status, "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
