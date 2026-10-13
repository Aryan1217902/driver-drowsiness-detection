import unittest
from types import SimpleNamespace

from drowsiness import DrowsinessDetector


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_eye_result(left_status, right_status, calibrated=True):
    return SimpleNamespace(
        left=SimpleNamespace(ear=0.3, status=left_status),
        right=SimpleNamespace(ear=0.3, status=right_status),
        calibrated=calibrated,
    )


class DrowsinessDetectorTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.detector = DrowsinessDetector(
            duration_threshold=2.0,
            persistence_interval=0.2,
            clock=self.clock,
        )

    def closed_result(self):
        return make_eye_result("CLOSED", "CLOSED")

    def test_reopening_before_duration_threshold_does_not_trigger(self):
        first = self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(1.0)
        result = self.detector.update(self.closed_result(), face_detected=True)

        self.assertEqual(first.closure_state, "VERIFYING CLOSURE")
        self.assertFalse(result.drowsiness_triggered)
        self.assertAlmostEqual(result.elapsed_seconds, 1.0)

        reopened = self.detector.update(
            make_eye_result("OPEN", "OPEN"),
            face_detected=True,
        )
        self.assertEqual(reopened.closure_state, "EYES OPEN")
        self.assertEqual(reopened.elapsed_seconds, 0.0)
        self.assertFalse(reopened.drowsiness_triggered)

    def test_sustained_closure_triggers_at_duration_threshold(self):
        self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(0.2)
        confirmed = self.detector.update(
            self.closed_result(),
            face_detected=True,
        )
        self.assertEqual(confirmed.closure_state, "EYES CLOSED")
        self.assertAlmostEqual(confirmed.elapsed_seconds, 0.2)
        self.assertFalse(confirmed.drowsiness_triggered)

        self.clock.advance(1.8)
        result = self.detector.update(self.closed_result(), face_detected=True)
        self.assertTrue(result.drowsiness_triggered)
        self.assertAlmostEqual(result.elapsed_seconds, 2.0)

    def test_reopening_resets_timer_before_next_closure(self):
        self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(0.2)
        self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(0.8)
        reopened = self.detector.update(
            make_eye_result("OPEN", "OPEN"),
            face_detected=True,
        )
        self.assertEqual(reopened.elapsed_seconds, 0.0)

        self.clock.advance(0.1)
        restarted = self.detector.update(
            self.closed_result(),
            face_detected=True,
        )
        self.assertEqual(restarted.closure_state, "VERIFYING CLOSURE")
        self.assertAlmostEqual(restarted.elapsed_seconds, 0.0)
        self.assertFalse(restarted.drowsiness_triggered)

    def test_missing_face_or_eye_measurement_resets_timer(self):
        self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(0.5)

        missing_face = self.detector.update(
            self.closed_result(),
            face_detected=False,
        )
        self.assertEqual(
            missing_face.closure_state,
            "MEASUREMENT UNAVAILABLE",
        )
        self.assertEqual(missing_face.elapsed_seconds, 0.0)
        self.assertFalse(missing_face.drowsiness_triggered)

        self.clock.advance(0.5)
        invalid_eye_result = make_eye_result("CLOSED", "CLOSED")
        invalid_eye_result.right.ear = None
        missing_eye = self.detector.update(
            invalid_eye_result,
            face_detected=True,
        )
        self.assertEqual(missing_eye.closure_state, "MEASUREMENT UNAVAILABLE")
        self.assertEqual(missing_eye.elapsed_seconds, 0.0)
        self.assertFalse(missing_eye.drowsiness_triggered)

    def test_brief_noisy_closure_is_filtered_by_persistence_interval(self):
        brief = self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(0.1)
        still_verifying = self.detector.update(
            self.closed_result(),
            face_detected=True,
        )
        self.assertEqual(brief.closure_state, "VERIFYING CLOSURE")
        self.assertEqual(still_verifying.closure_state, "VERIFYING CLOSURE")

        reopened = self.detector.update(
            make_eye_result("OPEN", "OPEN"),
            face_detected=True,
        )
        self.assertEqual(reopened.closure_state, "EYES OPEN")
        self.assertFalse(reopened.drowsiness_triggered)

    def test_single_closed_eye_resets_closure_state(self):
        self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(0.5)

        result = self.detector.update(
            make_eye_result("CLOSED", "OPEN"),
            face_detected=True,
        )

        self.assertEqual(result.closure_state, "EYES OPEN")
        self.assertEqual(result.elapsed_seconds, 0.0)
        self.assertFalse(result.drowsiness_triggered)

    def test_uncalibrated_or_unrecognized_eye_status_is_unavailable(self):
        self.detector.update(self.closed_result(), face_detected=True)
        self.clock.advance(0.5)

        eye_results = (
            make_eye_result("CLOSED", "CLOSED", calibrated=False),
            make_eye_result("CLOSED", "UNAVAILABLE"),
            make_eye_result("UNKNOWN", "CLOSED"),
        )
        for eye_result in eye_results:
            with self.subTest(eye_result=eye_result):
                result = self.detector.update(eye_result, face_detected=True)
                self.assertEqual(
                    result.closure_state,
                    "MEASUREMENT UNAVAILABLE",
                )
                self.assertEqual(result.elapsed_seconds, 0.0)
                self.assertFalse(result.drowsiness_triggered)

    def test_invalid_timing_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duration_threshold"):
            DrowsinessDetector(duration_threshold=0)
        with self.assertRaisesRegex(ValueError, "persistence_interval"):
            DrowsinessDetector(persistence_interval=-0.1)

    def test_zero_persistence_interval_confirms_on_first_closed_reading(self):
        detector = DrowsinessDetector(
            duration_threshold=1.0,
            persistence_interval=0,
            clock=self.clock,
        )

        result = detector.update(self.closed_result(), face_detected=True)

        self.assertEqual(result.closure_state, "EYES CLOSED")
        self.assertEqual(result.elapsed_seconds, 0.0)
        self.assertFalse(result.drowsiness_triggered)


if __name__ == "__main__":
    unittest.main()
