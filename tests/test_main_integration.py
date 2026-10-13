import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from alert import DrowsinessAlert as RealDrowsinessAlert
from drowsiness import DrowsinessDetector as RealDrowsinessDetector


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_eye_result(left_status, right_status, calibrated=True, ear=0.3):
    measurement = lambda status: SimpleNamespace(
        ear=ear if status != "UNAVAILABLE" else None,
        threshold=0.2,
        status=status,
    )
    return SimpleNamespace(
        left=measurement(left_status),
        right=measurement(right_status),
        calibration_status="CALIBRATION COMPLETE",
        calibrated=calibrated,
    )


class MainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.frame = SimpleNamespace(shape=(480, 640, 3))
        self.camera = Mock()
        self.camera.is_opened = True
        self.face_detector = Mock()
        self.eye_detector = Mock()
        self.eye_detector.sample_counts.return_value = {"left": 30, "right": 30}
        self.alerts = []
        self.play_calls = []
        self.frame_count = 0
        self.component_constructors = {}

        for target, value in (
            ("main.Camera", self.camera),
            ("main.FaceLandmarkDetector", self.face_detector),
            ("main.EyeDetector", self.eye_detector),
        ):
            constructor = patch(target)
            mocked = constructor.start()
            mocked.return_value = value
            self.component_constructors[target] = mocked
            self.addCleanup(constructor.stop)

        detector_constructor = patch("main.DrowsinessDetector")
        mocked_detector_constructor = detector_constructor.start()
        mocked_detector_constructor.side_effect = lambda: RealDrowsinessDetector(
            clock=self.clock
        )
        self.component_constructors["main.DrowsinessDetector"] = (
            mocked_detector_constructor
        )
        self.addCleanup(detector_constructor.stop)

        alert_constructor = patch("main.DrowsinessAlert")
        mocked_alert_constructor = alert_constructor.start()

        def create_alert():
            alert = RealDrowsinessAlert(
                play_warning=lambda: self.play_calls.append("warning"),
                duration_ms=10,
            )
            self.alerts.append(alert)
            return alert

        mocked_alert_constructor.side_effect = create_alert
        self.component_constructors["main.DrowsinessAlert"] = (
            mocked_alert_constructor
        )
        self.addCleanup(alert_constructor.stop)

        imshow = patch("main.cv2.imshow")
        imshow.start()
        self.addCleanup(imshow.stop)
        draw_status = patch("main._draw_status")
        draw_status.start()
        self.addCleanup(draw_status.stop)
        destroy = patch("main.cv2.destroyAllWindows")
        self.destroy_windows = destroy.start()
        self.addCleanup(destroy.stop)

    def configure_frames(self, faces, eye_results, keys=None):
        self.frame_count = len(faces)
        self.camera.read.side_effect = self.read_frame
        self.face_detector.process.side_effect = faces
        self.eye_detector.process.side_effect = eye_results
        if keys is None:
            keys = [0] * (self.frame_count - 1) + [ord("q")]
        key_wait = patch("main.cv2.waitKey", side_effect=keys)
        self.wait_key = key_wait.start()
        self.addCleanup(key_wait.stop)

    def read_frame(self):
        self.clock.advance(0.25)
        return True, self.frame

    def test_closed_eye_event_reaches_audio_once_then_reopens(self):
        faces = [object()] * 20
        results = [make_eye_result("CLOSED", "CLOSED") for _ in range(11)]
        results.append(make_eye_result("OPEN", "OPEN"))
        results.extend(
            make_eye_result("CLOSED", "CLOSED") for _ in range(8)
        )
        self.configure_frames(faces, results)

        main.main()

        self.assertEqual(self.eye_detector.process.call_count, 20)
        self.assertEqual(self.face_detector.process.call_count, 20)
        self.assertEqual(self.play_calls, ["warning"])
        self.assertTrue(self.wait_key.called)
        self.alerts[0].close()
        self.camera.release.assert_called_once_with()
        self.face_detector.close.assert_called_once_with()
        self.destroy_windows.assert_called_once_with()

    def test_missing_face_resets_closed_eye_timer_without_audio(self):
        faces = [object(), object(), None] + [object()] * 5
        results = [make_eye_result("CLOSED", "CLOSED") for _ in faces]
        self.configure_frames(faces, results)

        main.main()

        self.assertEqual(self.play_calls, [])
        self.eye_detector.process.assert_called()
        self.assertEqual(self.camera.read.call_count, len(faces))
        self.camera.release.assert_called_once_with()
        self.face_detector.close.assert_called_once_with()
        self.alerts[0].close()

    def test_unavailable_eye_measurement_resets_closed_eye_timer(self):
        faces = [object()] * 11
        results = [make_eye_result("CLOSED", "CLOSED") for _ in range(5)]
        results.append(make_eye_result("UNAVAILABLE", "CLOSED"))
        results.extend(
            make_eye_result("CLOSED", "CLOSED") for _ in range(5)
        )
        self.configure_frames(faces, results)

        main.main()

        self.assertEqual(self.play_calls, [])
        self.assertEqual(self.eye_detector.process.call_count, 11)
        self.alerts[0].close()

    def test_keyboard_exit_cleans_all_initialized_components(self):
        self.configure_frames(
            [None],
            [make_eye_result("UNAVAILABLE", "UNAVAILABLE")],
            keys=[ord("q")],
        )

        main.main()

        for constructor in self.component_constructors.values():
            constructor.assert_called_once_with()
        self.camera.release.assert_called_once_with()
        self.face_detector.close.assert_called_once_with()
        self.alerts[0].close()
        self.destroy_windows.assert_called_once_with()

    def test_frame_processing_exception_still_cleans_resources(self):
        self.camera.read.side_effect = self.read_frame
        self.face_detector.process.side_effect = RuntimeError("face processing failed")

        with self.assertRaisesRegex(RuntimeError, "face processing failed"):
            main.main()

        self.camera.release.assert_called_once_with()
        self.face_detector.close.assert_called_once_with()
        self.alerts[0].close()
        self.destroy_windows.assert_called_once_with()

    def test_face_detector_initialization_failure_cleans_prior_resources(self):
        self.face_detector_constructor = patch(
            "main.FaceLandmarkDetector",
            side_effect=RuntimeError("initialization failed"),
        )
        mocked_constructor = self.face_detector_constructor.start()
        self.addCleanup(self.face_detector_constructor.stop)

        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            main.main()

        self.camera.release.assert_called_once_with()
        self.alerts[0].close()
        self.destroy_windows.assert_called_once_with()
        mocked_constructor.assert_called_once_with()

    def test_unopened_camera_is_released_without_initializing_later_components(self):
        self.camera.is_opened = False

        main.main()

        self.camera.release.assert_called_once_with()
        self.face_detector.close.assert_not_called()
        self.assertEqual(self.alerts, [])
        self.destroy_windows.assert_called_once_with()

    def test_failed_camera_read_still_cleans_initialized_components(self):
        self.camera.read.return_value = (False, None)

        main.main()

        self.camera.release.assert_called_once_with()
        self.face_detector.close.assert_called_once_with()
        self.alerts[0].close()
        self.destroy_windows.assert_called_once_with()
        self.face_detector.process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
