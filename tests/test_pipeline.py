"""Unit tests for the reusable PipelineEngine using mocks."""

import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from config import CALIBRATION_SECONDS, MIN_CALIBRATION_SAMPLES
from drowsiness import DrowsinessResult
from eye_detection import EyeDetectionResult, EyeMeasurement, EyeDetector, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from pipeline import PipelineEngine


def make_eye_result(
    left_ear=0.3,
    right_ear=0.3,
    left_status="OPEN",
    right_status="OPEN",
    calibrated=True,
):
    return EyeDetectionResult(
        left=EyeMeasurement(
            ear=left_ear,
            threshold=0.22,
            status=left_status,
            raw_ear=left_ear,
            smoothed_ear=left_ear,
        ),
        right=EyeMeasurement(
            ear=right_ear,
            threshold=0.22,
            status=right_status,
            raw_ear=right_ear,
            smoothed_ear=right_ear,
        ),
        calibration_status="CALIBRATION COMPLETE",
        calibrated=calibrated,
    )


def make_face_landmarks():
    """Return synthetic landmarks with valid EAR for both eyes."""
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
    points = (
        (0.1, 0.5), (0.3, 0.35), (0.7, 0.35),
        (0.9, 0.5), (0.7, 0.65), (0.3, 0.65),
    )
    for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        for index, (x, y) in zip(indices, points):
            landmarks[index] = SimpleNamespace(x=x, y=y)
    return SimpleNamespace(landmark=landmarks)


class PipelineEngineTests(unittest.TestCase):
    def setUp(self):
        self.synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_default_telemetry_structure(self):
        engine = PipelineEngine()
        status = engine.get_status()

        self.assertFalse(status["running"])
        self.assertFalse(status["camera_available"])
        self.assertIsNone(status["camera_error"])
        self.assertIsNone(status["processing_error"])
        self.assertFalse(status["face_detected"])
        self.assertFalse(status["calibrated"])
        self.assertEqual(status["closure_state"], "MEASUREMENT UNAVAILABLE")
        self.assertFalse(status["drowsiness_triggered"])
        self.assertIsNone(engine.get_latest_jpeg())

    def test_process_frame_with_face_detected(self):
        mock_face_detector = Mock()
        mock_face_detector.process.return_value = object()
        mock_eye_detector = Mock()
        mock_eye_detector.process.return_value = make_eye_result(
            left_ear=0.32, right_ear=0.31
        )
        mock_eye_detector.sample_counts.return_value = {"left": 35, "right": 35}
        mock_drowsiness_detector = Mock()
        mock_drowsiness_detector.update.return_value = DrowsinessResult(
            closure_state="EYES OPEN",
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
        mock_alert = Mock()
        mock_alert._event_active = False

        engine = PipelineEngine(
            face_detector_factory=lambda: mock_face_detector,
            eye_detector_factory=lambda: mock_eye_detector,
            drowsiness_detector_factory=lambda: mock_drowsiness_detector,
            alert_factory=lambda: mock_alert,
            draw_status_fn=None,
        )
        # Manually bind components for direct single-frame processing test
        engine._face_detector = mock_face_detector
        engine._eye_detector = mock_eye_detector
        engine._drowsiness_detector = mock_drowsiness_detector
        engine._alert = mock_alert

        processed_frame, telemetry = engine.process_frame(
            self.synthetic_frame, elapsed=2.5
        )

        self.assertIs(processed_frame, self.synthetic_frame)
        self.assertTrue(telemetry["face_detected"])
        self.assertTrue(telemetry["calibrated"])
        self.assertEqual(telemetry["left_eye"]["ear"], 0.32)
        self.assertEqual(telemetry["right_eye"]["ear"], 0.31)
        self.assertTrue(telemetry["face_landmarks_available"])
        self.assertTrue(telemetry["left_eye"]["measurement_available"])
        self.assertEqual(telemetry["left_eye"]["raw_ear"], 0.32)
        self.assertEqual(telemetry["left_eye"]["smoothed_ear"], 0.32)
        self.assertEqual(telemetry["left_eye"]["threshold"], 0.22)
        self.assertEqual(telemetry["left_eye"]["status"], "OPEN")
        self.assertEqual(telemetry["calibration_status"], "CALIBRATION COMPLETE")
        self.assertIsNone(telemetry["processing_error"])
        self.assertEqual(telemetry["closure_state"], "EYES OPEN")
        self.assertFalse(telemetry["drowsiness_triggered"])
        self.assertEqual(telemetry["sample_counts"], {"left": 35, "right": 35})
        mock_face_detector.draw_landmarks.assert_called_once()
        mock_alert.update.assert_called_once_with(False)

    def test_process_frame_without_face(self):
        mock_face_detector = Mock()
        mock_face_detector.process.return_value = None
        mock_eye_detector = Mock()
        mock_eye_detector.process.return_value = make_eye_result(
            left_ear=None, right_ear=None, left_status="UNAVAILABLE", right_status="UNAVAILABLE"
        )
        mock_eye_detector.sample_counts.return_value = {"left": 0, "right": 0}
        mock_drowsiness_detector = Mock()
        mock_drowsiness_detector.update.return_value = DrowsinessResult(
            closure_state="MEASUREMENT UNAVAILABLE",
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
        mock_alert = Mock()

        engine = PipelineEngine(draw_status_fn=None)
        engine._face_detector = mock_face_detector
        engine._eye_detector = mock_eye_detector
        engine._drowsiness_detector = mock_drowsiness_detector
        engine._alert = mock_alert

        _, telemetry = engine.process_frame(self.synthetic_frame, elapsed=1.0)

        self.assertFalse(telemetry["face_detected"])
        self.assertFalse(telemetry["face_landmarks_available"])
        self.assertFalse(telemetry["left_eye"]["measurement_available"])
        self.assertIsNone(telemetry["left_eye"]["raw_ear"])
        self.assertEqual(telemetry["closure_state"], "MEASUREMENT UNAVAILABLE")
        mock_face_detector.draw_landmarks.assert_not_called()

    def test_frame_processing_error_is_reported_safely_and_logged_once(self):
        class QueuedCamera:
            is_opened = True

            def __init__(self):
                self.frames = queue.Queue()

            def read(self):
                frame = self.frames.get(timeout=1)
                return (False, None) if frame is None else (True, frame)

            def release(self):
                self.frames.put(None)

        camera = QueuedCamera()
        face_detector = Mock()
        process_calls = [0]

        def process_face(_frame):
            process_calls[0] += 1
            if process_calls[0] == 1:
                raise RuntimeError("internal path or stack detail")
            return None

        face_detector.process.side_effect = process_face
        eye_detector = Mock()
        eye_detector.process.return_value = make_eye_result()
        eye_detector.sample_counts.return_value = {"left": 30, "right": 30}
        eye_detector.baselines.return_value = {"left": 0.3, "right": 0.3}
        drowsiness_detector = Mock()
        drowsiness_detector.update.return_value = DrowsinessResult(
            closure_state="EYES OPEN",
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
        alert = Mock()
        alert._event_active = False

        engine = PipelineEngine(
            camera_factory=lambda: camera,
            face_detector_factory=lambda: face_detector,
            eye_detector_factory=lambda: eye_detector,
            drowsiness_detector_factory=lambda: drowsiness_detector,
            alert_factory=lambda: alert,
            draw_status_fn=None,
        )
        self.addCleanup(engine.stop)

        from unittest.mock import patch

        with patch("pipeline.logger.exception") as log_exception:
            self.assertTrue(engine.start())
            camera.frames.put(self.synthetic_frame.copy())

            deadline = time.monotonic() + 1
            while engine.get_status()["processing_error"] is None and time.monotonic() < deadline:
                time.sleep(0.005)

            self.assertEqual(
                engine.get_status()["processing_error"],
                "Frame processing failed. Check the server output.",
            )
            self.assertNotIn(
                "internal path",
                engine.get_status()["processing_error"],
            )

            camera.frames.put(self.synthetic_frame.copy())
            deadline = time.monotonic() + 1
            while engine.get_status()["processing_error"] is not None and time.monotonic() < deadline:
                time.sleep(0.005)

            self.assertIsNone(engine.get_status()["processing_error"])
            self.assertIsNotNone(engine.get_latest_jpeg())
            self.assertEqual(log_exception.call_count, 1)

    def test_camera_not_opened_reports_unavailable_without_crashing(self):
        mock_camera = Mock()
        mock_camera.is_opened = False

        engine = PipelineEngine(camera_factory=lambda: mock_camera)
        started = engine.start()

        self.assertFalse(started)
        status = engine.get_status()
        self.assertFalse(status["camera_available"])
        self.assertFalse(status["running"])
        self.assertIn("Could not open", status["camera_error"])
        mock_camera.release.assert_called_once()

    def test_camera_factory_exception_reports_error_cleanly(self):
        mock_factory = Mock(side_effect=RuntimeError("Webcam hardware fault"))

        engine = PipelineEngine(camera_factory=mock_factory)
        started = engine.start()

        self.assertFalse(started)
        status = engine.get_status()
        self.assertFalse(status["camera_available"])
        self.assertIn("Webcam hardware fault", status["camera_error"])

    def test_component_initialization_failure_releases_opened_camera(self):
        mock_camera = Mock()
        mock_camera.is_opened = True

        engine = PipelineEngine(
            camera_factory=lambda: mock_camera,
            face_detector_factory=Mock(side_effect=RuntimeError("Face model missing")),
        )
        started = engine.start()

        self.assertFalse(started)
        mock_camera.release.assert_called_once()
        status = engine.get_status()
        self.assertFalse(status["camera_available"])
        self.assertIn("Face model missing", status["camera_error"])

    def test_start_and_stop_lifecycle_with_mock_camera(self):
        mock_camera = Mock()
        mock_camera.is_opened = True
        mock_camera.read.return_value = (True, self.synthetic_frame.copy())

        mock_face_detector = Mock()
        mock_face_detector.process.return_value = None
        mock_eye_detector = Mock()
        mock_eye_detector.process.return_value = make_eye_result()
        mock_eye_detector.sample_counts.return_value = {"left": 30, "right": 30}
        mock_drowsiness_detector = Mock()
        mock_drowsiness_detector.update.return_value = DrowsinessResult(
            closure_state="EYES OPEN",
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
        mock_alert = Mock()
        mock_alert._event_active = False

        engine = PipelineEngine(
            camera_factory=lambda: mock_camera,
            face_detector_factory=lambda: mock_face_detector,
            eye_detector_factory=lambda: mock_eye_detector,
            drowsiness_detector_factory=lambda: mock_drowsiness_detector,
            alert_factory=lambda: mock_alert,
            draw_status_fn=None,
        )

        started = engine.start()
        self.assertTrue(started)
        self.assertTrue(engine.is_running)

        # Allow worker thread to run at least one capture cycle
        time.sleep(0.08)

        status = engine.get_status()
        self.assertTrue(status["running"])
        self.assertTrue(status["camera_available"])
        self.assertIsNotNone(engine.get_latest_jpeg())

        engine.stop()
        self.assertFalse(engine.is_running)
        self.assertFalse(engine.get_status()["running"])
        mock_camera.release.assert_called_once()
        mock_face_detector.close.assert_called_once()
        mock_alert.close.assert_called_once()
        self.assertIsNone(engine._thread)

    def test_restart_creates_fresh_session_and_releases_each_camera(self):
        cameras = []
        faces = []
        eyes = []
        drowsiness = []
        alerts = []

        def make_camera():
            camera = Mock()
            camera.is_opened = True
            camera.read.return_value = (False, None)
            cameras.append(camera)
            return camera

        def make_component(items):
            component = Mock()
            items.append(component)
            return component

        engine = PipelineEngine(
            camera_factory=make_camera,
            face_detector_factory=lambda: make_component(faces),
            eye_detector_factory=lambda: make_component(eyes),
            drowsiness_detector_factory=lambda: make_component(drowsiness),
            alert_factory=lambda: make_component(alerts),
            draw_status_fn=None,
        )

        self.assertTrue(engine.start())
        time.sleep(0.02)
        engine._latest_jpeg = b"stale frame"
        engine._latest_telemetry["left_eye"]["ear"] = 0.12
        first_thread = engine._thread
        self.assertTrue(engine.stop())
        self.assertFalse(first_thread.is_alive())
        self.assertIsNone(engine.get_latest_jpeg())
        self.assertFalse(engine.get_status()["running"])
        self.assertIsNone(engine.get_status()["left_eye"]["ear"])

        self.assertTrue(engine.start())
        status = engine.get_status()
        self.assertTrue(status["running"])
        self.assertIsNone(status["left_eye"]["ear"])
        self.assertEqual(status["elapsed_seconds"], 0.0)
        engine.stop()

        self.assertEqual(len(cameras), 2)
        self.assertEqual(len(faces), 2)
        self.assertEqual(len(eyes), 2)
        self.assertEqual(len(drowsiness), 2)
        self.assertEqual(len(alerts), 2)
        for camera in cameras:
            camera.release.assert_called_once_with()
        for face in faces:
            face.close.assert_called_once_with()
        for alert in alerts:
            alert.close.assert_called_once_with()


class PipelineEngineRecalibrateTests(unittest.TestCase):
    """Tests for PipelineEngine.recalibrate() and its interaction with the eye detector."""

    def setUp(self):
        self.synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _make_running_engine(self, mock_eye_detector):
        """Helper: create and start an engine wired to a given mock eye detector."""
        mock_camera = Mock()
        mock_camera.is_opened = True
        mock_camera.read.return_value = (True, self.synthetic_frame.copy())

        mock_face_detector = Mock()
        mock_face_detector.process.return_value = None

        mock_drowsiness_detector = Mock()
        mock_drowsiness_detector.update.return_value = DrowsinessResult(
            closure_state="MEASUREMENT UNAVAILABLE",
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
        mock_alert = Mock()
        mock_alert._event_active = False

        engine = PipelineEngine(
            camera_factory=lambda: mock_camera,
            face_detector_factory=lambda: mock_face_detector,
            eye_detector_factory=lambda: mock_eye_detector,
            drowsiness_detector_factory=lambda: mock_drowsiness_detector,
            alert_factory=lambda: mock_alert,
            draw_status_fn=None,
        )
        return engine, mock_camera

    def test_recalibrate_returns_false_when_engine_not_running(self):
        """recalibrate() must return False if the engine has not been started."""
        engine = PipelineEngine()
        self.assertFalse(engine.recalibrate())

    def test_recalibrate_calls_reset_calibration_on_eye_detector(self):
        """recalibrate() must call reset_calibration() on the eye detector when running."""
        mock_eye_detector = Mock()
        mock_eye_detector.process.return_value = make_eye_result()
        mock_eye_detector.sample_counts.return_value = {"left": 0, "right": 0}
        mock_eye_detector.baselines.return_value = {"left": None, "right": None}

        engine, _ = self._make_running_engine(mock_eye_detector)
        started = engine.start()
        self.assertTrue(started)
        time.sleep(0.05)

        result = engine.recalibrate()
        self.assertTrue(result)
        mock_eye_detector.reset_calibration.assert_called_once()

        engine.stop()

    def test_recalibrate_waits_for_in_flight_stale_calibration_frame(self):
        """A stale frame completes before reset; fresh calibration then collects normally."""
        class FakeClock:
            now = 0.0

            def __call__(self):
                return self.now

        class QueuedCamera:
            is_opened = True

            def __init__(self):
                self.frames = queue.Queue()
                self.read_started = threading.Event()
                self.released = False

            def read(self):
                self.read_started.set()
                try:
                    frame = self.frames.get(timeout=0.05)
                except queue.Empty:
                    return False, None
                return (False, None) if frame is None else (True, frame)

            def release(self):
                self.released = True
                self.frames.put(None)

        class RecordingEyeDetector(EyeDetector):
            def __init__(self):
                super().__init__()
                self.results = queue.Queue()

            def process(self, *args, **kwargs):
                result = super().process(*args, **kwargs)
                self.results.put((result, self.sample_counts()))
                return result

        clock = FakeClock()
        camera = QueuedCamera()
        eye_detector = RecordingEyeDetector()
        face_detector = Mock()
        face_detector.process.return_value = make_face_landmarks()
        drowsiness_detector = Mock()
        drowsiness_detector.update.return_value = DrowsinessResult(
            closure_state="MEASUREMENT UNAVAILABLE",
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
        alert = Mock()
        alert._event_active = False

        engine = PipelineEngine(
            camera_factory=lambda: camera,
            face_detector_factory=lambda: face_detector,
            eye_detector_factory=lambda: eye_detector,
            drowsiness_detector_factory=lambda: drowsiness_detector,
            alert_factory=lambda: alert,
            draw_status_fn=None,
            clock=clock,
        )
        self.addCleanup(engine.stop)

        frame_processing_started = threading.Event()
        release_stale_frame = threading.Event()
        recalibration_started = threading.Event()
        recalibration_finished = threading.Event()
        stale_elapsed_values = []
        process_frame = engine.process_frame

        def pause_first_frame(frame, elapsed):
            if not stale_elapsed_values:
                stale_elapsed_values.append(elapsed)
                frame_processing_started.set()
                if not release_stale_frame.wait(timeout=2):
                    raise TimeoutError("test did not release the in-flight frame")
            return process_frame(frame, elapsed)

        engine.process_frame = pause_first_frame
        self.assertTrue(engine.start())
        self.assertTrue(camera.read_started.wait(timeout=1))

        # Let the worker take a five-second-old calibration snapshot, then pause
        # it before EyeDetector.process() consumes that elapsed value.
        clock.now = CALIBRATION_SECONDS + 1
        camera.frames.put(self.synthetic_frame.copy())
        self.assertTrue(frame_processing_started.wait(timeout=1))
        self.assertGreaterEqual(stale_elapsed_values[0], CALIBRATION_SECONDS)

        def recalibrate_in_thread():
            recalibration_started.set()
            engine.recalibrate()
            recalibration_finished.set()

        recalibration_thread = threading.Thread(target=recalibrate_in_thread)
        recalibration_thread.start()
        self.assertTrue(recalibration_started.wait(timeout=1))
        was_blocked_by_frame = not recalibration_finished.wait(timeout=0.05)

        # The stale frame is allowed to finish against the old session. Reset
        # then runs before another frame can enter calibration processing.
        release_stale_frame.set()
        self.assertTrue(recalibration_finished.wait(timeout=1))
        recalibration_thread.join(timeout=1)
        self.assertTrue(was_blocked_by_frame)

        stale_result, stale_counts = eye_detector.results.get(timeout=1)
        self.assertTrue(stale_result.calibrated)
        self.assertEqual(stale_counts, {"left": 1, "right": 1})

        # After reset, a new frame starts calibration at elapsed zero. Collect
        # the minimum valid samples before advancing the five-second clock.
        camera.frames.put(self.synthetic_frame.copy())
        fresh_result, counts = eye_detector.results.get(timeout=1)
        self.assertFalse(fresh_result.calibrated)
        self.assertEqual(counts, {"left": 1, "right": 1})

        for expected_count in range(2, MIN_CALIBRATION_SAMPLES + 1):
            camera.frames.put(self.synthetic_frame.copy())
            result, counts = eye_detector.results.get(timeout=1)
            self.assertFalse(result.calibrated)
            self.assertEqual(counts, {"left": expected_count, "right": expected_count})

        clock.now += CALIBRATION_SECONDS
        camera.frames.put(self.synthetic_frame.copy())
        completed_result, completed_counts = eye_detector.results.get(timeout=1)
        self.assertTrue(completed_result.calibrated)
        self.assertEqual(completed_result.calibration_status, "CALIBRATION COMPLETE")
        self.assertGreaterEqual(completed_counts["left"], MIN_CALIBRATION_SAMPLES)
        self.assertGreaterEqual(completed_counts["right"], MIN_CALIBRATION_SAMPLES)

    def test_recalibrate_resets_calibration_elapsed_seconds_in_telemetry(self):
        """After recalibrate(), calibration_elapsed_seconds should be near zero in telemetry."""
        mock_eye_detector = Mock()
        mock_eye_detector.process.return_value = make_eye_result(calibrated=False)
        mock_eye_detector.sample_counts.return_value = {"left": 0, "right": 0}
        mock_eye_detector.baselines.return_value = {"left": None, "right": None}

        # Use a controllable clock to simulate time passing without real sleeping
        tick = [0.0]

        def fake_clock():
            return tick[0]

        mock_camera = Mock()
        mock_camera.is_opened = True
        mock_camera.read.return_value = (True, self.synthetic_frame.copy())

        mock_face_detector = Mock()
        mock_face_detector.process.return_value = None

        mock_drowsiness_detector = Mock()
        mock_drowsiness_detector.update.return_value = DrowsinessResult(
            closure_state="MEASUREMENT UNAVAILABLE",
            elapsed_seconds=0.0,
            drowsiness_triggered=False,
        )
        mock_alert = Mock()
        mock_alert._event_active = False

        engine = PipelineEngine(
            camera_factory=lambda: mock_camera,
            face_detector_factory=lambda: mock_face_detector,
            eye_detector_factory=lambda: mock_eye_detector,
            drowsiness_detector_factory=lambda: mock_drowsiness_detector,
            alert_factory=lambda: mock_alert,
            draw_status_fn=None,
            clock=fake_clock,
        )

        started = engine.start()
        self.assertTrue(started)

        # Advance clock to simulate 3 seconds elapsed since calibration start
        tick[0] = 3.0
        time.sleep(0.1)  # let worker tick a few frames at t=3.0

        # Recalibrate: _calibration_start_time is reset to the current clock (3.0)
        result = engine.recalibrate()
        self.assertTrue(result)

        # Advance clock slightly past reset — calib_elapsed should be ~0.05
        tick[0] = 3.05
        time.sleep(0.1)

        status = engine.get_status()
        # calibration_elapsed_seconds must be small (near 0), not 3+
        self.assertLess(status.get("calibration_elapsed_seconds", 999), 0.5,
                        "calibration_elapsed_seconds should be near zero after recalibrate()")

        engine.stop()


if __name__ == "__main__":
    unittest.main()
