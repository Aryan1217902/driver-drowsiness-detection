"""Reusable detection pipeline engine coordinating DDD modules for web and desktop."""

import logging
import threading
import time

import cv2

from alert import DrowsinessAlert
from camera import Camera
from drowsiness import DrowsinessDetector
from eye_detection import EyeDetector
from face_detection import FaceLandmarkDetector
from main import _draw_status

logger = logging.getLogger(__name__)


def _default_telemetry(running=False, camera_available=False, camera_error=None):
    """Return default structured telemetry when engine is uninitialized or idle."""
    return {
        "camera_available": camera_available,
        "camera_error": camera_error,
        "processing_error": None,
        "running": running,
        "face_detected": False,
        "face_landmarks_available": False,
        "calibrated": False,
        "calibration_status": "NOT STARTED",
        "sample_counts": {"left": 0, "right": 0},
        "left_eye": {
            "ear": None,
            "raw_ear": None,
            "smoothed_ear": None,
            "measurement_available": False,
            "threshold": None,
            "baseline": None,
            "status": "UNAVAILABLE",
        },
        "right_eye": {
            "ear": None,
            "raw_ear": None,
            "smoothed_ear": None,
            "measurement_available": False,
            "threshold": None,
            "baseline": None,
            "status": "UNAVAILABLE",
        },
        "closure_state": "MEASUREMENT UNAVAILABLE",
        "closure_duration": 0.0,
        "drowsiness_triggered": False,
        "alert_active": False,
        "elapsed_seconds": 0.0,
        "calibration_elapsed_seconds": 0.0,
        "fps": 0.0,
    }


class PipelineEngine:
    """Coordinates camera capture, detection modules, and thread-safe telemetry storage."""

    def __init__(
        self,
        camera_factory=Camera,
        face_detector_factory=FaceLandmarkDetector,
        eye_detector_factory=EyeDetector,
        drowsiness_detector_factory=DrowsinessDetector,
        alert_factory=DrowsinessAlert,
        draw_status_fn=_draw_status,
        clock=time.monotonic,
    ):
        self._camera_factory = camera_factory
        self._face_detector_factory = face_detector_factory
        self._eye_detector_factory = eye_detector_factory
        self._drowsiness_detector_factory = drowsiness_detector_factory
        self._alert_factory = alert_factory
        self._draw_status_fn = draw_status_fn
        self._clock = clock

        self._lock = threading.Lock()
        self._calibration_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._running = threading.Event()
        self._thread = None
        self._start_time = None
        self._calibration_start_time = None

        self._camera = None
        self._face_detector = None
        self._eye_detector = None
        self._drowsiness_detector = None
        self._alert = None

        self._fps = 0.0
        self._latest_jpeg = None
        self._latest_telemetry = _default_telemetry()
        self._processing_error_logged = False

    @property
    def is_running(self):
        """Return True if background capture thread is actively running."""
        return self._running.is_set()

    def process_frame(self, frame, elapsed):
        """Process a single frame through all detection stages and return annotated frame + telemetry."""
        face_landmarks = self._face_detector.process(frame)
        if face_landmarks is not None:
            self._face_detector.draw_landmarks(frame, face_landmarks)

        frame_height, frame_width, _ = frame.shape
        eye_result = self._eye_detector.process(
            face_landmarks,
            frame_width,
            frame_height,
            elapsed,
        )

        drowsiness_result = self._drowsiness_detector.update(
            eye_result,
            face_detected=face_landmarks is not None,
        )

        self._alert.update(drowsiness_result.drowsiness_triggered)
        sample_counts = self._eye_detector.sample_counts()

        if self._draw_status_fn is not None:
            self._draw_status_fn(
                frame,
                face_landmarks,
                eye_result,
                sample_counts,
                elapsed,
                drowsiness_result,
            )

        baselines = (
            self._eye_detector.baselines()
            if hasattr(self._eye_detector, "baselines")
            else {"left": None, "right": None}
        )

        telemetry = {
            "camera_available": True,
            "camera_error": None,
            "processing_error": None,
            "running": True,
            "face_detected": face_landmarks is not None,
            "face_landmarks_available": face_landmarks is not None,
            "calibrated": eye_result.calibrated,
            "calibration_status": eye_result.calibration_status,
            "sample_counts": sample_counts,
            "left_eye": {
                "ear": eye_result.left.ear,
                "raw_ear": getattr(eye_result.left, "raw_ear", eye_result.left.ear),
                "smoothed_ear": getattr(eye_result.left, "smoothed_ear", None),
                "measurement_available": getattr(
                    eye_result.left, "raw_ear", eye_result.left.ear
                ) is not None,
                "threshold": eye_result.left.threshold,
                "baseline": baselines.get("left"),
                "status": eye_result.left.status,
            },
            "right_eye": {
                "ear": eye_result.right.ear,
                "raw_ear": getattr(eye_result.right, "raw_ear", eye_result.right.ear),
                "smoothed_ear": getattr(eye_result.right, "smoothed_ear", None),
                "measurement_available": getattr(
                    eye_result.right, "raw_ear", eye_result.right.ear
                ) is not None,
                "threshold": eye_result.right.threshold,
                "baseline": baselines.get("right"),
                "status": eye_result.right.status,
            },
            "closure_state": drowsiness_result.closure_state,
            "closure_duration": round(drowsiness_result.elapsed_seconds, 2),
            "drowsiness_triggered": drowsiness_result.drowsiness_triggered,
            "alert_active": getattr(self._alert, "_event_active", False),
            "elapsed_seconds": round(elapsed, 2),
            "calibration_elapsed_seconds": round(elapsed, 2),
            "fps": self._fps,
        }

        return frame, telemetry

    def start(self):
        """Safely initialize detectors and camera, then start the single processing loop."""
        with self._lifecycle_lock:
            with self._lock:
                if self._running.is_set():
                    return True
                # Each start is a new detection and calibration session.
                self._latest_telemetry = _default_telemetry()
                self._latest_jpeg = None
                self._fps = 0.0
                self._processing_error_logged = False

            try:
                camera = self._camera_factory()
                if not getattr(camera, "is_opened", False):
                    try:
                        camera.release()
                    except Exception:
                        pass
                    with self._lock:
                        self._latest_telemetry = _default_telemetry(
                            running=False,
                            camera_available=False,
                            camera_error="Could not open the default webcam. Ensure a camera is connected.",
                        )
                    return False
                self._camera = camera
            except Exception as exc:
                with self._lock:
                    self._latest_telemetry = _default_telemetry(
                        running=False,
                        camera_available=False,
                        camera_error=f"Camera initialization failed: {exc}",
                    )
                return False

            try:
                self._alert = self._alert_factory()
                self._face_detector = self._face_detector_factory()
                self._eye_detector = self._eye_detector_factory()
                self._drowsiness_detector = self._drowsiness_detector_factory()
            except Exception as exc:
                self._cleanup_components()
                with self._lock:
                    self._latest_telemetry = _default_telemetry(
                        running=False,
                        camera_available=False,
                        camera_error=f"Component initialization failed: {exc}",
                    )
                return False

            self._running.set()
            with self._lock:
                self._latest_telemetry["running"] = True
                self._latest_telemetry["camera_available"] = True
                self._latest_telemetry["camera_error"] = None

            self._thread = threading.Thread(
                target=self._run_loop,
                name="ddd-pipeline-worker",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception:
                self._running.clear()
                self._thread = None
                self._cleanup_components()
                with self._lock:
                    self._latest_telemetry = _default_telemetry(
                        camera_error="Could not start the monitoring worker thread.",
                    )
                return False
            return True

    def stop(self, timeout=2.0):
        """Stop the worker and release the camera and detection components."""
        with self._lifecycle_lock:
            if not self._running.is_set():
                try:
                    self._cleanup_components()
                except Exception:
                    pass
                return False

            self._running.clear()
            thread = self._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive() and self._camera is not None:
                    # Releasing the capture can unblock a backend stuck in read().
                    try:
                        self._camera.release()
                    except Exception:
                        pass
                    thread.join()
            self._thread = None
            self._cleanup_components()
            with self._lock:
                self._latest_telemetry = _default_telemetry()
                self._latest_jpeg = None
                self._fps = 0.0
            return True

    def recalibrate(self):
        """Safely reset calibration and restart the calibration countdown."""
        with self._calibration_lock:
            with self._lock:
                if not self._running.is_set():
                    return False
                self._calibration_start_time = self._clock()
                if self._eye_detector is not None and hasattr(self._eye_detector, "reset_calibration"):
                    self._eye_detector.reset_calibration()
                return True

    def get_status(self):
        """Return a thread-safe snapshot copy of current telemetry."""
        with self._lock:
            return dict(self._latest_telemetry)

    def get_latest_jpeg(self):
        """Return the latest encoded JPEG frame bytes, or None if unavailable."""
        with self._lock:
            return self._latest_jpeg

    def _run_loop(self):
        """Background loop reading from camera and processing frames."""
        with self._lock:
            self._start_time = self._clock()
            self._calibration_start_time = self._start_time
            start_time = self._start_time

        frame_count = 0
        fps_timer = start_time

        while self._running.is_set():
            try:
                success, frame = self._camera.read()
            except Exception as exc:
                with self._lock:
                    self._latest_telemetry["camera_available"] = False
                    self._latest_telemetry["camera_error"] = f"Error reading frame: {exc}"
                time.sleep(0.05)
                continue

            if not success or frame is None:
                with self._lock:
                    self._latest_telemetry["camera_available"] = False
                    self._latest_telemetry["camera_error"] = "Could not read frame from webcam"
                time.sleep(0.05)
                continue

            # Keep the elapsed-time snapshot, eye-detector update, and telemetry
            # publication atomic with respect to a recalibration reset.
            with self._calibration_lock:
                now = self._clock()
                session_elapsed = now - start_time
                with self._lock:
                    calib_start = self._calibration_start_time
                if calib_start is None:
                    calib_start = start_time
                calib_elapsed = now - calib_start

                frame_count += 1
                if now - fps_timer >= 1.0:
                    self._fps = round(frame_count / (now - fps_timer), 1)
                    frame_count = 0
                    fps_timer = now

                try:
                    annotated_frame, telemetry = self.process_frame(frame, calib_elapsed)
                    telemetry["elapsed_seconds"] = round(session_elapsed, 2)
                    telemetry["calibration_elapsed_seconds"] = round(calib_elapsed, 2)

                    ret, jpeg_buf = cv2.imencode(
                        ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if not ret:
                        raise RuntimeError("JPEG frame encoding failed")
                    jpeg_bytes = jpeg_buf.tobytes()

                    with self._lock:
                        self._latest_telemetry = telemetry
                        if jpeg_bytes is not None:
                            self._latest_jpeg = jpeg_bytes
                    self._processing_error_logged = False
                except Exception:
                    # Keep the worker alive, expose a safe message, and log once
                    # per consecutive failure period until a frame succeeds.
                    with self._lock:
                        self._latest_telemetry["processing_error"] = (
                            "Frame processing failed. Check the server output."
                        )
                    if not self._processing_error_logged:
                        logger.exception("DDD frame processing failed")
                        self._processing_error_logged = True
                    time.sleep(0.05)

    def _cleanup_components(self):
        """Safely release and close all active hardware and detector resources."""
        try:
            if self._alert is not None:
                self._alert.close()
        except Exception:
            pass
        finally:
            try:
                if self._face_detector is not None:
                    self._face_detector.close()
            except Exception:
                pass
            finally:
                try:
                    if self._camera is not None:
                        self._camera.release()
                except Exception:
                    pass
                finally:
                    self._alert = None
                    self._face_detector = None
                    self._camera = None
                    self._eye_detector = None
                    self._drowsiness_detector = None
