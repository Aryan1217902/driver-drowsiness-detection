"""Unit tests for the FastAPI web application, telemetry, MJPEG video streaming, and dashboard."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from web_app import (
    _PLACEHOLDER_JPEG,
    _frame_streamer,
    app,
    health,
    index,
    start_monitoring,
    status,
    stop_monitoring,
    video_feed,
)


def request_asgi(app, path="/api/health", method="GET"):
    """Execute an in-memory HTTP request against the ASGI app without external dependencies."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }

    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))

    start = next((m for m in messages if m["type"] == "http.response.start"), None)
    body_part = next((m for m in messages if m["type"] == "http.response.body"), None)

    status_code = start["status"] if start else None
    headers = dict(start["headers"]) if start else {}
    body_bytes = body_part["body"] if body_part else b""
    json_data = None
    try:
        json_data = json.loads(body_bytes.decode("utf-8")) if body_bytes else None
    except Exception:
        pass

    return status_code, headers, body_bytes, json_data


class WebAppTests(unittest.TestCase):
    """Test health, telemetry status, video streaming, and dashboard endpoints."""

    def test_health_direct_call_returns_ok(self):
        """Test health function directly."""
        result = health()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["service"], "driver-drowsiness-detection")

    def test_health_endpoint_via_asgi_returns_200(self):
        """Test GET /api/health over ASGI pipeline."""
        status_code, headers, _, json_data = request_asgi(app, "/api/health")
        self.assertEqual(status_code, 200)
        self.assertEqual(headers.get(b"content-type"), b"application/json")
        self.assertIsNotNone(json_data)
        self.assertEqual(json_data.get("status"), "ok")
        self.assertEqual(json_data.get("service"), "driver-drowsiness-detection")

    def test_status_endpoint_direct_call(self):
        """Test status function directly."""
        data = status()
        self.assertIn("camera_available", data)
        self.assertIn("running", data)
        self.assertIn("closure_state", data)

    def test_status_endpoint_via_asgi_returns_telemetry(self):
        """Test GET /api/status over ASGI pipeline."""
        status_code, headers, _, json_data = request_asgi(app, "/api/status")
        self.assertEqual(status_code, 200)
        self.assertEqual(headers.get(b"content-type"), b"application/json")
        self.assertIsNotNone(json_data)

        for key in (
            "camera_available",
            "camera_error",
            "processing_error",
            "running",
            "face_detected",
            "face_landmarks_available",
            "calibrated",
            "calibration_status",
            "sample_counts",
            "left_eye",
            "right_eye",
            "closure_state",
            "closure_duration",
            "drowsiness_triggered",
            "alert_active",
            "elapsed_seconds",
            "fps",
        ):
            self.assertIn(key, json_data)

    def test_status_endpoint_reflects_engine_state(self):
        """Test that GET /api/status accurately reflects custom engine telemetry."""
        custom_telemetry = {
            "camera_available": True,
            "running": True,
            "face_detected": True,
            "face_landmarks_available": True,
            "calibrated": True,
            "calibration_status": "CALIBRATION COMPLETE",
            "left_eye": {"ear": 0.35, "threshold": 0.25, "status": "OPEN"},
            "right_eye": {"ear": 0.34, "threshold": 0.25, "status": "OPEN"},
            "closure_state": "EYES OPEN",
            "closure_duration": 0.0,
            "drowsiness_triggered": False,
        }
        with patch("web_app.engine.get_status", return_value=custom_telemetry):
            status_code, _, _, json_data = request_asgi(app, "/api/status")
            self.assertEqual(status_code, 200)
            self.assertEqual(json_data["left_eye"]["ear"], 0.35)
            self.assertEqual(json_data["closure_state"], "EYES OPEN")

    def test_video_feed_route_is_registered(self):
        """Verify GET /video_feed is properly registered in the FastAPI app."""
        route = next((r for r in app.routes if getattr(r, "path", None) == "/video_feed"), None)
        self.assertIsNotNone(route)
        self.assertIn("GET", route.methods)

    def test_video_feed_response_configuration_and_headers(self):
        """Verify video_feed response returns 200, multipart/x-mixed-replace, and cache headers."""
        response = asyncio.run(video_feed())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.media_type,
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.assertEqual(
            response.headers.get("cache-control"),
            "no-cache, no-store, must-revalidate",
        )
        self.assertEqual(response.headers.get("pragma"), "no-cache")
        self.assertEqual(response.headers.get("expires"), "0")

        # Safely close generator stream
        asyncio.run(response.body_iterator.aclose())

    def test_frame_streamer_yields_placeholder_when_no_frame(self):
        """Verify _frame_streamer yields fallback JPEG when engine has no frame."""
        async def verify():
            with patch("web_app.engine.get_latest_jpeg", return_value=None):
                gen = _frame_streamer(fps=100)
                chunk = await anext(gen)
                await gen.aclose()
                self.assertTrue(chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"))
                self.assertTrue(chunk.endswith(b"\r\n"))
                self.assertIn(_PLACEHOLDER_JPEG, chunk)

        asyncio.run(verify())

    def test_frame_streamer_yields_annotated_jpeg_frame(self):
        """Verify _frame_streamer yields custom annotated JPEG bytes when available."""
        fake_jpeg = b"\xff\xd8\xff\xe0fake_jpeg_content\xff\xd9"

        async def verify():
            with patch("web_app.engine.get_latest_jpeg", return_value=fake_jpeg):
                gen = _frame_streamer(fps=100)
                chunk = await anext(gen)
                await gen.aclose()
                self.assertTrue(chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"))
                self.assertTrue(chunk.endswith(b"\r\n"))
                self.assertIn(fake_jpeg, chunk)

        asyncio.run(verify())

    def test_video_feed_client_disconnect_handling(self):
        """Verify that client disconnection cleanly closes the frame generator without errors."""
        async def verify_disconnect():
            gen = _frame_streamer(fps=100)
            _ = await anext(gen)
            # Simulate client disconnection
            await gen.aclose()

        try:
            asyncio.run(verify_disconnect())
        except Exception as exc:
            self.fail(f"Client disconnect raised unexpected exception: {exc}")

    def test_dashboard_index_route_serves_html(self):
        """Verify GET / returns 200 OK and serves the frontend dashboard HTML with required components."""
        status_code, headers, body_bytes, _ = request_asgi(app, "/")
        self.assertEqual(status_code, 200)
        content_type = headers.get(b"content-type", b"").decode("utf-8")
        self.assertIn("text/html", content_type)

        html_text = body_bytes.decode("utf-8")
        # Check essential UI element IDs specified in dashboard requirements
        for required_id in (
            'id="drowsinessBanner"',
            'id="videoFeed"',
            'id="closureStateText"',
            'id="closureDurationText"',
            'id="leftEyeCard"',
            'id="leftEarValue"',
            'id="leftRawEarValue"',
            'id="leftMeasurementState"',
            'id="rightEyeCard"',
            'id="rightEarValue"',
            'id="rightRawEarValue"',
            'id="rightMeasurementState"',
            'id="landmarksState"',
            'id="pipelineState"',
            'id="audioState"',
            'id="videoFeedMessage"',
            'id="calibrationBadge"',
            'id="calibrationFill"',
            'id="cameraHealthPill"',
            'id="faceStatusPill"',
            'id="backendHealthPill"',
            'id="startMonitoringBtn"',
            'id="stopMonitoringBtn"',
            'id="monitoringMessage"',
        ):
            self.assertIn(required_id, html_text)

        script_text = (Path(__file__).parent.parent / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('addEventListener("click", () => requestMonitoringAction("/api/start"))', script_text)
        self.assertIn('addEventListener("click", () => requestMonitoringAction("/api/stop"))', script_text)
        self.assertIn("renderMonitoringControls(Boolean(data.running))", script_text)
        self.assertIn("recalibrateBtn.disabled = !monitoringRunning", script_text)
        self.assertIn("pollPending", script_text)
        self.assertIn("AbortController", script_text)
        self.assertIn("VIDEO_STALE_AFTER_MS", script_text)
        self.assertIn("data.processing_error", script_text)
        self.assertIn('result.message || "Recalibration initiated."', script_text)

    def test_static_assets_exist_on_disk(self):
        """Verify static asset files are present on disk."""
        static_dir = Path(__file__).parent.parent / "static"
        self.assertTrue((static_dir / "index.html").is_file())
        self.assertTrue((static_dir / "style.css").is_file())
        self.assertTrue((static_dir / "app.js").is_file())

    def test_nonexistent_endpoint_returns_404(self):
        """Test that unknown routes return 404 Not Found."""
        status_code, _, _, json_data = request_asgi(app, "/api/nonexistent")
        self.assertEqual(status_code, 404)
        self.assertEqual(json_data.get("detail"), "Not Found")



class RecalibrateEndpointTests(unittest.TestCase):
    """Tests for POST /api/recalibrate endpoint."""

    def test_recalibrate_endpoint_is_registered(self):
        """POST /api/recalibrate must be a registered route."""
        route = next(
            (r for r in app.routes if getattr(r, "path", None) == "/api/recalibrate"),
            None,
        )
        self.assertIsNotNone(route, "/api/recalibrate route not found")
        self.assertIn("POST", route.methods)

    def test_recalibrate_endpoint_returns_ok_when_engine_running(self):
        """POST /api/recalibrate returns status=ok when engine.recalibrate() succeeds."""
        with patch("web_app.engine.recalibrate", return_value=True):
            status_code, headers, _, json_data = request_asgi(
                app, "/api/recalibrate", method="POST"
            )
        self.assertEqual(status_code, 200)
        self.assertIsNotNone(json_data)
        self.assertEqual(json_data.get("status"), "ok")
        self.assertEqual(json_data.get("message"), "Recalibration initiated")

    def test_recalibrate_endpoint_returns_error_when_engine_not_running(self):
        """POST /api/recalibrate returns status=error when engine.recalibrate() fails."""
        with patch("web_app.engine.recalibrate", return_value=False):
            status_code, _, _, json_data = request_asgi(
                app, "/api/recalibrate", method="POST"
            )
        self.assertEqual(status_code, 200)
        self.assertEqual(json_data.get("status"), "error")
        self.assertEqual(json_data.get("message"), "Engine not running")

    def test_status_includes_baseline_ear_fields(self):
        """GET /api/status must include baseline EAR fields in left_eye and right_eye."""
        custom_telemetry = {
            "camera_available": True,
            "camera_error": None,
            "running": True,
            "face_detected": True,
            "face_landmarks_available": True,
            "calibrated": True,
            "calibration_status": "CALIBRATION COMPLETE",
            "calibration_elapsed_seconds": 5.1,
            "sample_counts": {"left": 35, "right": 35},
            "left_eye": {
                "ear": 0.33,
                "raw_ear": 0.34,
                "smoothed_ear": 0.33,
                "measurement_available": True,
                "threshold": 0.25,
                "baseline": 0.33,
                "status": "OPEN",
            },
            "right_eye": {
                "ear": 0.32,
                "raw_ear": 0.31,
                "smoothed_ear": 0.32,
                "measurement_available": True,
                "threshold": 0.24,
                "baseline": 0.32,
                "status": "OPEN",
            },
            "closure_state": "EYES OPEN",
            "closure_duration": 0.0,
            "drowsiness_triggered": False,
            "alert_active": False,
            "elapsed_seconds": 10.0,
            "fps": 28.0,
        }
        with patch("web_app.engine.get_status", return_value=custom_telemetry):
            status_code, _, _, json_data = request_asgi(app, "/api/status")

        self.assertEqual(status_code, 200)
        self.assertIn("baseline", json_data["left_eye"],
                      "left_eye must contain 'baseline' key")
        self.assertIn("baseline", json_data["right_eye"],
                      "right_eye must contain 'baseline' key")
        self.assertEqual(json_data["left_eye"]["baseline"], 0.33)
        self.assertEqual(json_data["right_eye"]["baseline"], 0.32)
        self.assertTrue(json_data["face_landmarks_available"])
        self.assertEqual(json_data["left_eye"]["raw_ear"], 0.34)
        self.assertEqual(json_data["left_eye"]["smoothed_ear"], 0.33)
        self.assertEqual(json_data["left_eye"]["threshold"], 0.25)
        self.assertEqual(json_data["left_eye"]["status"], "OPEN")
        self.assertTrue(json_data["left_eye"]["measurement_available"])
        self.assertEqual(json_data["right_eye"]["raw_ear"], 0.31)
        self.assertEqual(json_data["right_eye"]["smoothed_ear"], 0.32)
        self.assertEqual(json_data["right_eye"]["threshold"], 0.24)
        self.assertEqual(json_data["right_eye"]["status"], "OPEN")
        self.assertTrue(json_data["right_eye"]["measurement_available"])
        self.assertEqual(json_data["sample_counts"], {"left": 35, "right": 35})
        self.assertEqual(json_data["calibration_status"], "CALIBRATION COMPLETE")
        self.assertIn("calibration_elapsed_seconds", json_data)


class MonitoringEndpointTests(unittest.TestCase):
    """Tests for the dashboard monitoring lifecycle endpoints."""

    def test_start_and_stop_routes_are_registered(self):
        for path in ("/api/start", "/api/stop"):
            route = next(
                (r for r in app.routes if getattr(r, "path", None) == path), None
            )
            self.assertIsNotNone(route, f"{path} route not found")
            self.assertIn("POST", route.methods)

    def test_start_endpoint_reports_success_and_failure(self):
        with patch("web_app.engine.start", return_value=True) as start:
            status_code, _, _, data = request_asgi(app, "/api/start", method="POST")
        self.assertEqual(status_code, 200)
        self.assertEqual(data, {"status": "ok", "message": "Monitoring started"})
        start.assert_called_once_with()

        with patch("web_app.engine.start", return_value=False), patch(
            "web_app.engine.get_status",
            return_value={"camera_error": "Could not open webcam"},
        ):
            status_code, _, _, data = request_asgi(app, "/api/start", method="POST")
        self.assertEqual(status_code, 200)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "Could not open webcam")

    def test_stop_endpoint_stops_engine_and_returns_success(self):
        with patch("web_app.engine.stop") as stop:
            status_code, _, _, data = request_asgi(app, "/api/stop", method="POST")
        self.assertEqual(status_code, 200)
        self.assertEqual(data, {"status": "ok", "message": "Monitoring stopped"})
        stop.assert_called_once_with()

    def test_recalibrate_endpoint_still_requires_running_engine(self):
        with patch("web_app.engine.recalibrate", return_value=False) as recalibrate:
            _, _, _, data = request_asgi(app, "/api/recalibrate", method="POST")
        recalibrate.assert_called_once_with()
        self.assertEqual(data["status"], "error")


if __name__ == "__main__":
    unittest.main()
