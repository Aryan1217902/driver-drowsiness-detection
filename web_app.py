"""FastAPI web application for Driver Drowsiness Detection."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline import PipelineEngine

STATIC_DIR = Path(__file__).parent / "static"

engine = PipelineEngine()


def _create_placeholder_frame(message="Camera feed unavailable"):
    """Create a standardized fallback JPEG frame when webcam has no active frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (60, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    ret, jpeg_buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg_buffer.tobytes() if ret else b""


_PLACEHOLDER_JPEG = _create_placeholder_frame()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage lifecycle of the single camera capture and detection loop."""
    yield
    engine.stop()


app = FastAPI(
    title="Driver Drowsiness Detection API",
    description="Backend API and web interface for Driver Drowsiness Detection",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
def index():
    """Serve the Driver Drowsiness Detection web dashboard."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


@app.get("/api/health")
def health():
    """Return application health status."""
    return {
        "status": "ok",
        "service": "driver-drowsiness-detection",
    }


@app.get("/api/status")
def status():
    """Return latest detection telemetry snapshot."""
    return engine.get_status()


@app.post("/api/start")
def start_monitoring():
    """Start a fresh monitoring session and report initialization failures."""
    success = engine.start()
    if success:
        return {"status": "ok", "message": "Monitoring started"}
    telemetry = engine.get_status()
    return {
        "status": "error",
        "message": telemetry.get("camera_error") or "Monitoring could not be started.",
    }


@app.post("/api/stop")
def stop_monitoring():
    """Stop monitoring and release the active camera and worker resources."""
    engine.stop()
    return {"status": "ok", "message": "Monitoring stopped"}


@app.post("/api/recalibrate")
def recalibrate():
    """Trigger a new calibration cycle on the active detection engine."""
    success = engine.recalibrate()
    return {
        "status": "ok" if success else "error",
        "message": "Recalibration initiated" if success else "Engine not running",
    }


async def _frame_streamer(fps=30):
    """Yield MJPEG multipart chunks of the latest processed frame."""
    delay = 1.0 / fps
    try:
        while True:
            frame_bytes = engine.get_latest_jpeg()
            if frame_bytes is None:
                frame_bytes = _PLACEHOLDER_JPEG

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            await asyncio.sleep(delay)
    except (asyncio.CancelledError, GeneratorExit):
        return


@app.get("/video_feed")
async def video_feed():
    """Stream live annotated webcam frames as multipart/x-mixed-replace MJPEG."""
    return StreamingResponse(
        _frame_streamer(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
