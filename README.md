# Driver Drowsiness Detection

A modular, Windows-based webcam demonstration that uses MediaPipe Face Mesh
and the Eye Aspect Ratio (EAR) to classify each eye as open or closed. It
calibrates a threshold for each eye, smooths brief frame-to-frame EAR changes,
and detects sustained closure. When a drowsiness event is detected, a short
Windows warning beep plays on a background thread so video processing can
continue.

This is a demonstration project, not a validated safety system. It has not
been validated in real-world driving conditions and must not be relied on to
prevent drowsiness-related incidents.

## Requirements

- Windows with a desktop session (the app uses an OpenCV display window).
- Python 3.10.
- A working webcam connected and available to the app.
- Working audio output (speakers or headphones) to hear the warning; a
  microphone is not required.

The pinned dependencies are MediaPipe, OpenCV with contrib modules, and NumPy.
Install them in a virtual environment from the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the demonstration

From the project directory:

```powershell
.\.venv\Scripts\python.exe main.py
```

The app opens the default webcam (camera index `0`) and displays a video
window. At startup, look toward the camera with both eyes comfortably open
and keep your head neutral during the five-second calibration. Press **Q**
while the video window is focused to exit. The app releases the camera,
closes the face detector, waits for any active warning beep, and closes its
OpenCV windows on exit.

If the webcam does not open, check that it is connected, that Windows camera
privacy settings allow desktop apps to use it, and that another application
is not already using it. If the video window does not appear, run the app in
a normal interactive Windows desktop rather than a headless terminal/session.
If no warning is audible, check the Windows output device and volume.

## Detection and presentation settings

Values are in `config.py`:

- `CAMERA_INDEX = 0`: default webcam device index.
- `CALIBRATION_SECONDS = 5` and `MIN_CALIBRATION_SAMPLES = 30`: calibration
  duration and minimum valid samples needed per eye.
- `THRESHOLD_BASELINE_RATIO = 0.75`: calibrated threshold as a fraction of
  each eye's median calibration EAR.
- `FALLBACK_EAR_THRESHOLD = 0.21`: threshold used for an eye without enough
  valid calibration samples; the display indicates fallback calibration.
- `SMOOTHING_WINDOW_SIZE = 5`: rolling EAR median window.
- `CLOSURE_PERSISTENCE_SECONDS = 0.2`: filters brief closure classifications.
- `DROWSINESS_DURATION_SECONDS = 2.0`: both calibrated eyes must remain
  classified closed for this duration to trigger an event.
- `ALERT_FREQUENCY_HZ = 1000` and `ALERT_DURATION_MS = 600`: beep settings.
  The warning is asynchronous and is not repeated during one continuous
  event; reopening resets the event so a later closure can trigger another.

If calibration is incomplete, keep facing forward with eyes open during the
calibration period and ensure the face is clearly visible. Head turns can
distort the 2D EAR measurement; head-pose estimation is not included.

## Tests

Run the automated suite from the project directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover EAR and calibration, drowsiness state transitions, alert
handling, mocked camera behavior, and mocked application-flow integration.
Automated tests do not require a physical camera or speaker.

## Project files

- `main.py` coordinates capture, face landmarks, eye measurements, drowsiness,
  alerts, and the OpenCV display.
- `config.py` stores application, calibration, audio, and display settings.
- `camera.py` opens, mirrors, reads, and releases webcam frames.
- `face_detection.py` wraps MediaPipe Face Mesh processing and visualization.
- `eye_detection.py` calculates EAR and manages calibration and smoothing.
- `drowsiness.py` detects sustained eye closure.
- `alert.py` plays one asynchronous Windows warning per drowsiness event.
- `tests/test_eye_detection.py` tests EAR and calibration.
- `tests/test_drowsiness.py` tests drowsiness timer state transitions.
- `tests/test_alert.py` tests warning event and worker behavior.
- `tests/test_camera.py` tests camera behavior with mocked OpenCV calls.
- `tests/test_main_integration.py` tests application flow and cleanup with
  mocked hardware-dependent components.
