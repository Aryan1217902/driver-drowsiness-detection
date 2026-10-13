"""Application entry point for Driver Drowsiness Detection."""

import time
from math import ceil

import cv2

from alert import DrowsinessAlert
from camera import Camera
from config import (
    CALIBRATION_SECONDS,
    CLOSURE_PERSISTENCE_SECONDS,
    DROWSINESS_DURATION_SECONDS,
    STATUS_ERROR_COLOR,
    STATUS_TEXT_COLOR,
    WINDOW_TITLE,
)
from drowsiness import DrowsinessDetector
from eye_detection import EyeDetector
from face_detection import FaceLandmarkDetector


def _draw_status(
    frame, face_landmarks, eye_result, sample_counts, elapsed, drowsiness_result
):
    face_status = (
        "FACE DETECTED" if face_landmarks is not None else "NO FACE DETECTED"
    )
    face_color = (0, 200, 0) if face_landmarks is not None else STATUS_ERROR_COLOR
    cv2.putText(
        frame,
        face_status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        face_color,
        2,
    )

    calibration_text = eye_result.calibration_status
    if eye_result.calibration_status.startswith("CALIBRATION INCOMPLETE"):
        calibration_color = STATUS_ERROR_COLOR
    elif eye_result.calibrated:
        calibration_color = (0, 200, 0)
    else:
        calibration_color = STATUS_TEXT_COLOR
    if not eye_result.calibrated:
        remaining = max(
            0, ceil(CALIBRATION_SECONDS - elapsed)
        )
        calibration_text = (
            "CALIBRATING: eyes open, face forward, neutral head "
            f"({remaining}s)"
        )

    cv2.putText(
        frame,
        calibration_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        calibration_color,
        2,
    )
    if not eye_result.calibrated:
        cv2.putText(
            frame,
            f"Valid samples L/R: {sample_counts['left']}/{sample_counts['right']}",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            STATUS_TEXT_COLOR,
            1,
        )

    for y, label, measurement in (
        (112, "Left", eye_result.left),
        (142, "Right", eye_result.right),
    ):
        ear_text = (
            f"{measurement.ear:.2f}" if measurement.ear is not None else "N/A"
        )
        threshold_text = (
            f"{measurement.threshold:.2f}"
            if measurement.threshold is not None
            else "..."
        )
        text = (
            f"{label} eye EAR: {ear_text} | Threshold: {threshold_text} | "
            f"{measurement.status}"
        )
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            STATUS_TEXT_COLOR,
            2,
        )

    if drowsiness_result.drowsiness_triggered:
        drowsiness_text = (
            "DROWSINESS DETECTED - eyes closed "
            f"{drowsiness_result.elapsed_seconds:.1f}s"
        )
        drowsiness_color = STATUS_ERROR_COLOR
    elif drowsiness_result.closure_state == "VERIFYING CLOSURE":
        drowsiness_text = (
            f"VERIFYING CLOSURE: {drowsiness_result.elapsed_seconds:.1f}s / "
            f"{CLOSURE_PERSISTENCE_SECONDS:.1f}s"
        )
        drowsiness_color = STATUS_TEXT_COLOR
    elif drowsiness_result.closure_state == "EYES CLOSED":
        drowsiness_text = (
            f"EYES CLOSED: {drowsiness_result.elapsed_seconds:.1f}s / "
            f"{DROWSINESS_DURATION_SECONDS:.1f}s"
        )
        drowsiness_color = STATUS_TEXT_COLOR
    else:
        drowsiness_text = drowsiness_result.closure_state
        drowsiness_color = (
            STATUS_ERROR_COLOR
            if drowsiness_result.closure_state == "MEASUREMENT UNAVAILABLE"
            else STATUS_TEXT_COLOR
        )

    cv2.putText(
        frame,
        drowsiness_text,
        (10, 172),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        drowsiness_color,
        2,
    )


def main():
    camera = None
    face_detector = None
    alert = None
    try:
        camera = Camera()
        if not camera.is_opened:
            print(
                "Error: Could not open the default webcam. "
                "Check that a camera is connected and not being used by another app."
            )
            return

        alert = DrowsinessAlert()
        face_detector = FaceLandmarkDetector()
        eye_detector = EyeDetector()
        drowsiness_detector = DrowsinessDetector()
        start_time = time.monotonic()

        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read a frame from the webcam.")
                break

            face_landmarks = face_detector.process(frame)
            if face_landmarks is not None:
                face_detector.draw_landmarks(frame, face_landmarks)

            elapsed = time.monotonic() - start_time
            frame_height, frame_width, _ = frame.shape
            eye_result = eye_detector.process(
                face_landmarks,
                frame_width,
                frame_height,
                elapsed,
            )
            drowsiness_result = drowsiness_detector.update(
                eye_result,
                face_detected=face_landmarks is not None,
            )
            alert.update(drowsiness_result.drowsiness_triggered)
            _draw_status(
                frame,
                face_landmarks,
                eye_result,
                eye_detector.sample_counts(),
                elapsed,
                drowsiness_result,
            )

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            if alert is not None:
                alert.close()
        finally:
            try:
                if face_detector is not None:
                    face_detector.close()
            finally:
                try:
                    if camera is not None:
                        camera.release()
                finally:
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
