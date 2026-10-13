"""Webcam access and mirrored frame capture."""

import cv2

from config import CAMERA_INDEX


class Camera:
    """Open the configured webcam and return mirrored frames."""

    def __init__(self, camera_index=CAMERA_INDEX):
        self._capture = cv2.VideoCapture(camera_index)

    @property
    def is_opened(self):
        return self._capture.isOpened()

    def read(self):
        """Return (success, frame), mirroring successful frames."""
        success, frame = self._capture.read()
        if success:
            frame = cv2.flip(frame, 1)
        return success, frame

    def release(self):
        """Release the webcam handle."""
        self._capture.release()
