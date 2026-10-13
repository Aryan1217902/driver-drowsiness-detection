import unittest
from unittest.mock import Mock, patch

from camera import Camera


class CameraTests(unittest.TestCase):
    @patch("camera.cv2.VideoCapture")
    def test_opens_configured_camera_index_and_reports_state(self, video_capture):
        capture = Mock()
        capture.isOpened.return_value = True
        video_capture.return_value = capture

        camera = Camera(camera_index=3)

        video_capture.assert_called_once_with(3)
        self.assertTrue(camera.is_opened)

    @patch("camera.cv2.flip")
    @patch("camera.cv2.VideoCapture")
    def test_successful_read_returns_mirrored_frame(self, video_capture, flip):
        capture = Mock()
        frame = object()
        mirrored_frame = object()
        capture.read.return_value = (True, frame)
        video_capture.return_value = capture
        flip.return_value = mirrored_frame

        camera = Camera()
        success, result = camera.read()

        self.assertTrue(success)
        self.assertIs(result, mirrored_frame)
        flip.assert_called_once_with(frame, 1)

    @patch("camera.cv2.flip")
    @patch("camera.cv2.VideoCapture")
    def test_failed_read_returns_original_result_without_flipping(
        self, video_capture, flip
    ):
        capture = Mock()
        frame = object()
        capture.read.return_value = (False, frame)
        video_capture.return_value = capture

        camera = Camera()
        success, result = camera.read()

        self.assertFalse(success)
        self.assertIs(result, frame)
        flip.assert_not_called()

    @patch("camera.cv2.VideoCapture")
    def test_release_releases_capture(self, video_capture):
        capture = Mock()
        video_capture.return_value = capture

        camera = Camera()
        camera.release()

        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
