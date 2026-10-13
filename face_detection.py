"""MediaPipe Face Mesh processing and landmark visualization."""

import cv2
import mediapipe as mp

from config import (
    EYE_CONTOUR_COLOR,
    FACE_MAX_NUM_FACES,
    FACE_MESH_COLOR,
    FACE_MIN_DETECTION_CONFIDENCE,
    FACE_MIN_TRACKING_CONFIDENCE,
)


class FaceLandmarkDetector:
    """Detect one face and draw its mesh and eye contours."""

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=FACE_MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame):
        """Return the first face's landmarks, or None when no face is found."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def draw_landmarks(self, frame, face_landmarks):
        """Draw the face mesh and label both eye contours on the frame."""
        drawing_utils = mp.solutions.drawing_utils
        face_spec = drawing_utils.DrawingSpec(
            color=FACE_MESH_COLOR,
            thickness=1,
            circle_radius=1,
        )
        eye_spec = drawing_utils.DrawingSpec(
            color=EYE_CONTOUR_COLOR,
            thickness=2,
            circle_radius=1,
        )

        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=face_spec,
            connection_drawing_spec=face_spec,
        )

        self._draw_eye_region(
            frame,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
            eye_spec,
            "LEFT EYE",
        )
        self._draw_eye_region(
            frame,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
            eye_spec,
            "RIGHT EYE",
        )

    @staticmethod
    def _draw_eye_region(frame, face_landmarks, connections, drawing_spec, label):
        drawing_utils = mp.solutions.drawing_utils
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_spec,
        )

        eye_indices = {index for edge in connections for index in edge}
        height, width, _ = frame.shape
        eye_points = [
            (
                int(face_landmarks.landmark[index].x * width),
                int(face_landmarks.landmark[index].y * height),
            )
            for index in eye_indices
        ]
        label_x = min(point[0] for point in eye_points)
        label_y = min(point[1] for point in eye_points)
        cv2.putText(
            frame,
            label,
            (label_x, max(20, label_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            drawing_spec.color,
            1,
        )

    def close(self):
        """Release the MediaPipe Face Mesh resources."""
        self._face_mesh.close()
