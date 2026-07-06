import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def detect_face_landmarks(image_bytes):
    # Convert bytes → image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return image

        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                h, w, _ = image.shape
                x, y = int(lm.x * w), int(lm.y * h)

                # Draw point
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    return image