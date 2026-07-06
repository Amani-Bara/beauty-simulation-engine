import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
LIPS = mp.solutions.face_mesh.FACEMESH_LIPS


def process_lips(image_bytes, intensity=1.0):
    # Convert uploaded image bytes to OpenCV image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return image

        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            lip_points = []
            used_points = set()

            # Get full MediaPipe lip mesh points
            for connection in LIPS:
                for idx in connection:
                    if idx not in used_points:
                        used_points.add(idx)

                        lm = face_landmarks.landmark[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        lip_points.append((x, y))

            # Convert points to numpy
            lip_points_np = np.array(lip_points, np.int32)

            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(lip_points_np)

            # Extract lips region
            lip_region = image[y:y+h_box, x:x+w_box]

            # Scale region slightly
            scale_x = 1 + (0.05 * intensity)
            scale_y = 1 + (0.12 * intensity)

            scaled_lips = cv2.resize(
                lip_region,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_CUBIC
            )

            # New dimensions
            new_h, new_w = scaled_lips.shape[:2]

            # Center placement
            center_x = x + w_box // 2
            center_y = y + h_box // 2

            start_x = max(center_x - new_w // 2, 0)
            start_y = max(center_y - new_h // 2, 0)

            end_x = min(start_x + new_w, image.shape[1])
            end_y = min(start_y + new_h, image.shape[0])

            # Blend resized lips back
            image[start_y:end_y, start_x:end_x] = scaled_lips[
                0:end_y-start_y,
                0:end_x-start_x
            ]

        return image