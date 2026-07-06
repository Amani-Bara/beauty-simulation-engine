import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Lip landmark indexes
LIP_INDEXES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375
]

# Separate upper & lower lips
UPPER_LIP = [61, 146, 91, 181]
LOWER_LIP = [84, 17, 314, 405, 321, 375, 291]


def process_lips(image_bytes, intensity=1.0):
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
            h, w, _ = image.shape

            lip_points = []

            # Get lip coordinates
            for idx in LIP_INDEXES:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                lip_points.append((x, y))

            lip_points = np.array(lip_points)

            # Find center of lips
            center = np.mean(lip_points, axis=0)

            new_points = []

            # Expand lips
            for i, (x, y) in enumerate(lip_points):
                dx = x - center[0]
                dy = y - center[1]

                idx = LIP_INDEXES[i]

                # Balanced scaling
                if idx in UPPER_LIP:
                    scale = 1 + (0.10 * intensity)
                else:
                    scale = 1 + (0.18 * intensity)

                new_x = int(center[0] + dx * scale)
                new_y = int(center[1] + dy * scale)

                new_points.append((new_x, new_y))

            # Draw smooth lip outline
            pts = np.array(new_points, np.int32)

            overlay = image.copy()

            cv2.fillPoly(overlay, [pts], (80, 80, 255))

            # Blend naturally
            alpha = 0.25
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image