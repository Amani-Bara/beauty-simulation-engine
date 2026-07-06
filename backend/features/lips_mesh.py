import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
LIPS = mp.solutions.face_mesh.FACEMESH_LIPS
INNER_TOP_IDX    = [13, 82, 81, 80, 312, 311, 310]
INNER_BOTTOM_IDX = [14, 87, 88, 178, 317, 318, 402]


def process_lips(image_bytes, intensity=0.8):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None

    original = image.copy()
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return image

        lms = results.multi_face_landmarks[0].landmark

        lip_points = []
        used = set()
        for connection in LIPS:
            for idx in connection:
                if idx not in used:
                    used.add(idx)
                    lip_points.append((int(lms[idx].x * w), int(lms[idx].y * h)))

        lip_points = np.array(lip_points, dtype=np.int32)
        x, y, box_w, box_h = cv2.boundingRect(lip_points)
        center_x = x + box_w // 2

        inner_top_y    = np.mean([lms[i].y * h for i in INNER_TOP_IDX])
        inner_bottom_y = np.mean([lms[i].y * h for i in INNER_BOTTOM_IDX])
        mouth_line_y   = int((inner_top_y + inner_bottom_y) / 2)

        pad_x = int(box_w * 0.8)
        pad_y = int(box_h * 2.5)   # more vertical room for expansion
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + box_w + pad_x, w)
        y2 = min(y + box_h + pad_y, h)

        roi = original[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]

        local_cx         = center_x - x1
        local_mouth_line = mouth_line_y - y1

        map_x, map_y = np.meshgrid(
            np.arange(roi_w, dtype=np.float32),
            np.arange(roi_h, dtype=np.float32)
        )

        dx = map_x - local_cx
        dy = map_y - local_mouth_line

        radius_x = box_w * 0.75
        radius_y = box_h * 1.3

        dist   = (dx / radius_x) ** 2 + (dy / radius_y) ** 2
        weight = np.clip(1.0 - dist, 0, 1) ** 1.5

        upper_scale_y = 1.0 + (0.18 * intensity)
        lower_scale_y = 1.0 + (0.60 * intensity)
        scale_x       = 1.0 + (0.10 * intensity)

        upper_mask = dy < 0
        lower_mask = dy >= 0

        map_x_new = local_cx + dx * (1.0 - weight * (1.0 - 1.0 / scale_x))
        map_y_new = map_y.copy()

        map_y_new[upper_mask] = (
            local_mouth_line + dy[upper_mask] *
            (1.0 - weight[upper_mask] * (1.0 - 1.0 / upper_scale_y))
        )
        map_y_new[lower_mask] = (
            local_mouth_line + dy[lower_mask] *
            (1.0 - weight[lower_mask] * (1.0 - 1.0 / lower_scale_y))
        )

        warped_roi = cv2.remap(
            roi,
            map_x_new.astype(np.float32),
            map_y_new.astype(np.float32),
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )

        # ── Key fix: build mask from WARPED lip boundary, not original ──
        # Expand the lip polygon outward to cover where the lip will expand to
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        shifted = lip_points.copy()
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1

        # Draw original lip
        cv2.fillPoly(mask, [shifted], 255)

        # Expand downward specifically for lower lip — add a rectangle below
        lower_expand_px = int(box_h * 0.60 * intensity)
        lx = int(shifted[:, 0].min())
        rx = int(shifted[:, 0].max())
        lip_bot = int(shifted[:, 1].max())
        cv2.rectangle(mask,
                      (lx, lip_bot - lower_expand_px // 2),
                      (rx, lip_bot + lower_expand_px),
                      255, -1)

        # Expand upward for upper lip
        upper_expand_px = int(box_h * 0.18 * intensity)
        lip_top = int(shifted[:, 1].min())
        cv2.rectangle(mask,
                      (lx, lip_top - upper_expand_px),
                      (rx, lip_top + upper_expand_px // 2),
                      255, -1)

        # Now blur to feather edges naturally
        blur_px = int(box_h * 0.7)
        blur_px = blur_px if blur_px % 2 == 1 else blur_px + 1
        mask = cv2.GaussianBlur(mask, (blur_px, blur_px), 0)

        mask_f = mask.astype(np.float32) / 255.0
        mask_f = np.expand_dims(mask_f, axis=2)

        blended = (
            warped_roi.astype(np.float32) * mask_f +
            roi.astype(np.float32) * (1.0 - mask_f)
        ).astype(np.uint8)

        image[y1:y2, x1:x2] = blended

    return image