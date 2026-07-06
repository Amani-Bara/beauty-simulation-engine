import cv2
import mediapipe as mp
import numpy as np
from enum import Enum

mp_face_mesh = mp.solutions.face_mesh
LIPS = mp.solutions.face_mesh.FACEMESH_LIPS

INNER_TOP    = [13, 82, 81, 80, 312, 311, 310]
INNER_BOTTOM = [14, 87, 88, 178, 317, 318, 402]
CUPID_LEFT   = 37
CUPID_RIGHT  = 267
CUPID_DIP    = 0
LOWER_CENTER = 17

OUTER_LIP_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
INNER_LIP_IDX = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

# Used for tint — sits on actual lip tissue, not skin border
TINT_BOUNDARY_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]


class LipStyle(str, Enum):
    NATURAL = "natural"
    RUSSIAN = "russian"
    HEART   = "heart"


class TintColor(str, Enum):
    nude    = "nude"
    rose    = "rose"
    berry   = "berry"
    red     = "red"
    coral   = "coral"


# 5 clearly distinct colors in BGR
TINT_COLORS = {
    "nude":   ( 90, 120, 175),   # warm beige-pink
    "rose":   ( 55,  60, 155),   # classic rose
    "berry":  ( 30,  20, 110),   # deep berry-purple
    "red":    ( 15,  15, 160),   # bold true red
    "coral":  ( 45, 130, 210),   # warm orange-coral
}


def process_lips_advanced(
    image_bytes: bytes,
    style: LipStyle = LipStyle.NATURAL,
    intensity: float = 0.8,
    tint: bool = False,
    tint_color: tuple = (55, 60, 155),
    tint_strength: float = 0.30
) -> np.ndarray:

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    original = img.copy()
    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return _side_by_side(original, original)
        lms = res.multi_face_landmarks[0].landmark

    # ── Lip geometry — ordered contours ───────────────────
    outer_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in OUTER_LIP_IDX],
        dtype=np.int32
    )
    inner_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in INNER_LIP_IDX],
        dtype=np.int32
    )

    all_lip_pts = np.vstack([outer_pts, inner_pts])
    bx, by, bw, bh = cv2.boundingRect(all_lip_pts)
    cx = bx + bw // 2

    mouth_top_y = np.mean([lms[i].y * h for i in INNER_TOP])
    mouth_bot_y = np.mean([lms[i].y * h for i in INNER_BOTTOM])
    ml_y = int(mouth_top_y + (mouth_bot_y - mouth_top_y) * 0.3)

    upper_h = ml_y - by
    lower_h = (by + bh) - ml_y

    # ── Style definitions ──────────────────────────────────
    styles = {
        LipStyle.NATURAL: dict(
            upper_lift   = 0.18,
            lower_push   = 0.22,
            upper_spread = 1.00,
            lower_spread = 1.00,
            cupid_extra  = 0.00,
            corner_taper = 0.10,
        ),
        LipStyle.RUSSIAN: dict(
            upper_lift   = 0.55,
            lower_push   = 0.45,
            upper_spread = 0.82,
            lower_spread = 0.75,
            cupid_extra  = 0.35,
            corner_taper = 0.30,
        ),
        LipStyle.HEART: dict(
            upper_lift   = 0.32,
            lower_push   = 0.42,
            upper_spread = 0.45,
            lower_spread = 0.40,
            cupid_extra  = 0.38,
            corner_taper = 0.55,
        ),
    }
    s = styles[style]

    # ── ROI ───────────────────────────────────────────────
    pad_x = int(bw * 0.8)
    pad_y = int(bh * 2.2)
    x1 = max(cx - bw // 2 - pad_x, 0)
    y1 = max(by - pad_y, 0)
    x2 = min(cx + bw // 2 + pad_x, w)
    y2 = min(by + bh + pad_y, h)

    roi          = original[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]

    lcx = cx - x1
    lml = ml_y - y1

    # ── Warp field ────────────────────────────────────────
    gx, gy = np.meshgrid(
        np.arange(roi_w, dtype=np.float32),
        np.arange(roi_h, dtype=np.float32)
    )
    dx = gx - lcx
    dy = gy - lml

    def horiz_weight(spread):
        r = bw * spread * 0.5
        return np.exp(-(dx / r) ** 2 * 1.5)

    corner_dist = np.abs(dx) / (bw * 0.5 + 1e-6)
    corner_w    = 1.0 - np.clip(corner_dist * s["corner_taper"] * 2.0, 0, 0.85)

    # ── UPPER LIP ─────────────────────────────────────────
    upper_mask      = dy < 0
    hw_upper        = horiz_weight(s["upper_spread"]) * corner_w
    t_upper         = np.clip(-dy / (upper_h + 1e-6), 0, 1) ** 0.7
    base_upper_lift = s["upper_lift"] * upper_h * intensity * t_upper * hw_upper

    cupid_l_x   = int(lms[CUPID_LEFT].x  * w) - x1
    cupid_r_x   = int(lms[CUPID_RIGHT].x * w) - x1
    cupid_d_x   = int(lms[CUPID_DIP].x   * w) - x1

    cupid_peak  = (
        np.exp(-((gx - cupid_l_x) / (bw * 0.12)) ** 2) +
        np.exp(-((gx - cupid_r_x) / (bw * 0.12)) ** 2)
    )
    cupid_dip   = np.exp(-((gx - cupid_d_x) / (bw * 0.08)) ** 2)
    cupid_shape = np.clip(cupid_peak - cupid_dip * 0.5, 0, 1)
    cupid_lift  = s["cupid_extra"] * upper_h * intensity * t_upper * cupid_shape

    total_upper = base_upper_lift + cupid_lift
    map_y_new   = gy.copy()
    map_y_new[upper_mask] = gy[upper_mask] + total_upper[upper_mask]

    # ── LOWER LIP ─────────────────────────────────────────
    lower_mask = dy >= 0
    hw_lower   = horiz_weight(s["lower_spread"]) * corner_w
    t_lower    = np.clip(dy / (lower_h + 1e-6), 0, 1) ** 0.5
    lower_push = s["lower_push"] * lower_h * intensity * t_lower * hw_lower

    map_y_new[lower_mask] = gy[lower_mask] - lower_push[lower_mask]

    # ── Remap ─────────────────────────────────────────────
    warped = cv2.remap(
        roi, gx, map_y_new,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    # ── Blend mask ────────────────────────────────────────
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    outer_shifted = outer_pts.copy()
    outer_shifted[:, 0] -= x1
    outer_shifted[:, 1] -= y1
    cv2.fillPoly(mask, [outer_shifted], 255)

    dil    = int(bh * 0.4)
    mask   = cv2.dilate(mask, np.ones((dil, dil), np.uint8))
    blur_k = int(bh * 0.35)
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
    mask   = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    mask_f = mask.astype(np.float32)[:, :, np.newaxis] / 255.0

    blended = (
        warped.astype(np.float32) * mask_f +
        roi.astype(np.float32)    * (1.0 - mask_f)
    ).astype(np.uint8)

    result = original.copy()
    result[y1:y2, x1:x2] = blended

    # ── Lip Tint ──────────────────────────────────────────
    if tint:
        # Build hull from tint boundary — sits on lip tissue not skin
        tint_coords = np.array(
            [(int(lms[i].x * w), int(lms[i].y * h)) for i in TINT_BOUNDARY_IDX],
            dtype=np.int32
        )
        hull = cv2.convexHull(tint_coords)

        tint_mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(tint_mask_full, hull, 255)

        # Erode to pull tint well inside lip border
        erode_k = max(3, int(bh * 0.15))
        tint_mask_full = cv2.erode(
            tint_mask_full, np.ones((erode_k, erode_k), np.uint8)
        )

        # Small blur — soft edge without spreading into skin
        blur_size = int(bh * 0.20)
        if blur_size % 2 == 0:
            blur_size += 1
        tint_mask_full = cv2.GaussianBlur(
            tint_mask_full, (blur_size, blur_size), 0
        )

        # Even coverage, power < 1 means flatter distribution
        alpha = (tint_mask_full.astype(np.float32) / 255.0) ** 0.8
        alpha = np.clip(alpha * tint_strength, 0.0, 0.65)

        tint_layer = np.zeros((h, w, 3), dtype=np.float32)
        tint_layer[:, :, 0] = tint_color[0]
        tint_layer[:, :, 1] = tint_color[1]
        tint_layer[:, :, 2] = tint_color[2]

        result_f   = result.astype(np.float32)
        # Multiply blend — color darkens naturally with skin tone
        multiplied = np.clip(result_f * tint_layer / 128.0, 0, 255)

        alpha_3      = alpha[:, :, np.newaxis]
        blended_tint = result_f * (1.0 - alpha_3) + multiplied * alpha_3
        result       = np.clip(blended_tint, 0, 255).astype(np.uint8)

    return _side_by_side(original, result)


def _side_by_side(before, after):
    h, w = before.shape[:2]
    bar  = 40
    out  = np.zeros((h + bar, w * 2, 3), dtype=np.uint8)
    out[bar:, :w] = before
    out[bar:, w:] = after
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "BEFORE", (w//2 - 60, 28), f, 0.9, (255, 255, 255), 2)
    cv2.putText(out, "AFTER",  (w + w//2 - 50, 28), f, 0.9, (200, 255, 200), 2)
    cv2.line(out, (w, 0), (w, h + bar), (80, 80, 80), 2)
    return out