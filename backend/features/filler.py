import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# ── Landmark index sets ───────────────────────────────────────────────────────
CHEEK_LEFT_IDX  = [116, 123, 147, 213, 192]
CHEEK_RIGHT_IDX = [345, 352, 376, 433, 416]
CHEEK_LEFT_ANC  = 117
CHEEK_RIGHT_ANC = 346

JAW_LEFT_IDX  = [172, 136, 150, 149, 176, 148, 152]
JAW_RIGHT_IDX = [396, 365, 379, 378, 400, 377, 152]
JAW_LEFT_ANC  = 149
JAW_RIGHT_ANC = 378

CHIN_ANC     = 152
CHIN_ROI_IDX = [152, 148, 176, 400, 377, 378, 365, 396]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _odd(n: int) -> int:
    n = max(n, 3)
    return n if n % 2 == 1 else n + 1


def _detect(img):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0].landmark


def _face_metrics(lms, w, h):
    face_w = int((lms[454].x - lms[234].x) * w)
    face_h = int((lms[152].y - lms[10].y) * h)
    return face_w, face_h


def _poly_blend_mask(roi_h, roi_w, local_pts, dil_k, blur_k):
    """Convex-hull polygon → dilate → feather."""
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    pts  = np.clip(local_pts, [0, 0], [roi_w - 1, roi_h - 1])
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    mask = cv2.dilate(mask, np.ones((_odd(dil_k), _odd(dil_k)), np.uint8))
    return cv2.GaussianBlur(mask, (_odd(blur_k), _odd(blur_k)), 0)


def _apply_warp(roi, gx, gy, dx_field, dy_field):
    roi_h, roi_w = roi.shape[:2]
    map_x = np.clip(gx + dx_field, 0, roi_w - 1).astype(np.float32)
    map_y = np.clip(gy + dy_field, 0, roi_h - 1).astype(np.float32)
    return cv2.remap(roi, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _blend_back(image, warped, blend_mask, y1, x1):
    mask_f  = blend_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
    roi_f   = image[y1:y1 + warped.shape[0], x1:x1 + warped.shape[1]].astype(np.float32)
    blended = (warped.astype(np.float32) * mask_f + roi_f * (1.0 - mask_f)).astype(np.uint8)
    result  = image.copy()
    result[y1:y1 + warped.shape[0], x1:x1 + warped.shape[1]] = blended
    return result


# ── Per-zone warp helpers ─────────────────────────────────────────────────────

def _warp_cheek_side(image, lms, w, h, face_w, pt_indices, anchor_idx, is_left, intensity):
    """Push cheekbone outward and upward using a Gaussian centred on anchor_idx."""
    pts = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in pt_indices], dtype=np.int32)
    ax  = int(lms[anchor_idx].x * w)
    ay  = int(lms[anchor_idx].y * h)

    pad = int(face_w * 0.28)
    x1  = max(np.min(pts[:, 0]) - pad, 0)
    x2  = min(np.max(pts[:, 0]) + pad, w)
    y1  = max(np.min(pts[:, 1]) - pad, 0)
    y2  = min(np.max(pts[:, 1]) + pad, h)
    if y2 <= y1 or x2 <= x1:
        return image

    roi          = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    lax, lay     = ax - x1, ay - y1

    gx, gy = np.meshgrid(
        np.arange(roi_w, dtype=np.float32),
        np.arange(roi_h, dtype=np.float32)
    )

    sigma  = face_w * 0.17
    weight = np.exp(-((gx - lax) ** 2 + (gy - lay) ** 2) / (2 * sigma ** 2))

    # Outward: left cheek shifts left (+dx samples from inner/right side)
    # Upward:  +dy samples from below → content rises
    push_x = face_w * 0.055 * intensity
    push_y = face_w * 0.035 * intensity

    dx_field = weight * push_x if is_left else -weight * push_x
    dy_field = weight * push_y

    warped = _apply_warp(roi, gx, gy, dx_field, dy_field)

    local_pts = pts - np.array([x1, y1])
    mask      = _poly_blend_mask(
        roi_h, roi_w, local_pts,
        dil_k=int(face_w * 0.09),
        blur_k=int(face_w * 0.13),
    )
    return _blend_back(image, warped, mask, y1, x1)


def _warp_jaw_side(image, lms, w, h, face_w, face_h, pt_indices, anchor_idx, is_left, intensity):
    """Push jaw edge downward and slightly outward for sharper jaw definition."""
    pts = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in pt_indices], dtype=np.int32)
    ax  = int(lms[anchor_idx].x * w)
    ay  = int(lms[anchor_idx].y * h)

    pad = int(face_w * 0.18)
    x1  = max(np.min(pts[:, 0]) - pad, 0)
    x2  = min(np.max(pts[:, 0]) + pad, w)
    y1  = max(np.min(pts[:, 1]) - pad, 0)
    y2  = min(np.max(pts[:, 1]) + pad, h)
    if y2 <= y1 or x2 <= x1:
        return image

    roi          = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    lax, lay     = ax - x1, ay - y1

    gx, gy = np.meshgrid(
        np.arange(roi_w, dtype=np.float32),
        np.arange(roi_h, dtype=np.float32)
    )

    sigma  = face_w * 0.14
    weight = np.exp(-((gx - lax) ** 2 + (gy - lay) ** 2) / (2 * sigma ** 2))

    # Down: -dy samples from above → content drops
    # Outward: same sign convention as cheek
    push_down = face_h * 0.030 * intensity
    push_out  = face_w * 0.022 * intensity

    dx_field = weight * push_out if is_left else -weight * push_out
    dy_field = -weight * push_down

    warped = _apply_warp(roi, gx, gy, dx_field, dy_field)

    local_pts = pts - np.array([x1, y1])
    mask      = _poly_blend_mask(
        roi_h, roi_w, local_pts,
        dil_k=int(face_w * 0.07),
        blur_k=int(face_w * 0.11),
    )
    return _blend_back(image, warped, mask, y1, x1)


def _warp_chin(image, lms, w, h, face_w, face_h, intensity):
    """Project chin tip downward — Gaussian centred on landmark 152."""
    ax = int(lms[CHIN_ANC].x * w)
    ay = int(lms[CHIN_ANC].y * h)

    roi_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in CHIN_ROI_IDX],
        dtype=np.int32
    )
    pad = int(face_w * 0.22)
    x1  = max(np.min(roi_pts[:, 0]) - pad, 0)
    x2  = min(np.max(roi_pts[:, 0]) + pad, w)
    y1  = max(np.min(roi_pts[:, 1]) - pad, 0)
    y2  = min(np.max(roi_pts[:, 1]) + pad, h)
    if y2 <= y1 or x2 <= x1:
        return image

    roi          = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    lax, lay     = ax - x1, ay - y1

    gx, gy = np.meshgrid(
        np.arange(roi_w, dtype=np.float32),
        np.arange(roi_h, dtype=np.float32)
    )

    sigma  = face_w * 0.13
    weight = np.exp(-((gx - lax) ** 2 + (gy - lay) ** 2) / (2 * sigma ** 2))

    push_down = face_h * 0.045 * intensity
    dy_field  = -weight * push_down   # -dy: sample above → content projects down

    warped = _apply_warp(roi, gx, gy, np.zeros_like(dy_field), dy_field)

    # Ellipse mask centred on chin tip — tighter than a polygon here
    blend_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    rx = int(face_w * 0.20)
    ry = int(face_w * 0.16)
    cv2.ellipse(blend_mask, (lax, lay), (rx, ry), 0, 0, 360, 255, -1)
    blur_k = _odd(int(face_w * 0.11))
    blend_mask = cv2.GaussianBlur(blend_mask, (blur_k, blur_k), 0)

    return _blend_back(image, warped, blend_mask, y1, x1)


# ── Public API ────────────────────────────────────────────────────────────────

def process_cheek_filler(image_bytes: bytes, intensity: float = 0.7) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    original = img.copy()
    h, w     = img.shape[:2]
    lms      = _detect(img)
    if lms is None:
        return _side_by_side(original, original)

    face_w, _ = _face_metrics(lms, w, h)
    result    = original.copy()
    result    = _warp_cheek_side(result, lms, w, h, face_w, CHEEK_LEFT_IDX,  CHEEK_LEFT_ANC,  True,  intensity)
    result    = _warp_cheek_side(result, lms, w, h, face_w, CHEEK_RIGHT_IDX, CHEEK_RIGHT_ANC, False, intensity)
    return _side_by_side(original, result)


def process_jawline_filler(image_bytes: bytes, intensity: float = 0.7) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    original = img.copy()
    h, w     = img.shape[:2]
    lms      = _detect(img)
    if lms is None:
        return _side_by_side(original, original)

    face_w, face_h = _face_metrics(lms, w, h)
    result         = original.copy()
    result         = _warp_jaw_side(result, lms, w, h, face_w, face_h, JAW_LEFT_IDX,  JAW_LEFT_ANC,  True,  intensity)
    result         = _warp_jaw_side(result, lms, w, h, face_w, face_h, JAW_RIGHT_IDX, JAW_RIGHT_ANC, False, intensity)
    return _side_by_side(original, result)


def process_chin_filler(image_bytes: bytes, intensity: float = 0.7) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    original = img.copy()
    h, w     = img.shape[:2]
    lms      = _detect(img)
    if lms is None:
        return _side_by_side(original, original)

    face_w, face_h = _face_metrics(lms, w, h)
    result         = _warp_chin(original.copy(), lms, w, h, face_w, face_h, intensity)
    return _side_by_side(original, result)


def process_filler(
    image_bytes: bytes,
    cheek_intensity:    float = 0.0,
    jawline_intensity:  float = 0.0,
    chin_intensity:     float = 0.0,
    side_by_side:       bool  = True,
) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    original = img.copy()
    h, w     = img.shape[:2]
    lms      = _detect(img)
    if lms is None:
        return _side_by_side(original, original)

    face_w, face_h = _face_metrics(lms, w, h)
    result         = original.copy()

    if cheek_intensity > 0.01:
        result = _warp_cheek_side(result, lms, w, h, face_w, CHEEK_LEFT_IDX,  CHEEK_LEFT_ANC,  True,  cheek_intensity)
        result = _warp_cheek_side(result, lms, w, h, face_w, CHEEK_RIGHT_IDX, CHEEK_RIGHT_ANC, False, cheek_intensity)

    if jawline_intensity > 0.01:
        result = _warp_jaw_side(result, lms, w, h, face_w, face_h, JAW_LEFT_IDX,  JAW_LEFT_ANC,  True,  jawline_intensity)
        result = _warp_jaw_side(result, lms, w, h, face_w, face_h, JAW_RIGHT_IDX, JAW_RIGHT_ANC, False, jawline_intensity)

    if chin_intensity > 0.01:
        result = _warp_chin(result, lms, w, h, face_w, face_h, chin_intensity)

    if side_by_side:
        return _side_by_side(original, result)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _side_by_side(before, after):
    h, w = before.shape[:2]
    bar  = 40
    out  = np.zeros((h + bar, w * 2, 3), dtype=np.uint8)
    out[bar:, :w] = before
    out[bar:, w:] = after
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "BEFORE", (w // 2 - 60,     28), f, 0.9, (255, 255, 255), 2)
    cv2.putText(out, "AFTER",  (w + w // 2 - 50, 28), f, 0.9, (200, 255, 200), 2)
    cv2.line(out, (w, 0), (w, h + bar), (80, 80, 80), 2)
    return out
