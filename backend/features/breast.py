import cv2
import numpy as np


def _odd(n: int) -> int:
    n = max(n, 3)
    return n if n % 2 == 1 else n + 1


def _size_pass(image: np.ndarray, cx: int, cy: int, radius: int,
               size_intensity: float, shape: str) -> np.ndarray:
    """Push surrounding skin outward from the breast centre.

    The breast interior (d < radius*0.7) is untouched.
    The ring (radius*0.7 … radius*2.5) is pushed outward.
    The blend mask is zero inside the core so the breast itself is never distorted.
    """
    h, w = image.shape[:2]

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    ddx = gx - cx
    ddy = gy - cy
    d   = np.sqrt(ddx ** 2 + ddy ** 2)

    # Gaussian weight peaked at d = radius, active only in the push ring
    weight = np.exp(-((d - radius) / (radius * 0.5)) ** 2)
    weight[d < radius * 0.7] = 0.0
    weight[d > radius * 2.5] = 0.0

    push   = radius * 0.35 * size_intensity * weight
    d_safe = np.where(d > 1e-6, d, 1.0)   # avoid div-by-zero at centre

    dx_field = (push * ddx / d_safe).astype(np.float32)
    dy_field = (push * ddy / d_safe).astype(np.float32)

    # Shape: scale push magnitude by vertical position relative to cy
    if shape == "teardrop":
        vfactor = np.where(gy < cy, 0.1, 1.0).astype(np.float32)
        dx_field *= vfactor
        dy_field *= vfactor
    elif shape == "natural":
        vfactor = np.where(gy < cy, 0.3, 0.8).astype(np.float32)
        dx_field *= vfactor
        dy_field *= vfactor
    # round: equal in all directions — no modulation needed

    # Smooth to eliminate sharp ring boundary
    smooth_k = _odd(int(radius * 0.6))
    dx_field = cv2.GaussianBlur(dx_field, (smooth_k, smooth_k), 0)
    dy_field = cv2.GaussianBlur(dy_field, (smooth_k, smooth_k), 0)

    # Inward sampling → content pushed outward
    map_x = np.clip(gx - dx_field, 0, w - 1).astype(np.float32)
    map_y = np.clip(gy - dy_field, 0, h - 1).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REFLECT)

    # Ring blend mask: 0 inside breast core, feathered through the push zone
    inner_fade = np.clip((d - radius * 0.6) / (radius * 0.4), 0.0, 1.0)
    outer_fade = np.clip((radius * 2.5 - d) / (radius * 0.3), 0.0, 1.0)
    ring_mask  = (inner_fade * outer_fade * 255).astype(np.uint8)
    blur_k     = _odd(int(radius * 0.5))
    ring_mask  = cv2.GaussianBlur(ring_mask, (blur_k, blur_k), 0)
    mask_f     = ring_mask.astype(np.float32)[:, :, np.newaxis] / 255.0

    return (
        warped.astype(np.float32) * mask_f +
        image.astype(np.float32)  * (1.0 - mask_f)
    ).astype(np.uint8)


def _lift_pass(image: np.ndarray, cx: int, cy: int, radius: int,
               lift_intensity: float) -> np.ndarray:
    """Push the lower half of the breast upward."""
    h, w = image.shape[:2]

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)

    sigma           = radius * 0.7
    gaussian_weight = np.exp(-d ** 2 / (2 * sigma ** 2))
    gaussian_weight[d > radius * 1.5] = 0.0

    push_up = radius * 0.3 * lift_intensity * gaussian_weight

    # Negative dy_field → map_y = gy - (negative) = gy + positive
    # → samples from below → content appears to rise
    dy_field = np.where(gy >= cy, -push_up, 0.0).astype(np.float32)
    dx_field = np.zeros((h, w), dtype=np.float32)

    smooth_k = _odd(int(radius * 0.4))
    dx_field = cv2.GaussianBlur(dx_field, (smooth_k, smooth_k), 0)
    dy_field = cv2.GaussianBlur(dy_field, (smooth_k, smooth_k), 0)

    map_x = np.clip(gx - dx_field, 0, w - 1).astype(np.float32)
    map_y = np.clip(gy - dy_field, 0, h - 1).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REFLECT)

    lift_mask = np.clip(gaussian_weight * 255, 0, 255).astype(np.uint8)
    blur_k    = _odd(int(radius * 0.4))
    lift_mask = cv2.GaussianBlur(lift_mask, (blur_k, blur_k), 0)
    mask_f    = lift_mask.astype(np.float32)[:, :, np.newaxis] / 255.0

    return (
        warped.astype(np.float32) * mask_f +
        image.astype(np.float32)  * (1.0 - mask_f)
    ).astype(np.uint8)


def _process_one(image: np.ndarray, cx: int, cy: int, radius: int,
                 size_intensity: float, shape: str,
                 lift_intensity: float) -> np.ndarray:
    result = image
    if size_intensity > 0.01:
        result = _size_pass(result, cx, cy, radius, size_intensity, shape)
    if lift_intensity > 0.01:
        result = _lift_pass(result, cx, cy, radius, lift_intensity)
    return result


def process_breast(
    image_bytes:    bytes,
    left_x:         int,
    left_y:         int,
    right_x:        int,
    right_y:        int,
    radius:         int,
    size_intensity: float = 0.5,
    lift_intensity: float = 0.0,
    shape:          str   = "round",
    side_by_side:   bool  = True,
) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    original = img.copy()
    result   = original.copy()

    result = _process_one(result, left_x,  left_y,  radius, size_intensity, shape, lift_intensity)
    result = _process_one(result, right_x, right_y, radius, size_intensity, shape, lift_intensity)

    if side_by_side:
        return _side_by_side(original, result)
    return result


def _side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
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
