import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

FOREHEAD_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    162, 21, 54, 103, 67, 109,
    10, 67, 109, 338, 284
]

EYEBROW_IDX = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300
]

GLABELLA_IDX = [
    9, 8, 107, 55, 285, 336,
    193, 417,
    122, 351,
    168, 6, 197, 195, 5
]

CROWS_LEFT_IDX  = [33, 246, 161, 160, 159, 158, 157, 173,
                   133, 155, 154, 153, 145, 144, 163, 7]
CROWS_RIGHT_IDX = [362, 398, 384, 385, 386, 387, 388, 466,
                   263, 249, 390, 373, 374, 380, 381, 382]

UNDER_LEFT_IDX  = [133, 155, 154, 153, 145, 144, 163, 7,
                   110, 25, 24, 23, 22, 26, 112, 243]
UNDER_RIGHT_IDX = [362, 382, 381, 380, 374, 373, 390, 249,
                   339, 255, 254, 253, 252, 256, 341, 463]

SMILE_LEFT_IDX = [
    36, 31, 32, 50, 205, 207, 187, 92, 216, 206,
    203, 129, 218, 49, 131, 134
]
SMILE_RIGHT_IDX = [
    266, 261, 262, 280, 425, 427, 411, 322, 436, 426,
    423, 358, 438, 279, 360, 363
]

CHIN_IDX = [
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    176, 149, 150, 136, 172, 58, 132, 93, 234,
    32, 171, 175, 396, 369, 395, 394, 378
]

UPPER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]


def process_botox(
    image_bytes: bytes,
    forehead_intensity:        float = 0.65,
    frown_intensity:           float = 0.50,
    crows_intensity:           float = 0.60,
    under_eye_intensity:       float = 0.60,
    nasolabial_fold_intensity: float = 0.60,
    lip_intensity:             float = 0.60,
    chin_intensity:            float = 0.60,
    debug: bool = False,
) -> np.ndarray:

    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        return None

    original = img.copy()
    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return _side_by_side(original, original)
        lms = res.multi_face_landmarks[0].landmark

    result, wrinkle_debug = _treat_forehead(
        original.copy(), lms, w, h, forehead_intensity
    )

    result = _treat_frown_lines(result, lms, w, h, frown_intensity)

    result = _treat_eye_region(
        result, lms, w, h,
        indices=CROWS_LEFT_IDX, intensity=crows_intensity,
        pad_scale=2.0, is_crows=True,
    )
    result = _treat_eye_region(
        result, lms, w, h,
        indices=CROWS_RIGHT_IDX, intensity=crows_intensity,
        pad_scale=2.0, is_crows=True,
    )
    result = _treat_eye_region(
        result, lms, w, h,
        indices=UNDER_LEFT_IDX, intensity=under_eye_intensity,
        pad_scale=1.0, is_crows=False,
    )
    result = _treat_eye_region(
        result, lms, w, h,
        indices=UNDER_RIGHT_IDX, intensity=under_eye_intensity,
        pad_scale=1.0, is_crows=False,
    )

    result = _treat_nasolabial_folds(result, lms, w, h, nasolabial_fold_intensity)
    result = _treat_lip_lines(result, lms, w, h, lip_intensity)
    result = _treat_chin(result, lms, w, h, chin_intensity)

    if debug:
        return _three_panel(original, wrinkle_debug, result)

    return _side_by_side(original, result)


def _treat_forehead(image, lms, w, h, intensity):

    forehead_top_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in FOREHEAD_IDX],
        dtype=np.int32
    )
    eyebrow_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in EYEBROW_IDX],
        dtype=np.int32
    )

    top_y    = int(np.min(forehead_top_pts[:, 1]))
    bottom_y = int(np.max(eyebrow_pts[:, 1]))
    left_x   = int(np.min(forehead_top_pts[:, 0]))
    right_x  = int(np.max(forehead_top_pts[:, 0]))

    bw = right_x - left_x
    bh = bottom_y - top_y

    x1 = max(left_x  - int(bw * 0.05), 0)
    y1 = max(top_y   - int(bh * 0.80), 0)
    x2 = min(right_x + int(bw * 0.05), w)
    y2 = min(bottom_y, h)

    if y2 <= y1 or x2 <= x1:
        return image, image

    roi = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    bh, bw = roi_h, roi_w

    eyebrow_exclusion  = int(roi_h * 0.20)
    hairline_exclusion = int(roi_h * 0.15)

    region_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.rectangle(region_mask,
                  (0, hairline_exclusion),
                  (roi_w, roi_h - eyebrow_exclusion), 255, -1)
    blur_k = max(int(min(bw, bh) * 0.15), 21)
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
    region_mask_soft = cv2.GaussianBlur(region_mask, (blur_k, blur_k), 0)
    mask_f3 = (region_mask_soft.astype(np.float32) / 255.0)[:, :, np.newaxis]

    roi_f = roi.astype(np.float32)

    # Step 1: black tophat groove fill
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray).astype(np.float32)

    kw = int(bw * 0.20); kw = kw if kw % 2 == 1 else kw + 1; kw = max(kw, 15)
    kh = max(int(bh * 0.025), 5); kh = kh if kh % 2 == 1 else kh + 1

    line_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    black_tophat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_BLACKHAT, line_kernel)
    if black_tophat.max() > 0:
        black_tophat = black_tophat / black_tophat.max()

    wrinkle_map = (black_tophat > 0.30).astype(np.uint8) * 255
    wrinkle_map = cv2.bitwise_and(wrinkle_map, region_mask)

    dil_k = max(int(bh * 0.010), 3)
    dil_k = dil_k if dil_k % 2 == 1 else dil_k + 1
    wrinkle_map_dilated = cv2.dilate(wrinkle_map, np.ones((dil_k, dil_k), np.uint8))

    wrinkle_debug = image.copy()
    wrinkle_debug[y1:y2, x1:x2][wrinkle_map_dilated > 0] = [0, 0, 255]

    filled = roi_f.copy()
    sample_offset = max(int(bh * 0.06), 6)
    wrinkle_ys, wrinkle_xs = np.where(wrinkle_map_dilated > 0)

    for y_px, x_px in zip(wrinkle_ys, wrinkle_xs):
        samples = []
        for offset in range(1, sample_offset + 1):
            y_above = y_px - offset
            y_below = y_px + offset
            if y_above >= 0 and wrinkle_map_dilated[y_above, x_px] == 0:
                samples.append(roi_f[y_above, x_px])
            if y_below < roi_h and wrinkle_map_dilated[y_below, x_px] == 0:
                samples.append(roi_f[y_below, x_px])
            if len(samples) >= 4:
                break
        if samples:
            filled[y_px, x_px] = np.mean(samples, axis=0)

    filled_u8   = np.clip(filled, 0, 255).astype(np.uint8)
    after_step1 = cv2.GaussianBlur(filled_u8, (3, 3), 0)

    # Step 2: very subtle bilateral on top
    # bilateral_strength is INDEPENDENT of intensity — caps at 0.06
    bilateral_strength = min(intensity * 0.10, 0.10)
    smooth1     = cv2.bilateralFilter(after_step1, d=9, sigmaColor=60, sigmaSpace=60)
    smooth2     = cv2.bilateralFilter(smooth1,     d=9, sigmaColor=45, sigmaSpace=45)
    after_step2 = cv2.addWeighted(smooth2, bilateral_strength,
                                  after_step1, 1.0 - bilateral_strength, 0)

    # Final blend — intensity only controls groove fill strength
    treated = cv2.addWeighted(
        after_step2, intensity,
        roi,         1.0 - intensity,
        0
    ).astype(np.float32)

    blended = (
        treated * mask_f3 +
        roi_f   * (1.0 - mask_f3)
    ).astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = blended
    return result, wrinkle_debug


def _treat_frown_lines(image, lms, w, h, intensity):

    LEFT_INNER  = [55, 107]
    RIGHT_INNER = [285, 336]
    BROW_MID    = [9, 8, 107, 336]

    left_pts  = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in LEFT_INNER], dtype=np.int32)
    right_pts = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in RIGHT_INNER], dtype=np.int32)
    mid_pts   = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in BROW_MID], dtype=np.int32)

    cx     = int((np.mean(left_pts[:, 0]) + np.mean(right_pts[:, 0])) / 2)
    top_y  = int(np.min(mid_pts[:, 1]))
    bot_y  = int(np.max(left_pts[:, 1]))
    zone_h = max(bot_y - top_y, 20)
    zone_w = int(abs(np.mean(right_pts[:, 0]) - np.mean(left_pts[:, 0])) * 0.5)

    x1 = max(cx - int(zone_w * 1.15), 0)
    x2 = min(cx + int(zone_w * 1.15), w)
    y1 = max(top_y - int(zone_h * 2.5), 0)
    y2 = min(bot_y + int(zone_h * 0.45), h)

    if y2 <= y1 or x2 <= x1:
        return image

    roi          = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    roi_f        = roi.astype(np.float32)

    # =========================================
    # L CHANNEL ONLY
    # =========================================

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)
    L_f = L.astype(np.float32)

    # =========================================
    # WRINKLE DETECTION
    # "11" lines are vertical — tall narrow kernel
    # =========================================

    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    L_enhanced = clahe.apply(L)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    blackhat = cv2.morphologyEx(L_enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    wrinkle_mask = np.clip(
        (blackhat.astype(np.float32) - 10.0) / 35.0, 0.0, 1.0
    )

    # Protect eyebrow area at bottom
    wrinkle_mask[int(roi_h * 0.82):, :] = 0
    # Protect hairline at top
    wrinkle_mask[:int(roi_h * 0.10), :] = 0

    # =========================================
    # FREQUENCY SEPARATION
    # =========================================

    coarse = cv2.bilateralFilter(
        L, d=9, sigmaColor=30, sigmaSpace=30
    ).astype(np.float32)

    fine = L_f - coarse

    # =========================================
    # LIFT
    # =========================================

    lift          = wrinkle_mask * 25.0
    coarse_lifted = np.clip(coarse + lift, 0, 255)
    L_treated     = np.clip(coarse_lifted + fine * 0.50, 0, 255)

    # =========================================
    # MERGE BACK TO BGR
    # =========================================

    L_out   = L_treated.astype(np.uint8)
    lab_out = cv2.merge([L_out, A, B])
    treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # =========================================
    # BLEND
    # =========================================

    region_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.ellipse(
        region_mask,
        (roi_w // 2, roi_h // 2),
        (int(roi_w * 0.44), int(roi_h * 0.44)),
        0, 0, 360, 255, -1
    )

    blur_k = max(int(min(roi_w, roi_h) * 0.25), 11)
    if blur_k % 2 == 0:
        blur_k += 1

    mask_soft = cv2.GaussianBlur(region_mask, (blur_k, blur_k), 0)
    mask_f3   = (mask_soft.astype(np.float32) / 255.0)[:, :, np.newaxis]

    treated_f = treated.astype(np.float32)
    blended   = (
        roi_f + (treated_f - roi_f) * mask_f3 * intensity
    ).astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = blended
    return result


def _treat_eye_region(image, lms, w, h, indices, intensity, pad_scale, is_crows):

    eye_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices],
        dtype=np.int32
    )

    cx       = int(np.mean(eye_pts[:, 0]))
    left_x   = np.min(eye_pts[:, 0])
    right_x  = np.max(eye_pts[:, 0])
    top_y    = np.min(eye_pts[:, 1])
    bottom_y = np.max(eye_pts[:, 1])
    eye_w    = right_x - left_x
    eye_h    = bottom_y - top_y

    # =========================================
    # ROI EXTRACTION — different per region
    # =========================================

    if is_crows:
        is_left_eye = cx < w // 2
        if is_left_eye:
            x1 = max(left_x  - int(eye_w * 3.0), 0)
            x2 = min(right_x + int(eye_w * 0.2), w)
        else:
            x1 = max(left_x  - int(eye_w * 0.2), 0)
            x2 = min(right_x + int(eye_w * 3.0), w)
        y1 = max(top_y    - int(eye_h * 0.3), 0)
        y2 = min(bottom_y + int(eye_h * 1.5), h)
        face_left_x  = int(lms[234].x * w)
        face_right_x = int(lms[454].x * w)
        x1 = max(x1, face_left_x)
        x2 = min(x2, face_right_x)
    else:
        side_expand = int(eye_w * 0.5)
        x1 = max(left_x  - side_expand, 0)
        x2 = min(right_x + side_expand, w)
        y1 = max(bottom_y, 0)
        y2 = min(bottom_y + int(eye_h * 3.2), h)

    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return image
    if intensity <= 0.01:
        return image

    intensity    = float(np.clip(intensity, 0.0, 1.0))
    roi_f        = roi.astype(np.float32)
    roi_h, roi_w = roi.shape[:2]

    # =========================================
    # STEP 1 — WRINKLE DETECTION
    # =========================================

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if is_crows:
        # fine diagonal lines — smaller kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    else:
        # horizontal under-eye folds — larger kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    wrinkle_mask   = np.clip(
        (blackhat.astype(np.float32) - 10.0) / 35.0, 0.0, 1.0
    )
    wrinkle_mask_3 = wrinkle_mask[:, :, np.newaxis]

    # =========================================
    # STEP 2 — PROTECT EYE AREA
    # Different boundary per region
    # =========================================

    if is_crows:
        is_left_eye = cx < w // 2
        # protect top (eyebrow area)
        wrinkle_mask_3[:int(roi_h * 0.3), :, :] = 0
        # protect inner side where eye sits
        if is_left_eye:
            wrinkle_mask_3[:, int(roi_w * 0.6):, :] = 0
        else:
            wrinkle_mask_3[:, :int(roi_w * 0.4), :] = 0
    else:
        # under-eye: protect upper 55% where eye sits
        wrinkle_mask_3[:int(roi_h * 0.55), :, :] = 0

    # =========================================
    # STEP 3 — FREQUENCY SEPARATION
    # =========================================

    coarse = cv2.bilateralFilter(
        roi, d=9, sigmaColor=40, sigmaSpace=40
    ).astype(np.float32)

    fine = roi_f - coarse

    # =========================================
    # STEP 4 — LIFT WRINKLE VALLEYS
    # =========================================

    if is_crows:
        lift_strength = intensity * 30.0
    else:
        lift_strength = intensity * 38.0

    lift          = wrinkle_mask_3 * lift_strength
    coarse_lifted = np.clip(coarse + lift, 0, 255)

    # =========================================
    # STEP 5 — RECOMBINE
    # =========================================

    treated = np.clip(coarse_lifted + fine * 0.50, 0, 255)

    # =========================================
    # STEP 6 — BLEND BACK WITH ELLIPSE MASK
    # =========================================

    blend_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.ellipse(
        blend_mask,
        (roi_w // 2, roi_h // 2),
        (int(roi_w * 0.45), int(roi_h * 0.42)),
        0, 0, 360, 255, -1
    )

    blur_size = max(int(min(roi_h, roi_w) * 0.18), 31)
    if blur_size % 2 == 0:
        blur_size += 1

    blend_mask   = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0)
    blend_mask_f = (blend_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    blended = (
        roi_f + (treated - roi_f) * blend_mask_f * intensity
    ).astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = blended
    return result
def _treat_nasolabial_folds(image, lms, w, h, intensity=1.0):

    result = image.copy()

    for side_idx in [SMILE_LEFT_IDX, SMILE_RIGHT_IDX]:
        pts = np.array(
            [(int(lms[i].x * w), int(lms[i].y * h)) for i in side_idx],
            dtype=np.int32
        )

        # Proportional margins — avoids fixed-px bleed on large images
        fold_w = int(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        fold_h = int(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        mx = max(int(fold_w * 0.12), 6)
        my = max(int(fold_h * 0.10), 6)
        x1 = max(int(np.min(pts[:, 0])) - mx, 0)
        x2 = min(int(np.max(pts[:, 0])) + mx, w)
        y1 = max(int(np.min(pts[:, 1])) - my, 0)
        y2 = min(int(np.max(pts[:, 1])) + my, h)

        if y2 <= y1 or x2 <= x1:
            continue

        roi = result[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        roi_h, roi_w = roi.shape[:2]
        roi_f = roi.astype(np.float32)

        # Work on L channel only — prevents hue cast at blend boundary
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(roi_lab)
        L_f = L.astype(np.float32)

        # =========================================
        # WRINKLE DETECTION
        # Dual-kernel blackhat — vertical crease +
        # diagonal component.  Keep response raw
        # (do NOT clip * 6) so lift stays
        # proportional to actual shadow depth.
        # =========================================
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        L_enh = clahe.apply(L)

        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
        kd = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        bh_v = cv2.morphologyEx(L_enh, cv2.MORPH_BLACKHAT, kv)
        bh_d = cv2.morphologyEx(L_enh, cv2.MORPH_BLACKHAT, kd)
        blackhat = cv2.GaussianBlur(
            np.maximum(bh_v, bh_d), (5, 5), 0
        ).astype(np.float32)

        # Threshold above skin-texture noise floor (~12 on 8-bit CLAHE output).
        # Scale relative to 40 units (typical fold shadow depth post-CLAHE).
        # Result: 0 for pores/texture, 0–1 for the actual fold.
        NOISE_FLOOR = 12.0
        FOLD_SCALE  = 40.0
        wrinkle_mask = np.clip(
            (blackhat - NOISE_FLOOR) / FOLD_SCALE, 0.0, 1.0
        )

        # Protect mouth corner — bottom 15% of ROI where fold ends at lip
        wrinkle_mask[int(roi_h * 0.85):, :] = 0

        # =========================================
        # POLYGON BOUNDARY — keeps treatment inside
        # landmark region; NOT used as blend weight
        # (that comes from the detection above)
        # =========================================
        pts_local = pts - np.array([x1, y1])
        hull = cv2.convexHull(pts_local)
        poly = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(poly, [hull], 255)
        wrinkle_mask *= poly.astype(np.float32) / 255.0

        # Feather the detection-derived mask
        mask_u8 = np.clip(wrinkle_mask * 255, 0, 255).astype(np.uint8)
        blur_k = max(int(min(roi_h, roi_w) * 0.15), 11)
        blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
        mask_f = cv2.GaussianBlur(mask_u8, (blur_k, blur_k), 0).astype(np.float32) / 255.0

        # =========================================
        # FREQUENCY SEPARATION
        # =========================================
        coarse = cv2.bilateralFilter(
            L, d=9, sigmaColor=35, sigmaSpace=35
        ).astype(np.float32)
        fine = L_f - coarse

        # Max lift 20 L-channel units (≈8% of scale).
        # L-channel uniform lift amplifies warm undertones above ~200;
        # keeping it ≤20 avoids the orange cast.
        lift          = wrinkle_mask * 20.0
        coarse_lifted = np.clip(coarse + lift, 0, 255)
        L_treated     = np.clip(coarse_lifted + fine * 0.35, 0, 255)

        lab_out = cv2.merge([L_treated.astype(np.uint8), A, B])
        treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

        # =========================================
        # BLEND — detection mask × intensity
        # =========================================
        treated_f = treated.astype(np.float32)
        mask_3d   = mask_f[:, :, np.newaxis]
        blended   = (
            roi_f + (treated_f - roi_f) * mask_3d * intensity
        ).astype(np.uint8)

        result[y1:y2, x1:x2] = blended

    return result

def _treat_chin(image, lms, w, h, intensity):

    chin_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in CHIN_IDX],
        dtype=np.int32
    )

    left_x   = int(np.min(chin_pts[:, 0]))
    right_x  = int(np.max(chin_pts[:, 0]))
    top_y    = int(np.min(chin_pts[:, 1]))
    bottom_y = int(np.max(chin_pts[:, 1]))
    chin_w   = right_x - left_x
    chin_h   = bottom_y - top_y

    x1 = max(left_x  - int(chin_w * 0.05), 0)
    x2 = min(right_x + int(chin_w * 0.05), w)
    y1 = max(top_y   + int(chin_h * 0.4),  0)
    y2 = min(bottom_y + int(chin_h * 0.5), h)

    if y2 <= y1 or x2 <= x1:
        return image

    roi          = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    roi_f        = roi.astype(np.float32)

    # =========================================
    # WORK ON L CHANNEL ONLY — preserve color
    # =========================================

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)
    L_f = L.astype(np.float32)

    # =========================================
    # WRINKLE DETECTION
    # Chin lines are horizontal — wide flat kernel
    # =========================================

    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    L_enhanced = clahe.apply(L)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    blackhat = cv2.morphologyEx(L_enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    wrinkle_mask = np.clip(
        (blackhat.astype(np.float32) - 10.0) / 35.0, 0.0, 1.0
    )

    # Protect top edge — avoid blending into mouth/lip area
    wrinkle_mask[:int(roi_h * 0.12), :] = 0

    # =========================================
    # FREQUENCY SEPARATION
    # =========================================

    coarse = cv2.bilateralFilter(
        L, d=9, sigmaColor=30, sigmaSpace=30
    ).astype(np.float32)

    fine = L_f - coarse

    # =========================================
    # TWO-LAYER TREATMENT
    # =========================================

    # Layer 1 — lift detected wrinkle valleys
    lift          = wrinkle_mask * 25.0
    coarse_lifted = np.clip(coarse + lift, 0, 255)

    # Layer 2 — uniform gentle lift for orange-peel dimpling
    uniform_lift  = intensity * 5.0
    coarse_lifted = np.clip(coarse_lifted + uniform_lift, 0, 255)

    # Preserve fine detail — keeps skin texture natural
    L_treated = np.clip(coarse_lifted + fine * 0.50, 0, 255)

    # =========================================
    # MERGE BACK TO BGR
    # =========================================

    L_out   = L_treated.astype(np.uint8)
    lab_out = cv2.merge([L_out, A, B])
    treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # =========================================
    # BLEND
    # =========================================

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (roi_w // 2, roi_h // 2),
        (int(roi_w * 0.44), int(roi_h * 0.44)),
        0, 0, 360, 255, -1
    )

    blur_k = max(int(min(roi_h, roi_w) * 0.22), 15)
    if blur_k % 2 == 0:
        blur_k += 1

    mask   = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    treated_f = treated.astype(np.float32)
    blended   = (
        roi_f + (treated_f - roi_f) * mask_f * intensity
    ).astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = blended
    return result




def _treat_lip_lines(image, lms, w, h, intensity=1.0):
    if intensity <= 0.01:
        return image

    upper_lip_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h))
         for i in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]],
        dtype=np.int32
    )

    lip_left_x  = np.min(upper_lip_pts[:, 0])
    lip_right_x = np.max(upper_lip_pts[:, 0])
    lip_top_y   = np.min(upper_lip_pts[:, 1])
    lip_w       = lip_right_x - lip_left_x

    x1 = max(lip_left_x  + int(lip_w * 0.08), 0)
    x2 = min(lip_right_x - int(lip_w * 0.08), w)
    y1 = max(lip_top_y   - int(lip_w * 0.38), 0)
    y2 = min(lip_top_y   + int(lip_w * 0.04), h)

    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return image

    roi_h, roi_w = roi.shape[:2]
    roi_f        = roi.astype(np.float32)

    # =========================================
    # FREQUENCY SEPARATION ON L CHANNEL ONLY
    # No detection — uniform treatment across
    # the philtrum zone. Lines are too fine and
    # ROI too small for reliable wrinkle mapping.
    # =========================================

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)
    L_f = L.astype(np.float32)

    # Gentle bilateral — preserves lip border edge
    coarse = cv2.bilateralFilter(
        L, d=5, sigmaColor=20, sigmaSpace=20
    ).astype(np.float32)

    fine = L_f - coarse

    # Lift the coarse layer uniformly
    # This fills wrinkle valleys without targeting them
    lift          = intensity * 14.0
    coarse_lifted = np.clip(coarse + lift, 0, 255)

    L_treated = np.clip(coarse_lifted + fine * 0.45, 0, 255)

    L_out   = L_treated.astype(np.uint8)
    lab_out = cv2.merge([L_out, A, B])
    treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # =========================================
    # BLEND — full width, feathered top/bottom
    # No ellipse — this zone is wide and short
    # A horizontal gradient mask fits better
    # =========================================

    blend_mask = np.zeros((roi_h, roi_w), dtype=np.float32)

    for r in range(roi_h):
        # Fade in from top, fade out at bottom near lip border
        t = r / max(roi_h - 1, 1)
        # Peak in the middle rows, zero at very top and bottom
        val = np.sin(t * np.pi)
        blend_mask[r, :] = val

    # Also feather left/right edges
    for c in range(roi_w):
        t = c / max(roi_w - 1, 1)
        edge = np.sin(t * np.pi)
        blend_mask[:, c] *= edge

    blend_mask_f = (blend_mask * intensity)[:, :, np.newaxis]

    treated_f = treated.astype(np.float32)
    blended   = np.clip(
        roi_f + (treated_f - roi_f) * blend_mask_f, 0, 255
    ).astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = blended
    return result



def _side_by_side(before, after):
    h, w = before.shape[:2]
    bar  = 40
    out  = np.zeros((h + bar, w * 2, 3), dtype=np.uint8)
    out[bar:, :w] = before
    out[bar:, w:] = after
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "BEFORE", (60, 28),     f, 0.7, (255, 255, 255), 2)
    cv2.putText(out, "AFTER",  (w + 60, 28), f, 0.7, (255, 255, 255), 2)
    cv2.line(out, (w, 0), (w, h + bar), (80, 80, 80), 2)
    return out


def _three_panel(before, wrinkle_vis, after):
    h, w = before.shape[:2]
    bar  = 40
    out  = np.zeros((h + bar, w * 3, 3), dtype=np.uint8)
    out[bar:, :w]    = before
    out[bar:, w:w*2] = wrinkle_vis
    out[bar:, w*2:]  = after
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "BEFORE",   (60, 28),     f, 0.7, (255, 255, 255), 2)
    cv2.putText(out, "DETECTED", (w+60, 28),   f, 0.7, (0, 100, 255),   2)
    cv2.putText(out, "AFTER",    (w*2+60, 28), f, 0.7, (255, 255, 255), 2)
    cv2.line(out, (w,   0), (w,   h+bar), (80, 80, 80), 2)
    cv2.line(out, (w*2, 0), (w*2, h+bar), (80, 80, 80), 2)
    return out