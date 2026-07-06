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

# Marionette (puppet) lines: mouth corner → jaw on each side
MARIONETTE_LEFT_IDX  = [61, 146, 91, 181, 84, 83, 106, 182, 43, 57, 58, 32, 172]
MARIONETTE_RIGHT_IDX = [291, 375, 321, 405, 314, 313, 335, 406, 273, 287, 288, 262, 396]


def process_botox(
    image_bytes: bytes,
    forehead_intensity:        float = 0.65,
    frown_intensity:           float = 0.50,
    crows_intensity:           float = 0.60,
    under_eye_intensity:       float = 0.60,
    nasolabial_fold_intensity: float = 0.60,
    lip_intensity:             float = 0.60,
    marionette_intensity:      float = 0.60,
    chin_intensity:            float = 0.60,
    cheek_intensity:           float = 0.40,
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
    result = _treat_marionette_lines(result, lms, w, h, marionette_intensity)
    result = _treat_cheek_lines(result, lms, w, h, cheek_intensity)
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
    blur_k = max(int(min(bw, bh) * 0.23), 31) | 1
    region_mask_soft = cv2.GaussianBlur(region_mask, (blur_k, blur_k), 0)
    mask_f3 = (region_mask_soft.astype(np.float32) / 255.0)[:, :, np.newaxis]

    roi_f = roi.astype(np.float32)

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

    bilateral_strength = min(intensity * 0.06, 0.06)
    smooth1     = cv2.bilateralFilter(after_step1, d=9, sigmaColor=60, sigmaSpace=60)
    smooth2     = cv2.bilateralFilter(smooth1,     d=9, sigmaColor=45, sigmaSpace=45)
    after_step2 = cv2.addWeighted(smooth2, bilateral_strength,
                                  after_step1, 1.0 - bilateral_strength, 0)

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

    x1 = max(cx - int(zone_w * 0.85), 0)
    x2 = min(cx + int(zone_w * 0.85), w)
    y1 = max(top_y - int(zone_h * 2.5), 0)
    y2 = min(bot_y + int(zone_h * 0.45), h)

    if y2 <= y1 or x2 <= x1:
        return image

    roi      = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    roi_f    = roi.astype(np.float32)

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)
    L_f = L.astype(np.float32)

    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    L_enhanced = clahe.apply(L)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    blackhat = cv2.morphologyEx(L_enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    peak = blackhat.max()
    wrinkle_mask = (blackhat.astype(np.float32) / peak) if peak > 0.01 else blackhat.astype(np.float32) / 255.0
    wrinkle_mask = np.clip(wrinkle_mask * 0.85, 0, 1)
    wrinkle_mask[int(roi_h * 0.75):, :] = 0
    wrinkle_mask[:int(roi_h * 0.15), :] = 0

    coarse = cv2.bilateralFilter(L, d=9, sigmaColor=30, sigmaSpace=30).astype(np.float32)
    fine   = L_f - coarse

    lift          = wrinkle_mask * intensity * 55.0
    coarse_lifted = np.clip(coarse + lift, 0, 255)
    L_treated     = np.clip(coarse_lifted + fine * 0.25, 0, 255)

    L_out   = L_treated.astype(np.uint8)
    lab_out = cv2.merge([L_out, A, B])
    treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32)

    region_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.ellipse(region_mask, (roi_w//2, roi_h//2),
                (roi_w//2, roi_h//2), 0, 0, 360, 255, -1)
    blur_k = max(int(min(roi_w, roi_h) * 0.60), 17) | 1
    mask_soft = cv2.GaussianBlur(region_mask, (blur_k, blur_k), 0)
    mask_f3   = (mask_soft.astype(np.float32) / 255.0)[:, :, np.newaxis]

    blended = (
        treated  * mask_f3 +
        roi_f    * (1.0 - mask_f3)
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

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if is_crows:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    wrinkle_mask = blackhat.astype(np.float32) / 255.0
    wrinkle_mask = np.clip(wrinkle_mask * 5, 0, 1)

    if is_crows:
        is_left_eye = cx < w // 2
        wrinkle_mask[:int(roi_h * 0.3), :] = 0
        if is_left_eye:
            wrinkle_mask[:, int(roi_w * 0.6):] = 0
        else:
            wrinkle_mask[:, :int(roi_w * 0.4)] = 0
    else:
        wrinkle_mask[:int(roi_h * 0.55), :] = 0

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L_eye, A_eye, B_eye = cv2.split(roi_lab)
    L_eye_f = L_eye.astype(np.float32)

    coarse = cv2.bilateralFilter(L_eye, d=9, sigmaColor=40, sigmaSpace=40).astype(np.float32)
    fine   = L_eye_f - coarse

    if is_crows:
        lift_strength = intensity * 45.0
    else:
        lift_strength = intensity * 65.0

    lift          = wrinkle_mask * lift_strength
    coarse_lifted = np.clip(coarse + lift, 0, 255)
    L_treated     = np.clip(coarse_lifted + fine * 0.25, 0, 255)

    L_eye_out = L_treated.astype(np.uint8)
    lab_out   = cv2.merge([L_eye_out, A_eye, B_eye])
    treated   = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32)

    blend_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.ellipse(
        blend_mask,
        (roi_w // 2, roi_h // 2),
        (int(roi_w * 0.45), int(roi_h * 0.42)),
        0, 0, 360, 255, -1
    )

    blur_size = max(int(min(roi_h, roi_w) * 0.27), 47) | 1

    blend_mask   = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0)
    blend_mask_f = (blend_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    blended = (
        roi_f + (treated - roi_f) * blend_mask_f * intensity
    ).astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = blended
    return result

def _treat_nasolabial_folds(image, lms, w, h, intensity=1.0):

    left_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in SMILE_LEFT_IDX],
        dtype=np.int32
    )
    right_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in SMILE_RIGHT_IDX],
        dtype=np.int32
    )

    upper_lip_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in UPPER_LIP_IDX],
        dtype=np.int32
    )
    lip_top_y = int(np.min(upper_lip_pts[:, 1]))

    result = image.copy()

    for pts in [left_pts, right_pts]:
        x1 = max(np.min(pts[:, 0]) - 70, 0)
        x2 = min(np.max(pts[:, 0]) + 70, w)
        y1 = max(np.min(pts[:, 1]) - 40, 0)
        y2 = min(min(np.max(pts[:, 1]) + 60, lip_top_y - 3), h)

        if y2 <= y1 or x2 <= x1:
            continue

        roi = result[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        roi_h, roi_w = roi.shape[:2]
        roi_f = roi.astype(np.float32)

        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(roi_lab)
        L_f = L.astype(np.float32)

        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
        blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel)
        blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)

        wrinkle_mask = np.clip(
            blackhat.astype(np.float32) / 255.0 * 8.0, 0, 1
        )

        coarse = cv2.bilateralFilter(L, d=9, sigmaColor=35, sigmaSpace=35).astype(np.float32)
        fine   = L_f - coarse

        lift          = wrinkle_mask * intensity * 60.0
        coarse_lifted = np.clip(coarse + lift, 0, 255)
        L_treated     = np.clip(coarse_lifted + fine * 0.50, 0, 255)

        L_naso_out = L_treated.astype(np.uint8)
        lab_naso   = cv2.merge([L_naso_out, A, B])
        treated_u8 = cv2.cvtColor(lab_naso, cv2.COLOR_LAB2BGR)

        clone_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.ellipse(clone_mask, (roi_w // 2, int(roi_h * 0.55)),
                    (int(roi_w * 0.42), int(roi_h * 0.48)), 0, 0, 360, 255, -1)
        bp = 6
        clone_mask[:bp, :] = 0; clone_mask[-bp:, :] = 0
        clone_mask[:, :bp] = 0; clone_mask[:, -bp:] = 0

        cx_sc = x1 + roi_w // 2
        cy_sc = y1 + roi_h // 2
        can_sc = (cx_sc >= roi_w // 2 + 1 and cx_sc <= w - roi_w // 2 - 1 and
                  cy_sc >= roi_h // 2 + 1 and cy_sc <= h - roi_h // 2 - 1)
        cloned = False
        if can_sc:
            try:
                result = cv2.seamlessClone(
                    treated_u8, result, clone_mask, (cx_sc, cy_sc), cv2.MIXED_CLONE
                )
                cloned = True
            except cv2.error:
                pass

        if not cloned:
            blur_size = max(int(min(roi_h, roi_w) * 0.30), 31) | 1
            mask_s  = cv2.GaussianBlur(clone_mask, (blur_size, blur_size), 0)
            mask_f3 = (mask_s.astype(np.float32) / 255.0)[:, :, np.newaxis]
            treated_f = treated_u8.astype(np.float32)
            blended = (roi_f + (treated_f - roi_f) * mask_f3 * intensity).astype(np.uint8)
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
    lip_top_y   = int(np.min(upper_lip_pts[:, 1]))
    lip_w       = lip_right_x - lip_left_x

    # Expanded philtrum ROI — stops just above the vermilion border
    x1 = max(lip_left_x  + int(lip_w * 0.05), 0)
    x2 = min(lip_right_x - int(lip_w * 0.05), w)
    y1 = max(lip_top_y - int(lip_w * 0.85), 0)
    y2 = min(lip_top_y + int(lip_w * 0.08), h)

    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0 or roi.shape[0] < 5:
        return image

    roi_h, roi_w = roi.shape[:2]
    roi_f = roi.astype(np.float32)

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)
    L_f = L.astype(np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    L_enhanced = clahe.apply(L)

    # Narrow vertical kernel detects perioral (lipstick-bleed) wrinkles
    kw = max(3, int(lip_w * 0.04) | 1)
    kh = max(int(roi_h * 0.35), 7) | 1
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    blackhat = cv2.morphologyEx(L_enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    peak = float(blackhat.max())
    wrinkle_mask = np.clip(
        blackhat.astype(np.float32) / max(peak, 1.0) * 3.0, 0, 1
    )[:, :, np.newaxis]

    coarse = cv2.bilateralFilter(
        L, d=9, sigmaColor=45, sigmaSpace=45
    ).astype(np.float32)
    fine = L_f - coarse

    lift          = wrinkle_mask * intensity * 70.0
    uniform_lift  = intensity * 20.0
    coarse_lifted = np.clip(coarse + lift.squeeze() + uniform_lift, 0, 255)
    L_treated     = np.clip(coarse_lifted + fine * 0.10, 0, 255)

    L_out   = L_treated.astype(np.uint8)
    lab_out = cv2.merge([L_out, A, B])
    treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

# Sine blend — fades at all four borders for invisible seam
    blend_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
    for r in range(roi_h):
        t = (r / max(roi_h - 1, 1)) * 1.4 - 0.0
        blend_mask[r, :] = max(np.sin(t * np.pi), 0)
    for c in range(roi_w):
        blend_mask[:, c] *= np.sin(c / max(roi_w - 1, 1) * np.pi)

    blend_mask_f = np.clip(blend_mask * intensity, 0, 1)[:, :, np.newaxis]

    treated_f = treated.astype(np.float32)
    blended   = np.clip(
        roi_f + (treated_f - roi_f) * blend_mask_f, 0, 255
    ).astype(np.uint8)

    result = image.copy()
    result[y1:y2, x1:x2] = blended
    return result


def _treat_marionette_lines(image, lms, w, h, intensity=1.0):
    if intensity <= 0.01:
        return image

    left_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in MARIONETTE_LEFT_IDX],
        dtype=np.int32
    )
    right_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in MARIONETTE_RIGHT_IDX],
        dtype=np.int32
    )

    mouth_y = max(int(lms[61].y * h), int(lms[291].y * h))
    result  = image.copy()

    for pts in [left_pts, right_pts]:
        x1 = max(np.min(pts[:, 0]) - 20, 0)
        x2 = min(np.max(pts[:, 0]) + 20, w)
        y1 = max(mouth_y - 5, 0)
        y2 = min(np.max(pts[:, 1]) + 25, h)

        if y2 <= y1 or x2 <= x1:
            continue

        roi = result[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        roi_h, roi_w = roi.shape[:2]
        roi_f = roi.astype(np.float32)

        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(roi_lab)
        L_f = L.astype(np.float32)

        # Find darkest vertical column — that's the fold shadow
        col_means  = np.mean(L_f, axis=0)
        col_smooth = cv2.GaussianBlur(
            col_means.reshape(1, -1).astype(np.float32), (1, 15), 0
        ).flatten()

        darkest_col = int(np.argmin(col_smooth))

        # Gaussian band mask centered on darkest column
        band_half   = max(int(roi_w * 0.20), 8)
        shadow_mask = np.zeros((roi_h, roi_w), dtype=np.float32)
        for c in range(roi_w):
            dist = abs(c - darkest_col)
            if dist < band_half:
                shadow_mask[:, c] = np.exp(
                    -(dist ** 2) / (2 * (band_half * 0.4) ** 2)
                )

        coarse = cv2.bilateralFilter(
            L, d=9, sigmaColor=35, sigmaSpace=35
        ).astype(np.float32)
        fine = L_f - coarse

        lift          = shadow_mask * intensity * 55.0
        coarse_lifted = np.clip(coarse + lift, 0, 255)
        L_treated     = np.clip(coarse_lifted + fine * 0.25, 0, 255)

        L_out   = L_treated.astype(np.uint8)
        lab_out = cv2.merge([L_out, A, B])
        treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32)

        blend_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.ellipse(blend_mask, (roi_w // 2, roi_h // 2),
                    (int(roi_w * 0.44), int(roi_h * 0.46)),
                    0, 0, 360, 255, -1)
        blur_size = max(int(min(roi_h, roi_w) * 0.25), 21) | 1
        mask_s  = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0)
        mask_f3 = (mask_s.astype(np.float32) / 255.0)[:, :, np.newaxis]

        blended = (roi_f + (treated - roi_f) * mask_f3 * intensity).astype(np.uint8)
        result[y1:y2, x1:x2] = blended

    return result

def _treat_cheek_lines(image, lms, w, h, intensity=0.4):
    if intensity <= 0.01:
        return image

    naso_left_pts   = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in SMILE_LEFT_IDX],  dtype=np.int32)
    naso_right_pts  = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in SMILE_RIGHT_IDX], dtype=np.int32)
    under_left_pts  = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in UNDER_LEFT_IDX],  dtype=np.int32)
    under_right_pts = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in UNDER_RIGHT_IDX], dtype=np.int32)

    face_left_x  = int(lms[234].x * w)
    face_right_x = int(lms[454].x * w)
    mouth_left_y  = int(lms[61].y  * h)
    mouth_right_y = int(lms[291].y * h)

    # Left cheek box: face edge to nasolabial outer, under-eye bottom to mouth corner
    lx1 = max(face_left_x, 0)
    lx2 = max(np.min(naso_left_pts[:, 0]) - 5, lx1 + 1)
    ly1 = max(np.max(under_left_pts[:, 1]) - 5, 0)
    ly2 = min(mouth_left_y + 5, h)

    # Right cheek box: nasolabial outer to face edge
    rx1 = min(np.max(naso_right_pts[:, 0]) + 5, w)
    rx2 = min(face_right_x, w)
    ry1 = max(np.max(under_right_pts[:, 1]) - 5, 0)
    ry2 = min(mouth_right_y + 5, h)

    result = image.copy()

    for (x1, x2, y1, y2) in [(lx1, lx2, ly1, ly2), (rx1, rx2, ry1, ry2)]:
        if y2 <= y1 or x2 <= x1:
            continue

        roi = result[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        roi_h, roi_w = roi.shape[:2]
        roi_f = roi.astype(np.float32)

        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(roi_lab)
        L_f = L.astype(np.float32)

        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        L_enhanced = clahe.apply(L)

        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        blackhat = cv2.morphologyEx(L_enhanced, cv2.MORPH_BLACKHAT, kernel)
        blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

        peak = float(blackhat.max())
        wrinkle_mask = np.clip(
            blackhat.astype(np.float32) / max(peak, 1.0) * 5.0, 0, 1
        )

        coarse = cv2.bilateralFilter(
            L, d=9, sigmaColor=30, sigmaSpace=30
        ).astype(np.float32)
        fine = L_f - coarse

        lift          = wrinkle_mask * intensity * 50.0
        coarse_lifted = np.clip(coarse + lift, 0, 255)
        L_treated     = np.clip(coarse_lifted + fine * 0.25, 0, 255)

        L_out   = L_treated.astype(np.uint8)
        lab_out = cv2.merge([L_out, A, B])
        treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32)

        blend_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.ellipse(blend_mask, (roi_w // 2, roi_h // 2),
                    (int(roi_w * 0.48), int(roi_h * 0.48)),
                    0, 0, 360, 255, -1)
        blur_size = max(int(min(roi_h, roi_w) * 0.25), 15) | 1
        mask_s  = cv2.GaussianBlur(blend_mask, (blur_size, blur_size), 0)
        mask_f3 = (mask_s.astype(np.float32) / 255.0)[:, :, np.newaxis]

        blended = (roi_f + (treated - roi_f) * mask_f3 * intensity).astype(np.uint8)
        result[y1:y2, x1:x2] = blended

    return result


def _treat_chin(image, lms, w, h, intensity):

    chin_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in CHIN_IDX],
        dtype=np.int32
    )

    top_y    = int(np.min(chin_pts[:, 1]))
    bottom_y = int(np.max(chin_pts[:, 1]))
    right_x  = int(np.max(chin_pts[:, 0]))
    chin_h   = bottom_y - top_y

    # Symmetric box around lm 152 — CHIN_IDX has lm 234 (far cheek) which skews left_x
    chin_cx = int(lms[152].x * w)
    half_w  = (right_x - chin_cx) + int((right_x - chin_cx) * 0.10)

    x1 = max(chin_cx - half_w, 0)
    x2 = min(chin_cx + half_w, w)
    y1 = max(top_y   + int(chin_h * 0.4),  0)
    y2 = min(bottom_y + int(chin_h * 0.5), h)

    if y2 <= y1 or x2 <= x1:
        return image

    roi          = image[y1:y2, x1:x2].copy()
    roi_h, roi_w = roi.shape[:2]
    roi_f        = roi.astype(np.float32)

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)
    L_f = L.astype(np.float32)

    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    L_enhanced = clahe.apply(L)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    blackhat = cv2.morphologyEx(L_enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    wrinkle_mask = np.clip(
        blackhat.astype(np.float32) / 255.0 * 6.0, 0, 1
    )
    wrinkle_mask[:int(roi_h * 0.12), :] = 0

    coarse = cv2.bilateralFilter(
        L, d=9, sigmaColor=30, sigmaSpace=30
    ).astype(np.float32)
    fine = L_f - coarse

    lift          = wrinkle_mask * intensity * 55.0
    coarse_lifted = np.clip(coarse + lift, 0, 255)
    uniform_lift  = intensity * 8.0
    coarse_lifted = np.clip(coarse_lifted + uniform_lift, 0, 255)
    L_treated     = np.clip(coarse_lifted + fine * 0.30, 0, 255)

    L_out   = L_treated.astype(np.uint8)
    lab_out = cv2.merge([L_out, A, B])
    treated = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    clone_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.ellipse(clone_mask, (roi_w // 2, roi_h // 2),
                (int(roi_w * 0.44), int(roi_h * 0.44)), 0, 0, 360, 255, -1)
    bp = 6
    clone_mask[:bp, :] = 0; clone_mask[-bp:, :] = 0
    clone_mask[:, :bp] = 0; clone_mask[:, -bp:] = 0

    cx_sc = x1 + roi_w // 2
    cy_sc = y1 + roi_h // 2
    can_sc = (cx_sc >= roi_w // 2 + 1 and cx_sc <= w - roi_w // 2 - 1 and
              cy_sc >= roi_h // 2 + 1 and cy_sc <= h - roi_h // 2 - 1)

    result = image.copy()
    cloned = False
    if can_sc:
        try:
            result = cv2.seamlessClone(
                treated, result, clone_mask, (cx_sc, cy_sc), cv2.MIXED_CLONE
            )
            cloned = True
        except cv2.error:
            pass

    if not cloned:
        blur_k = max(int(min(roi_h, roi_w) * 0.33), 23) | 1
        mask_s  = cv2.GaussianBlur(clone_mask, (blur_k, blur_k), 0)
        mask_f3 = (mask_s.astype(np.float32) / 255.0)[:, :, np.newaxis]
        treated_f = treated.astype(np.float32)
        blended   = (roi_f + (treated_f - roi_f) * mask_f3 * intensity).astype(np.uint8)
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
