from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np

from backend.utils.face_detect import detect_face_landmarks
from backend.features.lips_mesh import process_lips
from backend.features.lips_mesh_advanced import (
    process_lips_advanced,
    LipStyle,
    TintColor,
    TINT_COLORS,
    OUTER_LIP_IDX,
    INNER_LIP_IDX,
)
from backend.features.botox import (
    process_botox,
    FOREHEAD_IDX, EYEBROW_IDX, CROWS_LEFT_IDX, CROWS_RIGHT_IDX,
    UNDER_LEFT_IDX, UNDER_RIGHT_IDX, SMILE_LEFT_IDX, SMILE_RIGHT_IDX,
    UPPER_LIP_IDX, CHIN_IDX,
    MARIONETTE_LEFT_IDX, MARIONETTE_RIGHT_IDX,
)
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# HOME
# =========================================

@app.get("/")
def home():
    return {
        "message": "Beauty Simulation Engine Running 🚀"
    }


# =========================================
# TINT COLORS LIST
# =========================================

@app.get("/tint-colors")
def get_tint_colors():
    return {
        "colors": [
            {
                "id":    key,
                "label": key.replace("_", " ").title(),
                "rgb": {
                    "r": val[2],   # BGR → RGB
                    "g": val[1],
                    "b": val[0],
                }
            }
            for key, val in TINT_COLORS.items()
        ]
    }


# =========================================
# BASIC SIMULATION
# =========================================

@app.post("/simulate")
async def simulate(
    file: UploadFile = File(...),
    procedure: str = Form(...),
    intensity: float = Form(1.0)
):
    image_bytes = await file.read()

    if procedure == "lips":
        result_img = process_lips(image_bytes, intensity)

    elif procedure == "botox":
        result_img = process_botox(image_bytes)

    else:
        result_img = detect_face_landmarks(image_bytes)

    _, img_encoded = cv2.imencode('.jpg', result_img)

    return Response(
        content=img_encoded.tobytes(),
        media_type="image/jpeg"
    )


# =========================================
# ADVANCED LIP SIMULATION
# =========================================

@app.post("/simulate/advanced")
async def simulate_advanced(
    file: UploadFile = File(...),
    style: LipStyle = Form(LipStyle.NATURAL),
    intensity: float = Form(0.8),
    tint: bool = Form(False),
    tint_color: TintColor = Form(TintColor.rose),   # ← was str
    tint_strength: float = Form(0.5),
):
    image_bytes = await file.read()
    color_bgr = TINT_COLORS.get(tint_color.value, TINT_COLORS["rose"])  # ← .value

    result = process_lips_advanced(
        image_bytes=image_bytes,
        style=style,
        intensity=intensity,
        tint=tint,
        tint_color=color_bgr,
        tint_strength=tint_strength,
    )

    if result is None:
        raise HTTPException(status_code=400, detail="Could not process image")

    _, encoded = cv2.imencode(".jpg", result)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")




# =========================================
# BOTOX SIMULATION (FOREHEAD ONLY)
# =========================================
@app.post("/simulate/botox")
async def simulate_botox(
    file: UploadFile = File(...),

    forehead_intensity: float = Form(1.0),
    frown_intensity: float = Form(0.70),

    crows_intensity: float = Form(0.75),

    under_eye_intensity: float = Form(0.90),

    nasolabial_fold_intensity: float = Form(0.60),
    lip_intensity:             float = Form(0.60),
    marionette_intensity:      float = Form(0.60),
    chin_intensity:            float = Form(0.60),


    debug: bool = Form(False),
):

    image_bytes = await file.read()

    from backend.features.botox import process_botox

    result = process_botox(
        image_bytes=image_bytes,

        forehead_intensity=forehead_intensity,

        frown_intensity=frown_intensity,

        crows_intensity=crows_intensity,

        under_eye_intensity=under_eye_intensity,

        nasolabial_fold_intensity=nasolabial_fold_intensity,
        lip_intensity=lip_intensity,
        marionette_intensity=marionette_intensity,
        chin_intensity=chin_intensity,
        
        debug=debug,
    )

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Could not process image"
        )

    _, encoded = cv2.imencode(".jpg", result)

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg"
    )


# =========================================
# FILLER SIMULATION
# =========================================
@app.post("/simulate/filler")
async def simulate_filler(
    file:               UploadFile = File(...),
    cheek_intensity:    float      = Form(0.0),
    jawline_intensity:  float      = Form(0.0),
    chin_intensity:     float      = Form(0.0),
    raw:                bool       = Form(False),
):
    from backend.features.filler import process_filler
    image_bytes = await file.read()
    result = process_filler(
        image_bytes,
        cheek_intensity=cheek_intensity,
        jawline_intensity=jawline_intensity,
        chin_intensity=chin_intensity,
        side_by_side=not raw,
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Could not process image")
    _, encoded = cv2.imencode(".jpg", result)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


# =========================================
# BREAST SIMULATION
# =========================================
@app.post("/simulate/breast")
async def simulate_breast(
    file:           UploadFile = File(...),
    left_x:         int        = Form(...),
    left_y:         int        = Form(...),
    right_x:        int        = Form(...),
    right_y:        int        = Form(...),
    radius:         int        = Form(80),
    size_intensity: float      = Form(0.5),
    lift_intensity: float      = Form(0.0),
    shape:          str        = Form("round"),
    raw:            bool       = Form(False),
):
    from backend.features.breast import process_breast
    image_bytes = await file.read()
    result = process_breast(
        image_bytes, left_x, left_y, right_x, right_y,
        radius, size_intensity, lift_intensity, shape,
        side_by_side=not raw,
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Could not process image")
    _, encoded = cv2.imencode(".jpg", result)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.post("/landmarks")
async def get_landmarks(file: UploadFile = File(...)):

    image_bytes = await file.read()

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise HTTPException(status_code=400, detail="No face detected")
        lms = res.multi_face_landmarks[0].landmark

    def pts(indices):
        return [[int(lms[i].x * w), int(lms[i].y * h)] for i in indices]

    def box_from_pts(p_list, pad=0):
        xs = [p[0] for p in p_list]
        ys = [p[1] for p in p_list]
        return [
            max(min(xs) - pad, 0),
            max(min(ys) - pad, 0),
            min(max(xs) + pad, w),
            min(max(ys) + pad, h)
        ]

    # Compute all landmark point sets
    forehead_pts       = pts(FOREHEAD_IDX)
    eyebrow_pts        = pts(EYEBROW_IDX)
    crows_left_pts     = pts(CROWS_LEFT_IDX)
    crows_right_pts    = pts(CROWS_RIGHT_IDX)
    under_left_pts     = pts(UNDER_LEFT_IDX)
    under_right_pts    = pts(UNDER_RIGHT_IDX)
    smile_left_pts     = pts(SMILE_LEFT_IDX)
    smile_right_pts    = pts(SMILE_RIGHT_IDX)
    upper_lip_pts      = pts(UPPER_LIP_IDX)
    lip_outer_pts      = pts(OUTER_LIP_IDX)
    lip_inner_pts      = pts(INNER_LIP_IDX)
    chin_pts           = pts(CHIN_IDX)
    mario_left_pts     = pts(MARIONETTE_LEFT_IDX)
    mario_right_pts    = pts(MARIONETTE_RIGHT_IDX)

    # Forehead box
    top_y    = min(p[1] for p in forehead_pts)
    bottom_y = max(p[1] for p in eyebrow_pts)
    left_x   = min(p[0] for p in forehead_pts)
    right_x  = max(p[0] for p in forehead_pts)
    bh = bottom_y - top_y
    bw = right_x - left_x

    # Frown box
    LEFT_INNER = [55, 107]
    RIGHT_INNER = [285, 336]
    BROW_MID = [9, 8, 107, 336]
    left_brow  = pts(LEFT_INNER)
    right_brow = pts(RIGHT_INNER)
    mid_brow   = pts(BROW_MID)
    cx = int((sum(p[0] for p in left_brow)/len(left_brow) + sum(p[0] for p in right_brow)/len(right_brow)) / 2)
    frown_top = min(p[1] for p in mid_brow)
    frown_bot = max(p[1] for p in left_brow)
    zone_h = max(frown_bot - frown_top, 20)
    zone_w = int(abs(sum(p[0] for p in right_brow)/len(right_brow) - sum(p[0] for p in left_brow)/len(left_brow)) * 0.5)

    # Lip
    lip_left_x  = min(p[0] for p in upper_lip_pts)
    lip_right_x = max(p[0] for p in upper_lip_pts)
    lip_top_y   = min(p[1] for p in upper_lip_pts)
    lip_w       = lip_right_x - lip_left_x

    # Marionette (mouth corner to jaw)
    mouth_corner_y = max(int(lms[61].y * h), int(lms[291].y * h))

    # Chin
    chin_left_x  = min(p[0] for p in chin_pts)
    chin_right_x = max(p[0] for p in chin_pts)
    chin_top_y   = min(p[1] for p in chin_pts)
    chin_bot_y   = max(p[1] for p in chin_pts)
    chin_h       = chin_bot_y - chin_top_y
    chin_w_      = chin_right_x - chin_left_x
    chin_cx = int(lms[152].x * w)

    # Convex hull helper for clean polygons
    import cv2 as _cv2
    import numpy as _np

    def convex_poly(point_list):
        arr = _np.array(point_list, dtype=_np.int32)
        hull = _cv2.convexHull(arr)
        return hull.reshape(-1, 2).tolist()
    




    return {
        "image_size": {"width": w, "height": h},
        "regions": {
            "forehead": {
                "box": [
                    max(left_x - int(bw*0.05), 0),
                    max(top_y - int(bh*0.80), 0),
                    min(right_x + int(bw*0.05), w),
                    min(bottom_y, h)
                ],
                "poly": convex_poly(forehead_pts + eyebrow_pts),
                "label": "Forehead Lines",
                "label_side": "left"
            },
            "frown": {
                "box": [
                    max(cx - int(zone_w*0.85), 0),
                    max(frown_top - int(zone_h*2.5), 0),
                    min(cx + int(zone_w*0.85), w),
                    min(frown_bot + int(zone_h*0.45), h)
                ],
                "poly": convex_poly(pts([55, 107, 336, 285, 9, 8, 107, 336])),
                "label": "Frown Lines",
                "label_side": "right"
            },
            "crows_left": {
                "box": [
                    max(min(p[0] for p in crows_left_pts) - int((max(p[0] for p in crows_left_pts)-min(p[0] for p in crows_left_pts))*3.0), int(lms[234].x * w)),
                    max(min(p[1] for p in crows_left_pts) - 10, 0),
                    min(max(p[0] for p in crows_left_pts) + 10, w),
                    min(max(p[1] for p in crows_left_pts) + int((max(p[1] for p in crows_left_pts)-min(p[1] for p in crows_left_pts))*1.5), h)
                ],
                "poly": convex_poly(crows_left_pts),
                "label": "Crow's Feet",
                "label_side": "left"
            },
            "crows_right": {
                "box": [
                    max(min(p[0] for p in crows_right_pts) - 10, 0),
                    max(min(p[1] for p in crows_right_pts) - 10, 0),
                    min(max(p[0] for p in crows_right_pts) + int((max(p[0] for p in crows_right_pts)-min(p[0] for p in crows_right_pts))*3.0), int(lms[454].x * w)),
                    min(max(p[1] for p in crows_right_pts) + int((max(p[1] for p in crows_right_pts)-min(p[1] for p in crows_right_pts))*1.5), h)
                ],
                "poly": convex_poly(crows_right_pts),
                "label": "Crow's Feet",
                "label_side": "right"
            },
            "under_left": {
                "box": [
                    max(min(p[0] for p in under_left_pts) - int((max(p[0] for p in under_left_pts)-min(p[0] for p in under_left_pts))*0.5), 0),
                    max(max(p[1] for p in under_left_pts), 0),
                    min(max(p[0] for p in under_left_pts) + int((max(p[0] for p in under_left_pts)-min(p[0] for p in under_left_pts))*0.5), w),
                    min(max(p[1] for p in under_left_pts) + int((max(p[1] for p in under_left_pts)-min(p[1] for p in under_left_pts))*3.2), h)
                ],
                "poly": convex_poly(under_left_pts),
                "label": "Under Eye",
                "label_side": "left"
            },
            "under_right": {
                "box": [
                    max(min(p[0] for p in under_right_pts) - int((max(p[0] for p in under_right_pts)-min(p[0] for p in under_right_pts))*0.5), 0),
                    max(max(p[1] for p in under_right_pts), 0),
                    min(max(p[0] for p in under_right_pts) + int((max(p[0] for p in under_right_pts)-min(p[0] for p in under_right_pts))*0.5), w),
                    min(max(p[1] for p in under_right_pts) + int((max(p[1] for p in under_right_pts)-min(p[1] for p in under_right_pts))*3.2), h)
                ],
                "poly": convex_poly(under_right_pts),
                "label": "Under Eye",
                "label_side": "right"
            },
            "nasolabial_left": {
                "box": [
                    max(min(p[0] for p in smile_left_pts) - 30, 0),
                    max(min(p[1] for p in smile_left_pts) - 30, 0),
                    min(max(p[0] for p in smile_left_pts) + 30, w),
                    min(max(p[1] for p in smile_left_pts) + 10, min(lip_top_y - 5, h))
                ],
                "poly": convex_poly(smile_left_pts),
                "label": "Nasolabial Folds",
                "label_side": "left"
            },
            "nasolabial_right": {
                "box": [
                    max(min(p[0] for p in smile_right_pts) - 30, 0),
                    max(min(p[1] for p in smile_right_pts) - 30, 0),
                    min(max(p[0] for p in smile_right_pts) + 30, w),
                    min(max(p[1] for p in smile_right_pts) + 10, min(lip_top_y - 5, h))
                ],
                "poly": convex_poly(smile_right_pts),
                "label": "Nasolabial Folds",
                "label_side": "right"
            },
            "lip_lines": {
                "box": [
                    max(lip_left_x + int(lip_w*0.05), 0),
                    max(lip_top_y - int(lip_w*0.65), 0),
                    min(lip_right_x - int(lip_w*0.05), w),
                    max(lip_top_y - 2, 0)
                ],
                "poly": convex_poly(upper_lip_pts),
                "label": "Lip Lines",
                "label_side": "right"
            },
            "marionette_left": {
                "box": [
                    max(min(p[0] for p in mario_left_pts) - 15, 0),
                    mouth_corner_y + 8,
                    min(max(p[0] for p in mario_left_pts) + 15, w),
                    min(max(p[1] for p in mario_left_pts) + 20, h)
                ],
                "poly": convex_poly(mario_left_pts),
                "label": "Marionette Lines",
                "label_side": "left"
            },
            "marionette_right": {
                "box": [
                    max(min(p[0] for p in mario_right_pts) - 15, 0),
                    mouth_corner_y + 8,
                    min(max(p[0] for p in mario_right_pts) + 15, w),
                    min(max(p[1] for p in mario_right_pts) + 20, h)
                ],
                "poly": convex_poly(mario_right_pts),
                "label": "Marionette Lines",
                "label_side": "right"
            },
            
            "chin": {
                "box": [
                    max(chin_cx - int(chin_w_*0.35), 0),
                    max(chin_top_y + int(chin_h*0.4), 0),
                    min(chin_cx + int(chin_w_*0.35), w),
                    min(chin_bot_y + int(chin_h*0.5), h)
                ],
                "poly": convex_poly(chin_pts),
                "label": "Chin",
                "label_side": "left"
            },
            "lip_filler": {
                "box": box_from_pts(lip_outer_pts + lip_inner_pts, pad=15),
                "poly": convex_poly(lip_outer_pts),
                "label": "Lip Filler",
                "label_side": "right"
            },
            "cheek_left": {
                "box": box_from_pts(pts([116, 123, 147, 213, 192]), pad=20),
                "poly": convex_poly(pts([116, 123, 147, 213, 192])),
                "label": "Cheek Filler",
                "label_side": "left"
            },
            "cheek_right": {
                "box": box_from_pts(pts([345, 352, 376, 433, 416]), pad=20),
                "poly": convex_poly(pts([345, 352, 376, 433, 416])),
                "label": "Cheek Filler",
                "label_side": "right"
            },
            "jawline_left": {
                "box": box_from_pts(pts([172, 136, 150, 149, 176, 148]), pad=15),
                "poly": convex_poly(pts([172, 136, 150, 149, 176, 148])),
                "label": "Jawline",
                "label_side": "left"
            },
            "jawline_right": {
                "box": box_from_pts(pts([396, 365, 379, 378, 400, 377]), pad=15),
                "poly": convex_poly(pts([396, 365, 379, 378, 400, 377])),
                "label": "Jawline",
                "label_side": "right"
            },
            "chin_filler": {
                "box": box_from_pts(pts([152, 148, 377, 400, 378, 379]), pad=10),
                "poly": convex_poly(pts([152, 148, 377, 400, 378, 379])),
                "label": "Chin Filler",
                "label_side": "right"
            }
        }
    }