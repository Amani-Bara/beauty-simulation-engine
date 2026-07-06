"""Full pipeline test — all regions at default intensities."""
import cv2
import sys
from backend.features.botox import process_botox

img_path = sys.argv[1] if len(sys.argv) > 1 else "botox1.webp"

with open(img_path, "rb") as f:
    data = f.read()

out = process_botox(
    data,
    forehead_intensity        = 0.65,
    frown_intensity           = 0.50,
    crows_intensity           = 0.60,
    under_eye_intensity       = 0.60,
    nasolabial_fold_intensity = 0.60,
    lip_intensity             = 0.60,
    chin_intensity            = 0.60,
)

cv2.imwrite("result_full_pipeline.jpg", out)
print("Saved result_full_pipeline.jpg")
