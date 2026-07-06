"""Quick visual test — nasolabial folds only (all other intensities = 0)."""
import cv2
import sys
from backend.features.botox import process_botox

img_path = sys.argv[1] if len(sys.argv) > 1 else "botox1.webp"

with open(img_path, "rb") as f:
    data = f.read()

out = process_botox(
    data,
    forehead_intensity        = 0.0,
    frown_intensity           = 0.0,
    crows_intensity           = 0.0,
    under_eye_intensity       = 0.0,
    nasolabial_fold_intensity = 1.0,
    lip_intensity             = 0.0,
    chin_intensity            = 0.0,
)

cv2.imwrite("result_nasolabial.jpg", out)
print("Saved result_nasolabial.jpg")
