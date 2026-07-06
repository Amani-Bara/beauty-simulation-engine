"""
Test all three breast shapes and verify visible differences.
Usage: python test_shapes.py [image_path]
Default image: breast.jpeg
"""
import sys
import cv2
import numpy as np

sys.path.insert(0, ".")
from backend.features.breast import process_breast, SHAPES

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "breast.jpeg"

img = cv2.imdecode(
    np.frombuffer(open(IMAGE_PATH, "rb").read(), np.uint8),
    cv2.IMREAD_COLOR,
)
assert img is not None, f"Could not load {IMAGE_PATH}"
h, w = img.shape[:2]
print(f"\nImage: {IMAGE_PATH}  size={w}x{h}\n")

# Approximate breast centres — adjust if wrong for your image
left_x  = int(w * 0.36)
left_y  = int(h * 0.42)
right_x = int(w * 0.64)
right_y = int(h * 0.42)
radius  = int(w * 0.13)   # ~13 % of image width

print(f"Marks: left=({left_x},{left_y})  right=({right_x},{right_y})  radius={radius}")

with open(IMAGE_PATH, "rb") as f:
    raw = f.read()

results = {}
for shape in ("round", "teardrop", "natural"):
    print(f"\n{'='*60}")
    print(f"SHAPE: {shape}")
    out = process_breast(
        raw,
        left_x, left_y, right_x, right_y,
        radius,
        size_intensity=0.8,
        lift_intensity=0.0,
        shape=shape,
        side_by_side=True,
    )
    assert out is not None
    results[shape] = out
    path = f"result_breast_{shape}_fixed.jpg"
    cv2.imwrite(path, out)
    print(f"  Saved -> {path}")

# ── pixel-diff diagnostics ──────────────────────────────────────────────────
# Extract the AFTER panel from each side-by-side (right half)
afters = {}
for shape, sbs in results.items():
    _, sw = sbs.shape[:2]
    panel_w = sw // 2
    afters[shape] = sbs[:, panel_w:].astype(np.float32)

orig = afters["round"]  # use round as baseline
for shape in ("teardrop", "natural"):
    diff = np.abs(afters[shape] - orig).mean()
    print(f"\nmean pixel diff  round vs {shape}: {diff:.4f}")

# ── combined comparison: all three side-by-side pairs stacked vertically ────
combined = np.vstack([results["round"], results["teardrop"], results["natural"]])

# Add shape labels on the left edge
bar_h, bar_w = results["round"].shape[:2]
for i, label in enumerate(("ROUND", "TEARDROP", "NATURAL")):
    y_off = i * bar_h + bar_h // 2
    cv2.putText(
        combined, label,
        (8, y_off),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 200, 255), 2,
    )

cv2.imwrite("result_shapes_comparison.jpg", combined)
print("\nCombined comparison -> result_shapes_comparison.jpg")
print("\nDone. Open the *_fixed.jpg files to visually verify shapes differ.")
