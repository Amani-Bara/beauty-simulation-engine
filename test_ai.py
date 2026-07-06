import cv2
import numpy as np
import requests
import replicate
import tempfile
from PIL import Image

# Load full image
img = cv2.imread("botox1.webp")

# 1. Save full image to temp file
temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(temp_file.name)

# 2. Send full image to CodeFormer
with open(temp_file.name, "rb") as f:
    output = replicate.run(
        "sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56",
        input={
            "image": f,
            "codeformer_fidelity": 0.3,
            "background_enhance": False,
            "face_upsample": True,
            "upscale": 1
        }
    )

# 3. Download result
url = output[0] if isinstance(output, list) else output
response = requests.get(url)
img_array = np.frombuffer(response.content, np.uint8)
result = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# 4. Resize to match original
result_resized = cv2.resize(result, (img.shape[1], img.shape[0]))

cv2.imwrite("result_full_face.jpg", result_resized)
print("Done! Check result_full_face.jpg")