import cv2
import numpy as np
import tempfile
import requests
import replicate
from PIL import Image


# =========================================
# AI UNDER EYE RECONSTRUCTION
# =========================================

def ai_under_eye_treatment(roi, intensity=0.7):
    """
    Takes a BGR numpy ROI crop,
    sends it to Replicate img2img,
    returns a BGR numpy array same size as input.
    """

    # 1. Save ROI to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).save(temp_file.name)

    # 2. Run CodeFormer on Replicate
    with open(temp_file.name, "rb") as f:
        output = replicate.run(
            "sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56",
            input={
                "image": f,
                "codeformer_fidelity": 0.5,
                "background_enhance": False,
                "face_upsample": True,
                "upscale": 1
            }
        )

    # 3. Download the result (output is a URL)
    if isinstance(output, list):
        url = output[0]
    else:
        url = output

    response = requests.get(url)
    img_array = np.frombuffer(response.content, np.uint8)
    result_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 4. Resize back to original ROI size
    result_resized = cv2.resize(result_rgb, (roi.shape[1], roi.shape[0]))

    # 5. Blend with original based on intensity
    blended = cv2.addWeighted(result_resized, intensity, roi, 1 - intensity, 0)

    return blended