from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline, CLIPProcessor, CLIPModel
from deepface import DeepFace
import easyocr
from PIL import Image
import io, os, traceback

# Ensure Hugging Face cache dirs are writable
os.environ["HF_HOME"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache/hub"

app = FastAPI()

# -----------------
# Load Models
# -----------------
# BLIP (captioning)
blip = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=-1
)

# ArcFace (via DeepFace) - lightweight, good for gender
arcface_model = "ArcFace"

# EasyOCR
ocr_reader = easyocr.Reader(["en"])

# CLIP (for future image similarity) - commented for now
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -----------------
# Endpoints
# -----------------
@app.get("/")
def home():
    return {"message": "API is running. Use /predict_caption, /predict_gender, /predict_ocr, or /verify_id."}


@app.post("/predict_caption")
async def predict_caption(file: UploadFile = File(...)):
    """BLIP captioning + keyword gender extraction"""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = blip(image, max_new_tokens=32)
        caption = result[0]["generated_text"].strip()

        # Keyword gender detection
        gender = "unknown"
        if "male" in caption.lower() or "man" in caption.lower():
            gender = "male"
        elif "female" in caption.lower() or "woman" in caption.lower():
            gender = "female"

        return JSONResponse({"caption": caption, "gender": gender})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict_gender")
async def predict_face_gender(file: UploadFile = File(...)):
    """ArcFace face-based gender classification"""
    try:
        image_bytes = await file.read()
        with open("temp.jpg", "wb") as f:
            f.write(image_bytes)

        analysis = DeepFace.analyze("temp.jpg", actions=['gender'], model_name=arcface_model)
        gender = analysis[0]["dominant_gender"]

        return JSONResponse({"gender": gender, "backend": arcface_model})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict_ocr")
async def predict_ocr(file: UploadFile = File(...)):
    """OCR on ID card text"""
    try:
        image_bytes = await file.read()
        with open("temp_ocr.jpg", "wb") as f:
            f.write(image_bytes)

        results = ocr_reader.readtext("temp_ocr.jpg", detail=0)
        sex_field = "unknown"

        for text in results:
            if "sex" in text.lower() or "gender" in text.lower():
                if "m" in text.lower():
                    sex_field = "male"
                elif "f" in text.lower():
                    sex_field = "female"

        return JSONResponse({"ocr_text": results, "sex_field": sex_field})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/verify_id")
async def verify_id(file: UploadFile = File(...)):
    """Combine BLIP, ArcFace, and OCR for ID verification"""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # BLIP
        blip_result = blip(image, max_new_tokens=32)
        caption = blip_result[0]["generated_text"].strip()
        blip_gender = "unknown"
        if "male" in caption.lower() or "man" in caption.lower():
            blip_gender = "male"
        elif "female" in caption.lower() or "woman" in caption.lower():
            blip_gender = "female"

        # ArcFace gender
        with open("temp_face.jpg", "wb") as f:
            f.write(image_bytes)
        analysis = DeepFace.analyze("temp_face.jpg", actions=['gender'], model_name=arcface_model)
        face_gender = analysis[0]["dominant_gender"]

        # OCR
        with open("temp_ocr.jpg", "wb") as f:
            f.write(image_bytes)
        ocr_results = ocr_reader.readtext("temp_ocr.jpg", detail=0)
        sex_field = "unknown"
        for text in ocr_results:
            if "sex" in text.lower() or "gender" in text.lower():
                if "m" in text.lower():
                    sex_field = "male"
                elif "f" in text.lower():
                    sex_field = "female"

        # Combine results
        return JSONResponse({
            "blip_caption": caption,
            "blip_gender": blip_gender,
            "face_gender": face_gender,
            "ocr_text": ocr_results,
            "ocr_sex_field": sex_field
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
