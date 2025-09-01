from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from deepface import DeepFace
import easyocr
from PIL import Image
import io, os, traceback

# Cache dirs
os.environ["HF_HOME"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache/hub"

app = FastAPI()

# BLIP
blip = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=-1
)

# ArcFace
arcface_model = "ArcFace"

# OCR
ocr_reader = easyocr.Reader(["en"])

# -----------------
# New helper for face comparison
# -----------------
def compare_faces(img1_path, img2_path, threshold=0.6):
    """Compare two faces with ArcFace embeddings"""
    try:
        result = DeepFace.verify(img1_path, img2_path, model_name=arcface_model, detector_backend="opencv")
        return {
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "model": arcface_model
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# -----------------
# Endpoints
# -----------------
@app.get("/")
def home():
    return {"message": "API is running. Use /predict_caption, /predict_gender, /predict_ocr, /verify_id, or /compare_faces."}

@app.post("/compare_faces")
async def compare_faces_api(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compare two faces and return similarity"""
    try:
        # Save both files temporarily
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()

        with open("face1.jpg", "wb") as f:
            f.write(img1_bytes)
        with open("face2.jpg", "wb") as f:
            f.write(img2_bytes)

        result = compare_faces("face1.jpg", "face2.jpg")

        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
