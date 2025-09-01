from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io, os

os.environ["HF_HOME"] = "/app/cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
os.environ["HF_HUB_CACHE"] = "/app/cache/hub"

app = FastAPI()

pipe = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=-1
)

@app.get("/")
def home():
    return {"message": "API is running. Use POST /predict with an image."}

@app.post("/predict")
async def predict_gender(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = pipe(image, max_new_tokens=32)
    caption = result[0]["generated_text"].strip()

    gender = "unknown"
    if "male" in caption.lower() or "man" in caption.lower():
        gender = "male"
    elif "female" in caption.lower() or "woman" in caption.lower():
        gender = "female"

    return JSONResponse({"caption": caption, "gender": gender})
