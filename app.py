from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI(title="CLIP Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# LAZY LOADING (Cloud Run Safe)
# ---------------------------------------------------------
model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        print("⏳ Loading CLIP model on demand...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP model ready!")
    return model, processor

# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"status": "running", "model_loaded": model is not None}


# ---------------------------------------------------------
# Image Encoding
# ---------------------------------------------------------
@app.post("/encode-image")
async def encode_image(file: UploadFile = File(...)):
    model, processor = load_model()

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return {"embedding": embedding.squeeze().tolist()}


# ---------------------------------------------------------
# Text Encoding
# ---------------------------------------------------------
@app.post("/encode-text")
async def encode_text(text: str = Form(...)):
    model, processor = load_model()

    inputs = processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        embedding = model.get_text_features(**inputs)

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return {"embedding": embedding.squeeze().tolist()}


# ---------------------------------------------------------
# Image Similarity
# ---------------------------------------------------------
@app.post("/similarity")
async def similarity(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    model, processor = load_model()

    img1 = Image.open(file1.file).convert("RGB")
    img2 = Image.open(file2.file).convert("RGB")

    inputs = processor(images=[img1, img2], return_tensors="pt")

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    sim = float(torch.mm(emb[0:1], emb[1:2].T))

    return {"similarity": sim}
