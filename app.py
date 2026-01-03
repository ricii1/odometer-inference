import base64
import logging
import io
import json
import re
import torch
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-VL Inference Engine", version="1.2.0")

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
LORA_ADAPTER_PATH = "./checkpoint-1500" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
processor = None
model_loaded = False

@app.on_event("startup")
async def load_model_on_startup():
    global model, processor, model_loaded
    try:
        logger.info(f"Loading {BASE_MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        base_model = AutoModelForVision2Seq.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        # Load your fine-tuned weights
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        model.eval()
        model_loaded = True
        logger.info("Model and Adapter loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

# --- HELPER: JSON EXTRACTION ---
def extract_json(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"raw_output": text}
    except Exception:
        return {"raw_output": text}

# --- INFERENCE CORE ---
def run_qwen3_inference(image_bytes: bytes) -> Dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    user_prompt = "Ekstrak data mileage dan tipe mesin dari gambar speedometer motor ini. Format Output: JSON."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    # If the user asked for JSON in their prompt, we try to parse it
    if "json" in user_prompt.lower():
        return extract_json(response_text)
    return {"response": response_text}

# --- ENDPOINTS ---
class Base64InferenceRequest(BaseModel):
    file: str  # base64 string

@app.post("/inference/upload/")
async def inference_upload(
    file: UploadFile = File(...), 
):
    """Takes a file and a prompt via Form data."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model starting up...")
    
    content = await file.read()
    return run_qwen3_inference(content)

@app.post("/inference/base64/")
async def inference_base64(payload: Base64InferenceRequest):
    """Takes a JSON payload with a base64 image and a prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model starting up...")
    
    try:
        image_bytes = base64.b64decode(payload.file)
        return run_qwen3_inference(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok" if model_loaded else "loading", "device": DEVICE}

@app.post("/inference/upload/false/")
async def inference_false(file: UploadFile = File(...)):
    """Dummy endpoint for testing false response"""
    # Read file content even if not used
    file_content = await file.read()
    
    # Validate with proper tuple syntax
    validate_image_content(file_content)
    
    return {
        "mileage": None, 
        "engineType": "JBK1E", 
        "keterangan": "Foto tidak valid: objek speedometer tidak terlihat jelas atau tidak sesuai"
    }

@app.post("/inference/base64/false/")
def inference_base64_false(file: str):
    return {
        "mileage": None, 
        "engineType": "JBK1E", 
        "keterangan": "Foto tidak valid: objek speedometer tidak terlihat jelas atau tidak sesuai"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)