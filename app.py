import base64
import gc
import logging
import io
import json
import re
import threading
import time
import torch
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
LORA_ADAPTER_PATH = "./lora-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDLE_TIMEOUT_SECONDS = 10  # Unload model setelah 5 menit idle

# --- MODEL MANAGER ---
class ModelManager:
    """Manages lazy loading and unloading of the model to save VRAM."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._unload_timer: Optional[threading.Timer] = None
        self._models_downloaded = False
    
    def predownload_models(self) -> bool:
        """Pre-download models to cache without loading to GPU."""
        try:
            logger.info(f"Pre-downloading {BASE_MODEL_ID} to cache...")
            # Download processor (lightweight, keep in memory)
            self.processor = AutoProcessor.from_pretrained(
                BASE_MODEL_ID, 
                trust_remote_code=True
            )
            
            # Download model weights to cache only (don't load to GPU)
            # This downloads the files but doesn't allocate VRAM
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=BASE_MODEL_ID,
                ignore_patterns=["*.md", "*.txt"],
            )
            
            self._models_downloaded = True
            logger.info("Models pre-downloaded to cache successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to pre-download models: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self.model is not None
    
    def load_model(self) -> bool:
        """Load model to GPU. Returns True if successful."""
        with self._lock:
            if self.model is not None:
                self._update_last_used()
                return True
            
            try:
                logger.info(f"Loading {BASE_MODEL_ID} to {DEVICE}...")
                start_time = time.time()
                
                # Ensure processor is loaded
                if self.processor is None:
                    self.processor = AutoProcessor.from_pretrained(
                        BASE_MODEL_ID, 
                        trust_remote_code=True
                    )
                
                # Load base model to GPU
                base_model = AutoModelForVision2Seq.from_pretrained(
                    BASE_MODEL_ID,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Optimize memory during loading
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
                self.model.eval()
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {load_time:.2f}s")
                
                self._update_last_used()
                self._schedule_unload()
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self.model = None
                return False
    
    def unload_model(self):
        """Unload model from GPU to free VRAM."""
        with self._lock:
            if self.model is None:
                return
            
            logger.info("Unloading model from GPU...")
            
            # Cancel any pending unload timer
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
            
            # Delete model and clear GPU memory
            del self.model
            self.model = None
            
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("Model unloaded, VRAM freed.")
    
    def _update_last_used(self):
        """Update last used timestamp."""
        self._last_used = time.time()
    
    def _schedule_unload(self):
        """Schedule automatic unload after idle timeout."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        
        def check_and_unload():
            elapsed = time.time() - self._last_used
            if elapsed >= IDLE_TIMEOUT_SECONDS and self.model is not None:
                logger.info(f"Model idle for {elapsed:.0f}s, unloading...")
                self.unload_model()
        
        self._unload_timer = threading.Timer(IDLE_TIMEOUT_SECONDS + 1, check_and_unload)
        self._unload_timer.daemon = True
        self._unload_timer.start()
    
    def run_inference(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run inference, loading model if needed."""
        # Ensure model is loaded
        if not self.load_model():
            raise RuntimeError("Failed to load model")
        
        try:
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

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            
            # Update last used and reschedule unload
            self._update_last_used()
            self._schedule_unload()
            
            # Parse JSON response
            if "json" in user_prompt.lower():
                return extract_json(response_text)
            return {"response": response_text}
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise

# Global model manager instance
model_manager = ModelManager()


# --- LIFESPAN CONTEXT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Pre-download models to cache
    model_manager.predownload_models()
    yield
    # Shutdown: Unload model to free resources
    model_manager.unload_model()


app = FastAPI(
    title="Qwen3-VL Inference Engine", 
    version="1.3.0",
    lifespan=lifespan
)


# --- HELPER: JSON EXTRACTION ---
def extract_json(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"raw_output": text}
    except Exception:
        return {"raw_output": text}


# --- HELPER: VALIDATE IMAGE ---
def validate_image_content(content: bytes) -> bool:
    """Validate that content is a valid image."""
    try:
        Image.open(io.BytesIO(content))
        return True
    except Exception:
        return False


# --- ENDPOINTS ---
class Base64InferenceRequest(BaseModel):
    file: str  # base64 string


@app.post("/inference/upload/")
async def inference_upload(file: UploadFile = File(...)):
    """Takes a file via Form data and runs inference."""
    try:
        content = await file.read()
        return model_manager.run_inference(content)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Upload inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/base64/")
async def inference_base64(payload: Base64InferenceRequest):
    """Takes a JSON payload with a base64 image and runs inference."""
    try:
        image_bytes = base64.b64decode(payload.file)
        return model_manager.run_inference(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Base64 inference error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint with model status."""
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": model_manager.is_loaded(),
        "models_cached": model_manager._models_downloaded
    }


@app.post("/model/load")
async def manual_load_model():
    """Manually trigger model loading to GPU."""
    try:
        success = model_manager.load_model()
        if success:
            return {"status": "Model loaded successfully"}
        raise HTTPException(status_code=503, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def manual_unload_model():
    """Manually trigger model unloading from GPU."""
    try:
        model_manager.unload_model()
        return {"status": "Model unloaded, VRAM freed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/upload/false/")
async def inference_false(file: UploadFile = File(...)):
    """Dummy endpoint for testing false response."""
    file_content = await file.read()
    validate_image_content(file_content)
    return {
        "mileage": None,
        "engineType": "JBK1E",
        "keterangan": "Foto tidak valid: objek speedometer tidak terlihat jelas atau tidak sesuai"
    }


@app.post("/inference/base64/false/")
def inference_base64_false(file: str):
    """Dummy endpoint for testing false response with base64."""
    return {
        "mileage": None,
        "engineType": "JBK1E",
        "keterangan": "Foto tidak valid: objek speedometer tidak terlihat jelas atau tidak sesuai"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
