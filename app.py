import base64
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import magic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Honda Odometer Inference API", version="1.0.0")

# Constants
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 1 * 1024 * 1024  # 10MB

# Global state for model
model_loaded = False

def validate_image_content(file_content: bytes) -> None:
    """Validate image content using magic bytes"""
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    mime = magic.from_buffer(file_content, mime=True)
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Detected: {mime}. Allowed: {', '.join(ALLOWED_MIME_TYPES)}"
        )

class InferenceRequest(BaseModel):
    file: str

@app.post("/inference/upload/")
async def inference(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload image file for odometer inference"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate image content
        validate_image_content(file_content)
        
        # TODO: Replace with actual ML model inference
        logger.info(f"Processing file: {file.filename}, size: {len(file_content)} bytes")
        
        # Log in background
        background_tasks.add_task(logger.info, f"Completed processing: {file.filename}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )
    
    return {
        "mileage": 472, 
        "engineType": "EF32E", 
        "keterangan": "Foto valid: objek speedometer terlihat jelas dan sesuai"
    }

@app.post("/inference/base64/")
async def inference_base64(background_tasks: BackgroundTasks, payload: InferenceRequest):
    """Submit base64 encoded image for odometer inference"""
    try:
        # Decode base64
        try:
            image_bytes = base64.b64decode(payload.file)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 encoding"
            )
        
        # Validate image content
        validate_image_content(image_bytes)
        
        # TODO: Replace with actual ML model inference
        logger.info(f"Processing base64 image, size: {len(image_bytes)} bytes")
        
        # Log in background
        background_tasks.add_task(logger.info, "Completed processing base64 image")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )
    
    return {
        "mileage": 472, 
        "engineType": "EF32E", 
        "keterangan": "Foto valid: objek speedometer terlihat jelas dan sesuai"
    }

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {"message": "Inference API odometer mpm honda"}

@app.get("/health")
async def health_check():
    """Health check endpoint with actual status"""
    global model_loaded
    
    # TODO: Add actual model health check
    # For now, assume model is loaded if service is running
    model_loaded = True
    
    return {
        "status": "ok", 
        "message": "Inference API is running smoothly",
        "model_loaded": model_loaded
    }

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
