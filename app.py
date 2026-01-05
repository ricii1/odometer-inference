"""
FastAPI Application with subprocess-based model loading.
Model runs in a separate process that can be killed to completely free VRAM.
"""
import base64
import json
import logging
import io
import subprocess
import sys
import threading
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IDLE_TIMEOUT_SECONDS = 10  # Kill worker process after 5 minutes idle


class WorkerManager:
    """Manages the model worker subprocess."""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._idle_timer: Optional[threading.Timer] = None
        self._worker_ready = False
    
    def is_running(self) -> bool:
        """Check if worker process is running."""
        return self.process is not None and self.process.poll() is None
    
    def start_worker(self) -> bool:
        """Start the worker subprocess."""
        with self._lock:
            if self.is_running():
                self._update_last_used()
                return True
            
            try:
                logger.info("Starting model worker subprocess...")
                self.process = subprocess.Popen(
                    [sys.executable, "model_worker.py"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Wait for ready signal
                response = self._read_response(timeout=300)  # 5 min timeout for loading
                if response and response.get("status") == "ready":
                    self._worker_ready = True
                    logger.info("Worker subprocess started and ready.")
                    self._update_last_used()
                    self._schedule_idle_check()
                    return True
                else:
                    logger.error(f"Worker did not respond with ready: {response}")
                    self.stop_worker()
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to start worker: {str(e)}")
                self.stop_worker()
                return False
    
    def stop_worker(self):
        """Stop the worker subprocess to free VRAM."""
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()
                self._idle_timer = None
            
            if self.process is not None:
                logger.info("Stopping worker subprocess...")
                try:
                    # Send exit command
                    self._send_command({"action": "exit"})
                    self.process.wait(timeout=5)
                except Exception:
                    pass
                
                # Force kill if still running
                if self.process.poll() is None:
                    logger.info("Force killing worker...")
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
                self._worker_ready = False
                logger.info("Worker stopped, VRAM should be completely freed.")
    
    def _send_command(self, command: dict):
        """Send command to worker."""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(json.dumps(command) + "\n")
                self.process.stdin.flush()
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
                raise
    
    def _read_response(self, timeout: float = 120) -> Optional[dict]:
        """Read response from worker with timeout."""
        if not self.process or not self.process.stdout:
            return None
        
        import select
        
        # Use select for timeout on Linux
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if ready:
            try:
                line = self.process.stdout.readline()
                if line:
                    return json.loads(line.strip())
            except Exception as e:
                logger.error(f"Failed to read response: {e}")
        return None
    
    def _update_last_used(self):
        """Update last used timestamp."""
        self._last_used = time.time()
    
    def _schedule_idle_check(self):
        """Schedule idle check."""
        if self._idle_timer:
            self._idle_timer.cancel()
        
        def check_idle():
            if time.time() - self._last_used >= IDLE_TIMEOUT_SECONDS:
                logger.info(f"Worker idle for {IDLE_TIMEOUT_SECONDS}s, stopping...")
                self.stop_worker()
            elif self.is_running():
                self._schedule_idle_check()
        
        self._idle_timer = threading.Timer(IDLE_TIMEOUT_SECONDS, check_idle)
        self._idle_timer.daemon = True
        self._idle_timer.start()
    
    def run_inference(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run inference via worker subprocess."""
        # Ensure worker is running
        if not self.is_running():
            if not self.start_worker():
                raise RuntimeError("Failed to start worker process")
        
        with self._lock:
            try:
                # Send inference command
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                self._send_command({
                    "action": "inference",
                    "image": image_b64
                })
                
                # Wait for response (up to 2 minutes for inference)
                response = self._read_response(timeout=120)
                
                if response is None:
                    raise RuntimeError("No response from worker")
                
                if response.get("status") == "success":
                    self._update_last_used()
                    self._schedule_idle_check()
                    return response.get("result", {})
                elif response.get("status") == "error":
                    raise RuntimeError(response.get("message", "Unknown error"))
                else:
                    raise RuntimeError(f"Unexpected response: {response}")
                    
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                # Worker might be dead, stop it
                self.stop_worker()
                raise


# Global worker manager
worker_manager = WorkerManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Application starting...")
    yield
    # Shutdown: stop worker to free resources
    worker_manager.stop_worker()


app = FastAPI(
    title="Qwen3-VL Inference Engine",
    version="2.0.0",
    lifespan=lifespan
)


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
        return worker_manager.run_inference(content)
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
        return worker_manager.run_inference(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Base64 inference error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint with worker status."""
    return {
        "status": "ok",
        "worker_running": worker_manager.is_running()
    }


@app.post("/model/load")
async def manual_load_model():
    """Manually start worker and load model."""
    try:
        success = worker_manager.start_worker()
        if success:
            return {"status": "Worker started and model loaded"}
        raise HTTPException(status_code=503, detail="Failed to start worker")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def manual_unload_model():
    """Manually stop worker to free VRAM completely."""
    try:
        worker_manager.stop_worker()
        return {"status": "Worker stopped, VRAM freed completely"}
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
