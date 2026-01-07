"""
Model Worker - Runs in a separate process to handle inference.
This process can be killed to completely free VRAM.
"""
import base64
import io
import json
import logging
import re
import sys
import torch
import pandas as pd
from typing import Dict, Any
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    filemode="w"
    filename="model_worker.log"
)
logger = logging.getLogger(__name__)

# Configuration
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
LORA_ADAPTER_PATH = "./lora-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).parent.resolve()
ODOMETER_CSV_PATH = SCRIPT_DIR / "odometer-engine-type" / "kode-odometer.csv"

# Prompt for comparing speedometer images
ENGINE_TYPE_PROMPT = """
Compare Image 1 and Image 2. 
Focus strictly on the speedometer/odometer shape, dial layout, and markings.
Are they the same model?
Answer absolutely ONLY with 'TRUE' or 'FALSE'. Do not explain.
"""


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text response."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"raw_output": text}
    except Exception:
        return {"raw_output": text}


def load_model():
    """Load model to GPU with detailed logging."""
    import time
    total_start = time.time()
    
    logger.info("=" * 50)
    logger.info("Starting model loading process...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Base model: {BASE_MODEL_ID}")
    logger.info(f"LoRA adapter: {LORA_ADAPTER_PATH}")
    logger.info("=" * 50)
    
    # Step 1: Load processor
    logger.info("[1/3] Loading processor...")
    step_start = time.time()
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True
    )
    logger.info(f"[1/3] Processor loaded in {time.time() - step_start:.2f}s")
    
    # Step 2: Load base model
    logger.info("[2/3] Loading base model to GPU (this may take a while)...")
    step_start = time.time()
    base_model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info(f"[2/3] Base model loaded in {time.time() - step_start:.2f}s")
    
    # Step 3: Load LoRA adapter
    logger.info("[3/3] Loading LoRA adapter...")
    step_start = time.time()
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()
    logger.info(f"[3/3] LoRA adapter loaded in {time.time() - step_start:.2f}s")
    
    total_time = time.time() - total_start
    logger.info("=" * 50)
    logger.info(f"Model loading complete! Total time: {total_time:.2f}s")
    logger.info("=" * 50)
    
    return model, processor


def compare_speedometer_images(model, processor, image: Image.Image, engine_type: str) -> bool:
    """Compare user's speedometer image with reference image based on engine type."""
    import time
    
    try:
        # Load odometer data
        df_data = pd.read_csv(ODOMETER_CSV_PATH)
        
        # Find reference image for engine type
        row = df_data[df_data['ENGINECODE'] == engine_type]
        
        if row.empty:
            logger.warning(f"Engine code '{engine_type}' not found in database.")
            return False
        
        reference_filename = row['SPEEDOMETERLINK'].values[0]
        reference_path = str(SCRIPT_DIR / "odometer-engine-type" / reference_filename)
        logger.info(f"Comparing with reference: {reference_path}")
        
        # Build messages for comparison
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "image", "image": reference_path},
                    {"type": "text", "text": ENGINE_TYPE_PROMPT},
                ],
            }
        ]
        
        # Process and run inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        logger.info(f"Comparison result: {output_text}")
        
        # Parse output to boolean
        clean_output = output_text.strip().upper()
        return "TRUE" in clean_output
        
    except Exception as e:
        logger.error(f"Error comparing speedometer images: {str(e)}")
        return False


def run_inference(model, processor, image_bytes: bytes, engine_type: str) -> Dict[str, Any]:
    """Run inference on image with timing."""
    import time
    start_time = time.time()
    
    logger.info("Starting inference...")
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.info(f"Image loaded: {image.size}")
    
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

    logger.info("Preparing inputs...")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(DEVICE)

    logger.info("Generating response...")
    gen_start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    
    gen_time = time.time() - gen_start
    logger.info(f"Generation completed in {gen_time:.2f}s")

    # Extract mileage and keterangan from JSON response
    result = extract_json(response_text)
    
    # Compare speedometer images if engine_type is provided
    is_engine_type_correct = False
    if engine_type:
        logger.info(f"Comparing speedometer for engine type: {engine_type}")
        is_engine_type_correct = compare_speedometer_images(model, processor, image, engine_type)
    
    total_time = time.time() - start_time
    logger.info(f"Total inference time: {total_time:.2f}s")
    
    # Build final response
    return {
        "mileage": result.get("mileage"),
        "isEngineTypeCorrect": is_engine_type_correct,
        "keterangan": result.get("keterangan", "")
    }


def main():
    """Main worker loop - reads commands from stdin, writes results to stdout."""
    model = None
    processor = None
    
    # Signal ready
    print(json.dumps({"status": "ready"}), flush=True)
    
    for line in sys.stdin:
        try:
            command = json.loads(line.strip())
            action = command.get("action")
            
            if action == "load":
                if model is None:
                    model, processor = load_model()
                    print(json.dumps({"status": "loaded"}), flush=True)
                else:
                    print(json.dumps({"status": "already_loaded"}), flush=True)
                    
            elif action == "inference":
                if model is None:
                    model, processor = load_model()
                
                image_b64 = command.get("image")
                engine_type = command.get("engine_type", "")
                image_bytes = base64.b64decode(image_b64)
                result = run_inference(model, processor, image_bytes, engine_type)
                print(json.dumps({"status": "success", "result": result}), flush=True)
                
            elif action == "health":
                print(json.dumps({
                    "status": "ok",
                    "model_loaded": model is not None,
                    "device": DEVICE
                }), flush=True)
                
            elif action == "exit":
                logger.info("Received exit command, shutting down...")
                print(json.dumps({"status": "exiting"}), flush=True)
                break
                
            else:
                print(json.dumps({"status": "error", "message": f"Unknown action: {action}"}), flush=True)
                
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": f"Invalid JSON: {str(e)}"}), flush=True)
        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            print(json.dumps({"status": "error", "message": str(e)}), flush=True)


if __name__ == "__main__":
    main()
