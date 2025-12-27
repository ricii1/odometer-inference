from fastapi import FastAPI, HTTPException, UploadFile

app = FastAPI()

@app.post("/inference/upload/")
def inference(file: UploadFile):
    try: 
        print("Received file:", file.filename)
        if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg, .jpeg, and .png files are allowed.")
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=400, detail='Invalid file upload.')
    return {
        "mileage": 472, 
        "engineType": "EF32E", 
        "keterangan": "Foto valid: objek speedometer terlihat jelas dan sesuai"
    }

@app.post("/inference/base64/")
def inference_base64(file: str):
    return {
        "mileage": 472, 
        "engineType": "EF32E", 
        "keterangan": "Foto valid: objek speedometer terlihat jelas dan sesuai"
    }

@app.get("/")
def read_root():
    return {"message": "Inference API odometer mpm honda"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Inference API is running smoothly"}

@app.post("/inference/upload/false/")
def inference_false(file: UploadFile):
    if not file.filename.endswith('.jpg', '.jpeg', '.png'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg, .jpeg, and .png files are allowed.")
    return {
        "mileage": 1, 
        "engineType": "JBK1E", 
        "keterangan": "Foto tidak valid: objek speedometer tidak terlihat jelas atau tidak sesuai"
    }

@app.post("/inference/base64/false/")
def inference_base64_false(file: str):
    return {
        "mileage": 1, 
        "engineType": "JBK1E", 
        "keterangan": "Foto tidak valid: objek speedometer tidak terlihat jelas atau tidak sesuai"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
