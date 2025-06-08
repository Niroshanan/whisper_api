from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import subprocess
import os
import shutil
import uuid

app = FastAPI()

WHISPER_CLI_PATH = os.path.join("Release", "whisper-cli.exe")
MODEL_PATH = os.path.join("models", "ggml-base.bin")  # ✅ Use multilingual model
SAMPLES_DIR = "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

@app.post("/translate")
async def translate(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(SAMPLES_DIR, f"{file_id}_{file.filename}")

    # Save uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Run whisper-cli with translation task
        result = subprocess.run(
            [
                WHISPER_CLI_PATH,
                "-m", MODEL_PATH,
                "-f", file_path,
                "--task", "translate",
                "--language", "ta"  # Tamil
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )

        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr.strip()})

        output = result.stdout.strip()

        if not output:
            return JSONResponse(status_code=500, content={"error": "Translation failed or returned empty result."})

        return {"translated_text": output}

    finally:
        os.remove(file_path)  # Clean up uploaded file
