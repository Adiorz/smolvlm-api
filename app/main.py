import asyncio
import io
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.models.state import model_state
from app.schemas.responses import HealthResponse
from app.services.model import initialize_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def start_model_loading():
    logger.info("Starting background model initialization...")
    try:
        await initialize_model()
        logger.info("Background model initialization complete")
    except Exception as e:
        logger.error(f"Error in background model loading: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading without awaiting
    loading_task = asyncio.create_task(initialize_model())

    # Let the application start immediately
    yield

    # Cleanup on shutdown
    if not loading_task.done():
        loading_task.cancel()
        try:
            await loading_task
        except asyncio.CancelledError:
            pass

    if model_state.model is not None:
        del model_state.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create the FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "message": "Server is running",
        "model_status": model_state.load_progress,
        "is_ready": model_state.is_ready,
    }


@app.get("/health")
async def health_check():
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**2:.2f} MB",
            "memory_cached": f"{torch.cuda.memory_reserved(0)/1024**2:.2f} MB",
        }

    return HealthResponse(
        status="ok",
        device=model_state.device,
        cuda_available=torch.cuda.is_available(),
        model_loaded=model_state.is_ready,
        loading_status=model_state.load_progress,
        loading_progress=model_state.download_progress,
        gpu_info=gpu_info,
    )


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...), prompt: str = "Describe this image"
):
    if not model_state.is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is still loading",
                "status": model_state.load_progress,
                "progress": model_state.download_progress,
                "download_info": {
                    "current_file": model_state.current_file,
                    "downloaded": f"{model_state.downloaded_bytes/1024**2:.1f} MB",
                    "total_size": f"{model_state.download_size/1024**2:.1f} MB",
                }
                if model_state.download_size > 0
                else None,
            },
        )

    try:
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content))

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        prompt = model_state.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = model_state.processor(
            text=prompt, images=[pil_image], return_tensors="pt"
        ).to(model_state.device)

        outputs = model_state.model.generate(**inputs, max_new_tokens=500)
        response = model_state.processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

        return {
            "response": response,
            "device": model_state.device,
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB"
            if torch.cuda.is_available()
            else "N/A",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
