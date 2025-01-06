import asyncio
import concurrent.futures
import logging
from functools import partial

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from app.config import ACTIVE_MODEL, MODEL_CONFIGS
from app.models.state import model_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_model():
    if model_state.is_loading or model_state.is_ready:
        return

    model_config = MODEL_CONFIGS[ACTIVE_MODEL]
    model_state.is_loading = True

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()

            model_state.update_progress("Loading processor")
            model_state.processor = await loop.run_in_executor(
                executor,
                partial(AutoProcessor.from_pretrained, model_config["processor_name"]),
            )

            model_state.update_progress("Loading model")
            model_state.model = await loop.run_in_executor(
                executor,
                partial(
                    AutoModelForVision2Seq.from_pretrained,
                    model_config["name"],
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    offload_folder="offload",
                ),
            )

            model_state.update_progress("Moving model to GPU")
            await loop.run_in_executor(
                executor, lambda: model_state.model.to(model_state.device)
            )

            model_state.is_ready = True
            model_state.update_progress("Ready (100%)")

    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        model_state.load_progress = f"Error: {str(e)}"
        raise e
    finally:
        model_state.is_loading = False
