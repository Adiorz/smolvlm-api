import logging

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_model_files_info():
    try:
        api = HfApi()
        model_info = api.model_info("HuggingFaceTB/SmolVLM-Instruct")
        files = []
        for f in model_info.siblings:
            if f.rfilename.endswith(".bin") or f.rfilename.endswith(".json"):
                size = getattr(f, "size", None)
                if size is not None and isinstance(size, (int, float)):
                    files.append({"name": f.rfilename, "size": size})
                else:
                    logger.warning("No valid size for file %s", f.rfilename)
        return files
    except Exception as e:
        logger.error("Error getting model info: %s", str(e))
        return []
