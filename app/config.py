from typing import Dict

MODEL_CONFIGS: Dict[str, Dict] = {
    "SmolVLM": {
        "name": "HuggingFaceTB/SmolVLM-Instruct",
        "processor_name": "HuggingFaceTB/SmolVLM-Instruct", 
    }
}

ACTIVE_MODEL = "SmolVLM"