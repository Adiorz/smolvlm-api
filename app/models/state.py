import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelState:
    def __init__(self):
        self.processor = None
        self.model = None
        self.is_loading = False
        self.load_progress = "Not started"
        self.is_ready = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.download_progress = 0
        self.total_steps = 4
        self.current_step = 0
        self.step_progress = 0
        self.download_size = 0
        self.downloaded_bytes = 0
        self.current_file = ""

    def update_progress(self, step_name: str, step_progress: float = 100):
        self.current_step += 1
        self.step_progress = step_progress
        total_progress = (
            (self.current_step - 1) * 100 + step_progress
        ) / self.total_steps
        self.load_progress = f"{step_name} ({total_progress:.1f}%)"
        self.download_progress = total_progress
        logger.info(f"Progress: {self.load_progress}")


model_state = ModelState()
