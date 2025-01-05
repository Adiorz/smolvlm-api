import typing

if typing.TYPE_CHECKING:
    from app.models.model_state import ModelState


class DownloadProgressCallback:
    def __init__(self, model_state: "ModelState"):
        self.model_state = model_state

    def __call__(self, downloaded: int, total: int, file: str):
        self.model_state.update_download_progress(downloaded, total, file)
