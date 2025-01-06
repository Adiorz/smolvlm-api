# smolvlm-api
A FastAPI application for running the SmolVLM-Instruct vision-language model from HuggingFace with GPU support. This project provides an API for image analysis and vision-language tasks using state-of-the-art optimizations for efficient inference.

## Model Details
The project uses the SmolVLM-Instruct model from HuggingFace, optimized for vision-language tasks. Key optimizations include:
- **Automatic device mapping:** Automatically detects and utilizes available GPUs.
- **Low CPU memory usage:** Efficient memory management for large models.
- **Disk offloading:** Handles large models by offloading parts to disk when necessary.
- **BFloat16 precision:** Reduces memory usage while maintaining model accuracy.

## Features
- **Asynchronous model loading:** Loads the model in the background with progress tracking.
- **GPU support:** Utilizes NVIDIA GPUs for accelerated inference.
- **Health check endpoint:** Provides real-time GPU status and model loading progress.
- **Image analysis:** Accepts custom prompts for image analysis tasks.
- **HuggingFace model caching:** Persists model files in a Docker volume for faster reloads.

## Development
### Prerequisites
- **Docker:** For containerized development and deployment.
- **Visual Studio Code:** Recommended IDE with Remote Development Extension.
- **NVIDIA GPU:** Required for GPU-accelerated inference (optional for CPU-only development).

### Development Setup
1. Build the Docker image:
   ```bash
   docker build -t smolvlm-api .
   ```
2. Open the project in VS Code:
- Open the project folder in Visual Studio Code.
- When prompted, click "Reopen in Container" or:
   - Press F1.
   - Select "Dev Containers: Reopen in Container".
This will:
- Use the existing smolvlm-api image.
- Install development dependencies.
- Set up pre-commit hooks.
- Configure VS Code with Python and Ruff extensions.
- Provide Docker access inside the container.
- Enable GPU support (if available).
3. Run the development server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Code Quality
The project uses Ruff for linting and formatting, configured in ```pyproject.toml```.
Pre-commit hooks are configured to run automatically on every commit.

## Production Deployment
### Prerequisites
- **Docker:* For containerized deployment.
- **NVIDIA GPU:** Required for GPU-accelerated inference.

## Building
Build the Docker image:
```bash
docker build -t smolvlm-api . --progress=plain --target production
```

## Running
Run the container with GPU support and HuggingFace cache persistence:
```bash
docker run --rm --gpus all -v huggingface_cache:/data/huggingface -p 8000:8000 smolvlm-api
```
This command:
- Uses NVIDIA GPU support (```--gpus all```)
- Mounts a persistent volume for HuggingFace models (huggingface_cache)
- Maps port 8000 to host
- Automatically removes the container when stopped (```--rm```)

## API Access
Once running, the API is available at: [http://localhost:8000](http://localhost:8000)
For detailed API documentation, visit [http://localhost:8000/docs](http://localhost:8000/docs) when the server is running.

## Configuration
The model configuration is defined in ```app/config.py```:
```python
MODEL_CONFIGS: Dict[str, Dict] = {
    "SmolVLM": {
        "name": "HuggingFaceTB/SmolVLM-Instruct",
        "processor_name": "HuggingFaceTB/SmolVLM-Instruct", 
    }
}
```
You can modify this to use different models or configurations.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgments
- HuggingFace for the SmolVLM-Instruct model and Transformers library.
- FastAPI for the web framework.
- NVIDIA for GPU support and optimizations.

Enjoy using the smolvlm-api! 🚀