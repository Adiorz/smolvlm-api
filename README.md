# smolvlm-api
A FastAPI application for running SmolVLM-Instruct vision-language model with GPU support.

## Model Details
Uses the SmolVLM-Instruct model from HuggingFace for vision-language tasks with the following optimizations:
- Automatic device mapping
- Low CPU memory usage
- Disk offloading for large models
- BFloat16 precision

## Features
- Asynchronous model loading with progress tracking
- GPU support with memory management
- Health check endpoint with GPU status
- Image analysis with custom prompts
- HuggingFace model caching

## Development
### Prerequisites
- Docker
- Visual Studio Code
- VS Code Remote Development Extension

### Development Setup
1. First, build the base image:
   ```bash
   docker build -t smolvlm-api .
   ```
2. Open the project in VS Code
3. When prompted, click "Reopen in Container" or:
   - Press F1
   - Select "Dev Containers: Reopen in Container"

This will:
- Use the existing smolvlm-api image
- Install development dependencies
- Set up pre-commit hooks
- Configure VS Code with Python and Ruff extensions
- Provide Docker access inside the container
- Enable GPU support

### Code Quality
The project uses Ruff for linting and formatting, configured in pyproject.toml.
Pre-commit hooks are configured to run automatically on every commit.

## Production Deployment
### Prerequisites
- Docker
- NVIDIA GPU support for Docker

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
- Mounts a persistent volume for HuggingFace models (huggingface_cache) according to HF_HOME variable use in the Dockerfile
- Maps port 8000 to host
- Automatically removes the container when stopped (```--rm```)

## API Access
Once running, the API is available at: [http://localhost:8000](http://localhost:8000)

## API Endpoints
For detailed API documentation, visit [http://localhost:8000/docs](http://localhost:8000/docs) when the server is running.