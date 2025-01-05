FROM huggingface/transformers-pytorch-gpu:latest AS base

WORKDIR /app

# Copy project files
COPY pyproject.toml .

# Production stage
FROM base AS production
COPY app/ ./app/

ENV HF_HOME=/data/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN mkdir -p /data/huggingface
RUN pip install .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base AS development
RUN pip install -e ".[dev]"