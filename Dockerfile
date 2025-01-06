FROM huggingface/transformers-pytorch-gpu:latest AS base

WORKDIR /app

COPY pyproject.toml .

ENV HF_HOME=/data/huggingface

RUN mkdir -p /data/huggingface

# Production stage
FROM base AS production
COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN pip install .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base AS development

RUN pip install -e ".[dev]"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]