FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ ./data/
COPY models/ ./models/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY checkpoints/ ./checkpoints/
COPY results/ ./results/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["python", "api/main.py"]
