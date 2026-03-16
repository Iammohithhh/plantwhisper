FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API entrypoint and the shared backend module
COPY app.py .
COPY backend.py .

# If diffusion_model.pt is present, copy it
COPY diffusion_model.pt* ./

ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
