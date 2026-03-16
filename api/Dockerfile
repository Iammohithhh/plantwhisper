FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API and the webapp backend it depends on
COPY app.py .
COPY ../webapp/backend.py ./webapp/backend.py

# If diffusion_model.pt is in the repo root, it gets copied here
COPY diffusion_model.pt* ./

ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
