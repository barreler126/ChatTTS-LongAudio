# ChatTTS-LongAudio — HF Spaces / Docker (CPU-only) adaptation

This PR makes the project easier to deploy on Hugging Face Spaces (Docker, no GPU) and in Linux containers by:

- Adding Dockerfile (CPU-only) that installs PyTorch CPU wheel and audio dependencies (ffmpeg, libsndfile).
- Adding start.sh (container-friendly entrypoint) and run.sh (local dev script with venv support).
- Updating app.py to default to CPU, load torch tensors with map_location when appropriate, and read HOST/PORT from environment variables (default host 0.0.0.0, default port 7860). It also disables auto-opening the browser in container mode by default.

Notes:
- The model is downloaded on first run (~1GB+). Consider placing the model in the repo's models/ or using a persistent cache in Spaces.
- requirements.txt is left unchanged; torch is installed in the Dockerfile explicitly as CPU wheel.

How to test locally with Docker:

1. Build:
   docker build -t chattts-space .
2. Run:
   docker run -e PORT=7860 -p 7860:7860 chattts-space
3. Visit: http://localhost:7860/

How to run locally without Docker:

1. chmod +x run.sh
2. ./run.sh

