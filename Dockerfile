FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV DEVICE=cpu
ENV OPEN_BROWSER=false
ENV PORT=7860

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      ffmpeg \
      libsndfile1 \
      wget \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel

# 安装 PyTorch CPU 版本（若需指定版本请调整）
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch

# 安装其余依赖（requirements.txt 中请不要包含带 CUDA 的 torch）
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 确保脚本可执行（如果使用 start.sh）
RUN chmod +x /app/start.sh || true

EXPOSE 7860

CMD ["./start.sh"]
