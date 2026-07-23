#!/usr/bin/env bash
set -euo pipefail

# 容器/生产用启动脚本（不使用 venv）
# 环境变量：
# - PORT: 监听端口（默认 7860）
# - HOST: 监听地址（默认 0.0.0.0）
# - DEVICE: 设备（默认 cpu）
# - OPEN_BROWSER: 是否在容器中尝试打开浏览器（默认 false）

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cpu}"
OPEN_BROWSER="${OPEN_BROWSER:-false}"

export PORT
export HOST
export DEVICE
export OPEN_BROWSER

echo "Starting ChatTTS (container mode) on ${HOST}:${PORT} using DEVICE=${DEVICE}"

exec python app.py
