#!/usr/bin/env bash
set -euo pipefail

# 本地开发运行脚本：创建 venv（若不存在），安装依赖并运行
VENV_DIR="venv"
PYTHON="${PYTHON:-python3}"
REQUIREMENTS="requirements.txt"
PORT="${PORT:-9966}"   # 本地默认端口 9966，可通过环境变量覆盖

echo "Using python: ${PYTHON}"
if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment in ${VENV_DIR}..."
  ${PYTHON} -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel
if [ -f "${REQUIREMENTS}" ]; then
  echo "Installing requirements from ${REQUIREMENTS}..."
  pip install -r "${REQUIREMENTS}"
fi

export PORT
export HOST="${HOST:-127.0.0.1}"
export DEVICE="${DEVICE:-cpu}"
export OPEN_BROWSER="${OPEN_BROWSER:-true}"

echo "Starting ChatTTS (dev mode) on ${HOST}:${PORT} using DEVICE=${DEVICE}"
python app.py
