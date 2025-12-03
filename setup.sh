#!/bin/bash
set -e  # 에러 발생 시 중단

# -------------------------------------------------------
# 1. 설정 변수
# -------------------------------------------------------
ENV_NAME="20190980env"
PYTHON_VERSION="3.10"

# -------------------------------------------------------
# 2. Conda 초기화 및 기존 환경 삭제
# -------------------------------------------------------
source ~/.bashrc
eval "$(conda shell.bash hook)"

echo "========================================================"
echo " [Start] Clean Installation (One-by-One Mode)"
echo "========================================================"

conda deactivate 2> /dev/null || true

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "[Info] Removing existing environment..."
    conda remove -n "$ENV_NAME" --all -y
fi

# -------------------------------------------------------
# 3. 가상환경 생성
# -------------------------------------------------------
echo "[Info] Creating new environment..."
conda create -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION" -y

echo "[Info] Activating environment..."
conda activate "$ENV_NAME"

# -------------------------------------------------------
# 4. [1단계] 시스템 및 데이터 라이브러리 (하나씩 설치)
# -------------------------------------------------------
echo "--------------------------------------------------------"
echo " [Step 1] Installing Conda Packages One by One"
echo "--------------------------------------------------------"

# [중요] 버전 제약이 있는 패키지를 가장 먼저 설치해야 꼬이지 않습니다.

echo "[1/9] Installing NumPy (<2.0)..."
conda install -y --override-channels -c conda-forge "numpy<2.0"

echo "[2/9] Installing PyArrow (>=15.0)..."
conda install -y --override-channels -c conda-forge "pyarrow>=15.0.0"

echo "[3/9] Installing Pandas..."
conda install -y --override-channels -c conda-forge pandas

echo "[4/9] Installing CMake..."
conda install -y --override-channels -c conda-forge cmake

echo "[5/9] Installing Packaging..."
conda install -y --override-channels -c conda-forge packaging

echo "[6/9] Installing Psutil..."
conda install -y --override-channels -c conda-forge psutil

#echo "[7/9] Installing Matplotlib..."
#conda install -y --override-channels -c conda-forge matplotlib

echo "[8/9] Installing Graphviz (Binary)..."
conda install -y --override-channels -c conda-forge graphviz

echo "[9/9] Installing Python-Graphviz (Bindings)..."
conda install -y --override-channels -c conda-forge python-graphviz

# -------------------------------------------------------
# 5. [2단계] PyTorch 설치 (Pip + Index URL)
# -------------------------------------------------------
echo "--------------------------------------------------------"
echo " [Step 2] Installing PyTorch 2.5.1 (GPU) via Pip"
echo "--------------------------------------------------------"

echo "[Info] Upgrading pip..."
python -m pip install --upgrade pip
python -m pip install matplotlib

# GLIBC 호환성을 위한 공식 Wheel 설치
python -m pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# -------------------------------------------------------
# 6. [3단계] 나머지 Python 라이브러리 설치 (Pip)
# -------------------------------------------------------
echo "--------------------------------------------------------"
echo " [Step 3] Installing Remaining Python Libraries (Pip)"
echo "--------------------------------------------------------"

# 하나씩 설치하여 어디서 에러나는지 명확히 파악 (Pip은 빨라서 괜찮습니다)
python -m pip install "datasets>=2.19.0"
python -m pip install "transformers"
python -m pip install "seaborn"
python -m pip install "tensorboard"
python -m pip install "lion-pytorch"

echo ""
echo "========================================================"
echo " [Complete] Installation Finished Successfully!"
echo " Environment: $ENV_NAME"
echo "========================================================"

# -------------------------------------------------------
# 7. 검증
# -------------------------------------------------------
echo "[Check] Verifying Installation..."
python -c "
import torch
import numpy
import pyarrow
import datasets
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'NumPy Version: {numpy.__version__}')
print(f'PyArrow Version: {pyarrow.__version__}')
"
