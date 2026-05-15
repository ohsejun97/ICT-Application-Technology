#!/usr/bin/env bash
# ══════════════════════════════════════════════════════
# Agentic FusionDTI — 환경 설치 스크립트
# 사용법: bash setup_env.sh
# ══════════════════════════════════════════════════════

set -e

CONDA_ENV="bioinfo"
PYTHON_VERSION="3.10"

echo "================================================"
echo " Agentic FusionDTI 환경 설치 시작"
echo "================================================"

# ── conda 환경 확인 ────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "❌ conda가 설치되어 있지 않습니다."
    exit 1
fi

echo "[1] conda 환경: $CONDA_ENV"
conda activate $CONDA_ENV 2>/dev/null || true
PYTHON_BIN="$(conda run -n $CONDA_ENV which python3 2>/dev/null || echo python3)"

# ── PyTorch 설치 (CUDA 버전 자동 감지) ────────────────
echo ""
echo "[2] PyTorch 설치..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    echo "    CUDA $CUDA_VER 감지됨"
    if [ "$CUDA_VER" -ge "12" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_VER" -ge "11" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "    GPU 없음 — CPU 버전 설치"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ── 나머지 의존성 설치 ─────────────────────────────────
echo ""
echo "[3] 핵심 의존성 설치..."
pip install \
    transformers>=4.36.0 \
    accelerate>=0.24.0 \
    huggingface_hub>=0.20.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    tqdm>=4.65.0 \
    rdkit \
    bitsandbytes>=0.41.0

echo ""
echo "[4] 선택적 의존성 (에이전트/프론트엔드)..."
pip install \
    smolagents \
    fastapi uvicorn \
    streamlit \
    matplotlib seaborn || echo "⚠️  일부 선택적 패키지 설치 실패 (무시 가능)"

# ── 설치 확인 ──────────────────────────────────────────
echo ""
echo "[5] 설치 확인..."
python3 -c "
import torch; print(f'  ✅ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'  ✅ transformers {transformers.__version__}')
import pandas; print(f'  ✅ pandas {pandas.__version__}')
from rdkit import Chem; print('  ✅ rdkit')
from scipy.stats import pearsonr; print('  ✅ scipy')
from transformers import EsmModel, EsmTokenizer; print('  ✅ EsmModel, EsmTokenizer')
"

echo ""
echo "================================================"
echo " 설치 완료! 이제 실행 가능:"
echo " python run_reference.py"
echo "================================================"
