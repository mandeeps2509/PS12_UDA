#!/bin/bash
set e

echo ">> Detecting GPU backend ..."
if command -v nvidia-smi &> /dev/null; then
    echo "Nvidia GPU detected -> using CUDA backend"
    export DEVICE="cuda"
elif python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q True; then
    echo "Apple MPS backend detected -> using MPS"
    export DEVICE="mps"
else
    echo "CPU mode"
    export DEVICE="cpu"
fi

echo "=> User: $(whoami) UID: $(id -u) GID: $(id -g)"

# python3 src/fine_tune_ast.py --device $DEVICE