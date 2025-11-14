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

echo ">>> ls src/trained_ast_clf:"
ls src/trained_ast_clf

echo ">>> ls src/trained_ast_clf_lora:"
ls src/trained_ast_clf_lora

# inference
mkdir results
echo ">>> ls src"
ls src

# echo ">>> torchcodec.__version__"
python3 -c "import torchcodec; print('>>>> Imported torchcodec:', torchcodec.__version__);" 
python3 -c "from torchcodec.encoders import AudioEncoder; print('>>>> Imported AudioEncoder from torchcodec.encoders')"

python3 src/ast_anomaly_detection.py -ft True \
                                    -ftd "./src/trained_ast_clf_lora" \
                                    -td "./data/pre-eval/20251103_PS12" \
                                    -md "./src/trained_ast_clf" \
                                    -n 16 \
                                    -o "./results/20251103_S1_pre2_lora_n16_dithering_output_2.json"

# python3 -c "import json; with open('./results/20251103_S1_pre2_lora_n16_dithering_output_2.json') as f: print(json.load(f))'"