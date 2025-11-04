python -m venv .venv
source .venv/bin/activate
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.zshrc
micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
pip install ninja
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade --no-cache-dir \
  tensordict torchdata triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb ninja liger-kernel==0.5.8 \
  pytest yapf py-spy pyext pre-commit ruff packaging wheel
pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download
pip install -e .
pip install -e lm-evaluation-harness human-eval-infilling