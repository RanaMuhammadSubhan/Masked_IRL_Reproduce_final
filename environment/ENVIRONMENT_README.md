# Environment

Saved-prediction verification needs Python and NumPy only. The completed learners used Python 3.10.21, NumPy 1.26.4, PyTorch 2.5.0+cu124 and Transformers 4.48.1 on Windows with two RTX 3080 GPUs. Training requires CUDA; the scheduler accepts one or more GPU indices. Cross-platform execution is supported by relative-path adapters, but was not experimentally retrained on another OS.

## Review without training
```sh
python -m pip install numpy==1.26.4
python scripts/check_environment.py
python scripts/verify_dataset.py
python verification/verify_submission.py --package .
```

## Training environment
Use a fresh Python 3.10 environment. The exported environment.yml and conda-explicit-win-64.txt preserve the original Windows dependency record; they are not cross-platform lock files. requirements_frozen.txt is the original pip inventory with NumPy/packaging build paths replaced by versions 1.26.4/26.3; the local PyBullet build is documented as conda pybullet=3.25 rather than an unavailable pip wheel.

```sh
python -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 transformers==4.48.1 sentencepiece scipy matplotlib tqdm wandb
python scripts/check_environment.py --training --fetch-model
```

For exact dependency versions use the frozen inventory after installing the CUDA wheels. Some packages in the full environment served the historical simulator or notebooks and are unnecessary for verification. GPU driver compatibility remains a local setup responsibility.

The explicit --fetch-model command downloads public T5-base encoder/tokenizer assets at pinned revision a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1. No API key is needed. Default Hugging Face cache behavior or an explicit HF_HOME override is respected consistently by launchers and trainers. Training uses local_files_only=True after setup. Model weights, CUDA packages and environments are deliberately excluded from the ZIP. The delivered checkpoint files contain the trainable projection/reward network, not the frozen T5 backbone.

Extract into a short directory on Windows if your archive tool lacks long-path support. The supplied verifier enables Windows long paths internally. Use forward slashes in CLI paths; quote any path with spaces. Python launchers avoid shell-specific activation or PowerShell execution-policy changes.
