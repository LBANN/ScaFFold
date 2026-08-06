ml load python/3.13.2 && python3 -m venv .venvs/scaffoldvenv-tuo && source .venvs/scaffoldvenv-tuo/bin/activate && pip install --upgrade pip
ml cce/21.0.2 cray-mpich/9.1.0 rocm/7.2.1 rccl/fast-env-slows-mpi
pip install -e .[rocm] --find-links https://download.pytorch.org/whl/torch/ --find-links https://download.pytorch.org/whl/torchaudio/ --find-links https://download.pytorch.org/whl/torchvision/ --find-links https://download.pytorch.org/whl/triton-rocm/ 2>&1 | tee install.log
