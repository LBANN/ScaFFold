. install-rccl.sh
ml load python/3.11.5 && python3 -m venv .venvs/scaffoldvenv-tuo-pypi && source .venvs/scaffoldvenv-tuo-pypi/bin/activate && pip install --upgrade pip
ml cce/21.0.0 cray-mpich/9.1.0 rocm/7.1.0 rccl/fast-env-slows-mpi
pip install -e .[rocm] --prefix=.venvs/scaffoldvenv-tuo-pypi --extra-index-url https://download.pytorch.org/whl/rocm7.1 2>&1 | tee install.log
