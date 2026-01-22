ml load python/3.11.5 && python3 -m venv .venvs/scaffoldvenv-tuo && source .venvs/scaffoldvenv-tuo/bin/activate && pip install --upgrade pip
ml load rocm/6.4.2 rccl/fast-env-slows-mpi libfabric
pip install -e .[rocmwci] --prefix=.venvs/scaffoldvenv-tuo 2>&1 | tee install.log
