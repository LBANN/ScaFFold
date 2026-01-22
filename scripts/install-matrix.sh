ml load python/3.11.5 && python3 -m venv .venvs/scaffoldvenv-matrix && source .venvs/scaffoldvenv-matrix/bin/activate && pip install --upgrade pip
ml cuda/12.6.0 gcc/12.1.1 mvapich2/2.3.7
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
pip install --no-binary=mpi4py -e .[cuda] --prefix=.venvs/scaffoldvenv-matrix --extra-index-url https://download.pytorch.org/whl/cu126 2>&1 | tee install.log
