ml load python/3.13.2
python3 -m venv .venvs/scaffoldvenv-matrix
source .venvs/scaffoldvenv-matrix/bin/activate

ml cuda/13.1.1 gcc/13.3.1 mvapich2/2.3.7
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Matrix-specific PyTorch and MPI builds
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu132 'torch==2.12.0+cu132'

# ScaFFold's shared dependencies come from pyproject.toml.
python -m pip instal --no-binary=mpi4py -e .
python -m pip install --no-deps 'distconv @ git+https://github.com/LBANN/DistConv.git@084b20a'
