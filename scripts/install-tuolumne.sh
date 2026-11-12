ml load python/3.13.2 rocm/7.14
# For mpi4py
ml cce/21.0.3 cray-mpich/9.1.0
python3 -m venv .venvs/scaffoldvenv-tuo
source .venvs/scaffoldvenv-tuo/bin/activate

# "--no-deps" because rocm libraries will be vendored by loading rocm/7.14
python -m pip install --no-deps --no-index \
    --find-links https://download.pytorch.org/whl/torch/ \
    --find-links https://download.pytorch.org/whl/triton-rocm \
    'torch==2.15.0+rocm7.14' triton-rocm

# ScaFFold's base dependencies and PyTorch's non-ROCm Python dependencies
python -m pip install --no-binary=mpi4py -e '.[torch]'

python -m pip install --no-deps "distconv @ git+https://github.com/LBANN/DistConv.git@084b20a"
