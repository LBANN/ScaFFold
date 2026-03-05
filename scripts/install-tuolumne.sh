ml load python/3.11.5 && python3 -m venv .venvs/scaffoldvenv-tuo && source .venvs/scaffoldvenv-tuo/bin/activate && pip install --upgrade pip
ml cce/21.0.0 cray-mpich/9.1.0 rocm/7.1.0 rccl/fast-env-slows-mpi
pip install -e .[rocmwci] --prefix=.venvs/scaffoldvenv-tuo 2>&1 | tee install.log
# Needed until new wheel exists for torch using mpich 9.1.0
TORCH_LIB_DIR=".venvs/scaffoldvenv-tuo/lib/python3.11/site-packages/torch/lib"
OLD="libmpi_gnu_112.so.12"
NEW="libmpi_gnu.so.12"
cd "$TORCH_LIB_DIR" || exit 1
# Patch every file that has OLD in its DT_NEEDED
for f in *.so*; do
  [ -f "$f" ] || continue

  if patchelf --print-needed "$f" 2>/dev/null | grep -Fxq "$OLD"; then
    echo "Patching $f"
    patchelf --replace-needed "$OLD" "$NEW" "$f"
  fi
done
echo
echo "Verification (should show no $OLD):"
for f in *.so*; do
  [ -f "$f" ] || continue
  if patchelf --print-needed "$f" 2>/dev/null | grep -Fxq "$OLD"; then
    echo "STILL NEEDS $OLD -> $f"
  fi
done
cd -
