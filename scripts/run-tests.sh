#!/bin/bash
# Run the ScaFFold test suites with the environment they require.
#
#   scripts/run-tests.sh                    # both suites
#   scripts/run-tests.sh scaffold           # tests/ only
#   scripts/run-tests.sh triton             # triton_conv3d/tests/ only
#   scripts/run-tests.sh scaffold -x -k gn  # extra args go to pytest
#
# Set PYTHON to choose an interpreter; otherwise the first virtualenv under
# .venvs/ is used, falling back to python3 on PATH.
#
# Everything this script exports is here because omitting it changes the
# result, not because it seemed prudent.  See the comments at each one.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- interpreter ------------------------------------------------------------
# There is no editable install: the packages are importable from the repo root
# and nowhere else, hence the cd above and `python -m pytest` rather than a
# bare `pytest` (which would run from wherever its console script resolves).
if [ -z "${PYTHON:-}" ]; then
    for _venv in .venvs/*/bin/python; do
        [ -x "$_venv" ] && PYTHON="$_venv" && break
    done
    PYTHON="${PYTHON:-python3}"
fi
# --- required: channels-last has to reach MIOpen ----------------------------
# Without this, channels_last_3d is inert on ROCm and MIOpen is silently handed
# NCDHW -- a different problem than the one under test.  This is not a tuning
# preference: both parametrizations of
# tests/test_groupnorm.py::test_gpu_triton_dctensor_matches_eager_and_stays_wrapped
# fail deterministically when it is unset.  It is also what production runs set.
export PYTORCH_MIOPEN_SUGGEST_NHWC=1

# --- required: ROCm needs a writable TMPDIR ---------------------------------
# ROCm aborts the process (SIGABRT, no Python traceback) when it cannot write
# to TMPDIR, so an unwritable one reads as a crashed test run rather than as a
# configuration error.  Check it here, where the message can say so.
_tmp="${TMPDIR:-/tmp}"
if ! ( : > "$_tmp/.scaffold-write-probe.$$" ) 2>/dev/null; then
    echo "error: TMPDIR ($_tmp) is not writable; ROCm will abort the run." >&2
    echo "       Set TMPDIR to a writable directory and re-run." >&2
    exit 1
fi
rm -f "$_tmp/.scaffold-write-probe.$$"
export TMPDIR="$_tmp"

# --- if set, these caches have to be writable -------------------------------
# Neither Triton nor MIOpen fails when it cannot write its cache; both just
# redo the work every time.  Nothing reports it, so the suite reads as hung
# rather than as misconfigured.  Validate whatever the caller has set.
for _var in TRITON_CACHE_DIR MIOPEN_USER_DB_PATH MIOPEN_CUSTOM_CACHE_DIR; do
    _dir="${!_var:-}"
    [ -n "$_dir" ] || continue
    if ! mkdir -p "$_dir" 2>/dev/null ||
       ! ( : > "$_dir/.scaffold-write-probe.$$" ) 2>/dev/null; then
        echo "error: $_var ($_dir) is not writable; unset it or point it somewhere else." >&2
        exit 1
    fi
    rm -f "$_dir/.scaffold-write-probe.$$"
done

# --- runtime warning: a cold MIOpen find database dominates the run ---------
# The tests compare against MIOpen, and with an empty find database MIOpen
# searches for an algorithm per convolution problem instead of looking one up.
# Measured on triton_conv3d/tests/test_bwd_data.py (305 tests): 322 s cold
# against 14.7 s with a populated database -- 22x, and it is all search, not
# test work.  MIOPEN_USER_DB_PATH defaults to ~/.config/miopen; point it at a
# warm database to avoid paying this on every run.
_miopen_db="${MIOPEN_USER_DB_PATH:-$HOME/.config/miopen}"
if ! ls "$_miopen_db"/*.ufdb.txt >/dev/null 2>&1; then
    echo "note: MIOpen find database ($_miopen_db) is cold, so this run will be" >&2
    echo "      slow -- ~22x on the convolution tests, all of it algorithm search." >&2
    echo "      Set MIOPEN_USER_DB_PATH to a warm database to skip it." >&2
fi

# --- interpreter check + coverage warning -----------------------------------
# One import for both: torch is slow to load, and this is the only thing the
# script needs from it.  Runs after the exports above so it inherits them.
#
# The cross-device tests skip themselves when only one device is visible, so a
# one-device run reports a healthy pass count with those clauses never
# exercised.  Warn rather than fail: a one-device run is still worth doing, it
# is just not the full one.
_devices=$("$PYTHON" - <<'PY' 2>/dev/null
import ScaFFold, torch  # noqa: F401  -- import is the check
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
) || {
    echo "error: $PYTHON cannot import ScaFFold and torch." >&2
    echo "       Set PYTHON to the right interpreter, or run from the repo root." >&2
    exit 1
}
if [ "$_devices" -lt 2 ]; then
    echo "warning: $_devices GPU(s) visible; the cross-device tests in" >&2
    echo "         test_gather_gemm.py and test_bwd_weight.py will skip." >&2
    echo "         Two or more devices are needed for full coverage." >&2
fi

# --- run --------------------------------------------------------------------
# The mpi-marked tests skip themselves when no launcher is present, so they need
# no deselection here.
_suite="${1:-all}"
case "$_suite" in
    scaffold|triton|all) shift || true ;;
    *) _suite=all ;;
esac

_status=0
if [ "$_suite" = all ] || [ "$_suite" = scaffold ]; then
    echo "== ScaFFold suite =="
    "$PYTHON" -m pytest tests -q "$@" || _status=$?
fi
if { [ "$_suite" = all ] || [ "$_suite" = triton ]; } && [ -d triton_conv3d/tests ]; then
    echo "== triton_conv3d suite =="
    "$PYTHON" -m pytest triton_conv3d/tests -q "$@" || _status=$?
fi
exit $_status
