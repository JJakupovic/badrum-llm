#!/usr/bin/env bash
# Run once on a fresh pod, from inside the cloned repository.
#
#   cd /workspace
#   git clone https://github.com/JJakupovic/badrum-llm.git
#   cd badrum-llm
#   bash runpod/setup.sh
#
# Assumes a PyTorch template, so torch and CUDA are already present and matched
# to the driver. Do not reinstall torch. Everything must live under /workspace to
# survive the pod stopping, and that only works if you attached a network volume.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "── environment ───────────────────────────────────────────"
python -c "import torch; print(f'torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "!! No GPU visible. You are on a CPU pod, or the driver is mismatched."
  echo "   Data generation still works; training will be very slow."
fi

# /workspace is where the network volume mounts. If the repo is not under it,
# the checkpoints die with the pod. Worth knowing before a long run, not after.
echo
echo "── storage ───────────────────────────────────────────────"
case "$REPO_ROOT" in
  /workspace/*) ;;
  *) echo "!! The repository is at $REPO_ROOT, outside /workspace."
     echo "   Checkpoints written here are DESTROYED when the pod terminates."
     echo "   Move it: mv $REPO_ROOT /workspace/" ;;
esac

if mountpoint -q /workspace 2>/dev/null; then
  echo "/workspace is a mounted volume. Data persists when the pod stops."
else
  echo "!! /workspace is NOT a mounted network volume."
  echo "   Stop now, attach one, and redeploy. Everything here dies with the pod."
fi
df -h /workspace 2>/dev/null | tail -1 || true

echo
echo "── dependencies ──────────────────────────────────────────"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo "installed requirements.txt (torch deliberately not touched)"

echo
echo "── dataset ───────────────────────────────────────────────"
python -m data.generate --products 400
python -m data.instructions

echo
echo "── tests ─────────────────────────────────────────────────"
python tests/test_data.py
python tests/test_instructions.py
python tests/test_evals.py

cat <<'EOF'

Ready. Start both experiments with:

  bash run_experiments.sh --exclude-holdout

Then stop the pod. Billing is per second and an idle GPU costs what a busy one
does. Community-cloud pods can also be reclaimed mid-run; every stage checkpoints
and reruns skip finished stages, so restarting the same command continues.
EOF
