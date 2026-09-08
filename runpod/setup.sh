#!/usr/bin/env bash
# Run once on a fresh RunPod pod.
#
#   bash runpod/setup.sh
#
# Assumes a PyTorch template (torch + CUDA already present, so do not reinstall it,
# the image's build is matched to the driver). Everything lives under /workspace
# so it survives the pod being stopped, provided you attached a network volume.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT="$WORKSPACE/badrum-llm"

echo "── environment ───────────────────────────────────────────"
python -c "import torch; print(f'torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "!! No GPU visible. You are on a CPU pod, or the driver is mismatched."
  echo "   Data generation still works; training will not."
fi

# /workspace is the network volume mount point. If it isn't a separate device,
# there is no volume attached and everything dies with the pod. Worth knowing
# BEFORE a six-hour run, not after.
echo
echo "── storage ───────────────────────────────────────────────"
if mountpoint -q "$WORKSPACE" 2>/dev/null; then
  echo "$WORKSPACE is a mounted volume. Data persists when the pod stops."
else
  echo "!! $WORKSPACE is NOT a mounted network volume."
  echo "   Checkpoints will be DESTROYED when this pod terminates."
  echo "   Stop now, attach a network volume, and redeploy."
fi
df -h "$WORKSPACE" | tail -1

echo
echo "── project ───────────────────────────────────────────────"
mkdir -p "$PROJECT"
cd "$PROJECT"

python -m pip install --quiet --upgrade pip
if [ -f requirements.txt ]; then
  python -m pip install --quiet -r requirements.txt
  echo "installed requirements.txt"
fi

echo
echo "── dataset ───────────────────────────────────────────────"
python -m data.generate --products 400 --out "$PROJECT/data/out"

echo
echo "── tests ─────────────────────────────────────────────────"
python tests/test_data.py

cat <<'EOF'

Ready.

  Persist everything to /workspace. Anything written elsewhere dies with the pod.
  Checkpoint often; community-cloud pods can be reclaimed.
  Stop the pod the moment a run finishes. Billing is per second, and an idle
  GPU costs exactly as much as a busy one.
EOF
