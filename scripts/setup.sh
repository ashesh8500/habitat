#!/usr/bin/env bash
# One-time environment bootstrap for habitat-spatial-reasoning
# Run once after cloning the repository.
set -euo pipefail

echo "==> Setting up habitat-spatial-reasoning"

# ── UV environment ──────────────────────────────────────────────
echo ""
echo "--- Python environment (uv) ---"
uv sync
uv pip install clip depth-pro

# ── Conda environment for habitat-sim ────────────────────────────
echo ""
echo "--- Conda environment (habitat-sim) ---"
if conda env list | grep -q "^habitat-sim "; then
    echo "  habitat-sim conda env already exists"
else
    conda create -n habitat-sim python=3.9 -y \
      --override-channels -c conda-forge
    conda install -n habitat-sim habitat-sim -y \
      --override-channels -c aihabitat -c conda-forge
    conda run -n habitat-sim pip install habitat-lab numpy "pillow==10.4.0" opencv-python
fi

# ── Depth Pro model ─────────────────────────────────────────────
echo ""
echo "--- Depth Pro model ---"
DEPTH_PRO_DIR="/tmp/ml-depth-pro"
if [ ! -d "$DEPTH_PRO_DIR" ]; then
    git clone --depth 1 https://github.com/apple/ml-depth-pro.git "$DEPTH_PRO_DIR"
fi
if [ ! -f "$DEPTH_PRO_DIR/checkpoints/depth_pro.pt" ]; then
    echo "  Downloading model (1.8 GB)..."
    (cd "$DEPTH_PRO_DIR" && bash get_pretrained_models.sh)
fi
ln -sf "$DEPTH_PRO_DIR/checkpoints/depth_pro.pt" checkpoints/depth_pro.pt

echo ""
echo "==> Setup complete"
echo ""
echo "Next steps:"
echo "  1. Download HM3D data (requires Matterport token):"
echo "     conda run -n habitat-sim python -m habitat_sim.utils.datasets_download \\"
echo "       --username <token-id> --password <token-secret> \\"
echo "       --uids hm3d_minival_v0.2 --data-path data/"
echo ""
echo "  2. Run the pipeline:"
echo "     conda run -n habitat-sim python scripts/collect_trajectory.py \\"
echo "       --scene data/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
echo "     uv run habitat-detect"
echo "     uv run habitat-analyze"
