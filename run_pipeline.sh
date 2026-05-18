#!/usr/bin/env bash
# Run the full Habitat spatial reasoning pipeline with the correct environment
# for each stage.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SCENE="${SCENE:-data/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb}"
OUTPUT_DIR="${OUTPUT_DIR:-output/pipeline_run}"
FRAMES="${FRAMES:-26}"
CONFIDENCE="${CONFIDENCE:-0.15}"
DEVICE="${DEVICE:-mps}"
MAX_ANALYSIS_FRAMES="${MAX_ANALYSIS_FRAMES:-0}"
HABITAT_PYTHON="${HABITAT_PYTHON:-/opt/homebrew/anaconda3/envs/habitat-sim/bin/python}"

mkdir -p "$OUTPUT_DIR"

echo "==> Habitat spatial reasoning pipeline"
echo "Scene: $SCENE"
echo "Output: $OUTPUT_DIR"
echo "Frames: $FRAMES"
echo "Confidence: $CONFIDENCE"
echo "Analysis frames: $MAX_ANALYSIS_FRAMES (0 = all)"
echo ""

echo "==> Stage 1/3: Habitat-Sim trajectory collection"
if [ -x "$HABITAT_PYTHON" ]; then
  "$HABITAT_PYTHON" scripts/collect_trajectory.py \
    --scene "$SCENE" \
    --output "$OUTPUT_DIR/frames" \
    --frames "$FRAMES" \
    --height 1.5 \
    --step 0.5 \
    --width 640 \
    --height_px 480
else
  conda run -n habitat-sim python scripts/collect_trajectory.py \
  --scene "$SCENE" \
  --output "$OUTPUT_DIR/frames" \
  --frames "$FRAMES" \
  --height 1.5 \
  --step 0.5 \
  --width 640 \
  --height_px 480
fi

echo ""
echo "==> Stage 2/3: YOLO-World detection"
uv run python scripts/detect_objects.py \
  --input "$OUTPUT_DIR/frames" \
  --output "$OUTPUT_DIR/detections" \
  --model yolov8s-worldv2.pt \
  --confidence "$CONFIDENCE" \
  --device "$DEVICE"

echo ""
echo "==> Stage 3/3: Depth Pro + FastSAM analysis"
uv run python scripts/analyze_scene.py \
  --frames "$OUTPUT_DIR/frames" \
  --detections "$OUTPUT_DIR/detections/detections.json" \
  --output "$OUTPUT_DIR/analysis" \
  --device "$DEVICE" \
  --max-frames "$MAX_ANALYSIS_FRAMES"

echo ""
echo "==> Done"
echo "Frames: $OUTPUT_DIR/frames"
echo "Detections: $OUTPUT_DIR/detections"
echo "Analysis: $OUTPUT_DIR/analysis"
