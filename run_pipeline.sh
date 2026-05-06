#!/usr/bin/env bash
# End-to-end habitat spatial reasoning pipeline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CONDA_ENV="habitat-sim"
CONDA_PYTHON="/opt/homebrew/anaconda3/envs/${CONDA_ENV}/bin/python"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

SCENE="${HABITAT_SCENE:-/tmp/habitat-sim/data/test_assets/scenes/simple_room.glb}"
FRAMES="${FRAMES:-25}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
CONFIDENCE="${CONFIDENCE:-0.15}"

echo "━━━ Habitat Spatial Reasoning Pipeline ━━━"
echo "Scene: $SCENE"
echo "Frames: $FRAMES at ${WIDTH}x${HEIGHT}"
echo ""

# ── Step 1: Collect trajectory ────────────────────────────────────
echo "📷 Step 1/3: Collecting egocentric trajectory..."
"$CONDA_PYTHON" "$PROJECT_DIR/scripts/collect_trajectory.py" \
  --scene "$SCENE" \
  --output "$PROJECT_DIR/output/frames" \
  --frames "$FRAMES" \
  --width "$WIDTH" \
  --height_px "$HEIGHT" \
  2>&1 | grep -vE "PluginManager|duplicate|Warning|OpenGL|Using|Renderer|workaround|GL_ARB|vertex_array|ES2|separate|texture_storage|filter_aniso|debug_label|debug_marker|layout-qual|apple-buffer" || true
echo ""

# ── Step 2: Object detection ──────────────────────────────────────
echo "🔍 Step 2/3: Running YOLO-World detection..."
"$VENV_PYTHON" "$PROJECT_DIR/scripts/detect_objects.py" \
  --input "$PROJECT_DIR/output/frames" \
  --output "$PROJECT_DIR/output/detections" \
  --confidence "$CONFIDENCE" \
  2>&1 | grep -v "^$\|^requirements:\|^Downloading\|^$" || true
echo ""

# ── Step 3: Depth + Segmentation + Analysis ──────────────────────
echo "📐 Step 3/3: Depth estimation + segmentation + analysis..."
"$VENV_PYTHON" "$PROJECT_DIR/scripts/analyze_scene.py" \
  --detections "$PROJECT_DIR/output/detections/detections.json" \
  --frames "$PROJECT_DIR/output/frames" \
  --output "$PROJECT_DIR/output/analysis" \
  2>&1 || true
echo ""

echo "━━━ Pipeline Complete ━━━"
echo "Outputs:"
echo "  Frames:       $PROJECT_DIR/output/frames/"
echo "  Detections:   $PROJECT_DIR/output/detections/detections.json"
echo "  Visuals:      $PROJECT_DIR/output/detections/annotated/"
echo "  Analysis:     $PROJECT_DIR/output/analysis/analysis.json"
echo "  Depth viz:    $PROJECT_DIR/output/analysis/visualizations/"
