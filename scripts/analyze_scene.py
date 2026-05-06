#!/usr/bin/env python3
"""
Combined depth estimation and object segmentation, integrated with
YOLO-World detection results.

This script:
1. Loads detection results from detect_objects.py output
2. Runs Depth Pro for pixel-level monocular depth on each frame
3. Runs FastSAM (ultralytics) for object segmentation on each frame
4. Aligns segmentation masks with detection bounding boxes
5. Computes per-object depth statistics
6. Produces a combined analysis JSON and visualizations

Usage:
    python scripts/analyze_scene.py --detections output/detections/detections.json \\
                                    --frames output/frames \\
                                    --output output/analysis

Note: Requires depth_pro to be installed (conda env recommended).
      Also uses ultralytics FastSAM for segmentation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Depth Pro imports ──────────────────────────────────────────────
try:
    import torch
    from depth_pro import create_model_and_transforms
    HAS_DEPTH_PRO = True
except ImportError:
    HAS_DEPTH_PRO = False
    print("WARNING: depth_pro not available. Depth estimation will be skipped.")
    print("  Install: pip install /tmp/ml-depth-pro && download model with get_pretrained_models.sh")

# ── Ultralytics FastSAM ────────────────────────────────────────────
try:
    from ultralytics import SAM, FastSAM
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("WARNING: ultralytics SAM not available. Segmentation will be skipped.")
    print("  Install: pip install ultralytics")

# ── Defaults ───────────────────────────────────────────────────────
DEPTH_PRO_MODEL_DIR = Path("/tmp/ml-depth-pro/checkpoints")


def load_depth_model(device: str = "cpu") -> Tuple:
    """Load Depth Pro model."""
    if not HAS_DEPTH_PRO:
        raise RuntimeError("depth_pro not installed")
    model_path = DEPTH_PRO_MODEL_DIR / "depth_pro.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Depth Pro model not found at {model_path}. "
            f"Run: cd /tmp/ml-depth-pro && bash get_pretrained_models.sh"
        )
    os.environ.setdefault("DEPTH_PRO_CHECKPOINT_PATH", str(model_path.parent))
    model, transform = create_model_and_transforms(
        device=torch.device(device),
        precision=torch.float32,
    )
    model.eval()
    return model, transform


def load_sam_model(device: str = "cpu") -> SAM:
    """Load FastSAM or SAM model for segmentation."""
    if not HAS_SAM:
        raise RuntimeError("ultralytics not installed")

    # Try FastSAM first (lighter, faster)
    try:
        model = FastSAM("FastSAM-s.pt")
        print("  Using FastSAM-s (lightweight)")
    except Exception:
        # Fallback to standard SAM
        model = SAM("sam_b.pt")
        print("  Using SAM-b (standard)")
    return model


def estimate_depth(model, transform, image_bgr: np.ndarray) -> np.ndarray:
    """Run Depth Pro on a single image.
    
    Args:
        image_bgr: (H, W, 3) BGR image as read by cv2
        
    Returns:
        depth_map: (H, W) float32 depth in meters
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Transform and predict
    transformed = transform(image_rgb)
    with torch.no_grad():
        prediction = model.infer(transformed, f_px=None)
    
    depth = prediction["depth"].cpu().numpy()  # (H, W) in meters
    return depth.astype(np.float32)


def segment_objects(sam_model, image_bgr: np.ndarray, 
                    detections: List[Dict]) -> List[Dict]:
    """Segment objects detected by YOLO-World using prompts from detections.
    
    Uses bounding boxes as prompts for SAM/FastSAM.
    """
    if not detections:
        return []
    
    results = []
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        
        # Use the bounding box as a prompt for SAM
        try:
            sam_result = sam_model(
                image_bgr.copy(),
                bboxes=[det["bbox"]],
                verbose=False,
            )
        except Exception as e:
            print(f"    SAM failed for {det['class']}: {e}")
            continue
        
        # Extract mask and resize to match original image dimensions
        if sam_result and len(sam_result) > 0:
            masks = sam_result[0].masks
            if masks is not None and len(masks.data) > 0:
                mask_raw = masks.data[0].cpu().numpy().astype(np.uint8)
                h_img, w_img = image_bgr.shape[:2]
                h_mask, w_mask = mask_raw.shape[:2]
                if h_mask != h_img or w_mask != w_img:
                    mask_raw = cv2.resize(mask_raw, (w_img, h_img),
                                          interpolation=cv2.INTER_NEAREST)
                mask = mask_raw.astype(bool)
                results.append({
                    **det,
                    "mask_area": int(mask.sum()),
                    "mask_np": mask,
                })
                continue
        
        # Fallback: use bounding box as rough mask
        h, w = image_bgr.shape[:2]
        rough_mask = np.zeros((h, w), dtype=bool)
        rough_mask[y1:y2, x1:x2] = True
        results.append({
            **det,
            "mask": None,
            "mask_np": rough_mask,
            "mask_area": int(rough_mask.sum()),
            "mask_source": "bbox_fallback",
        })
    
    return results


def compute_object_depth(depth_map: np.ndarray, mask: np.ndarray) -> Dict:
    """Compute depth statistics for a masked object region."""
    depths = depth_map[mask]
    if len(depths) == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "count": 0}
    
    finite = np.isfinite(depths)
    if not finite.any():
        return {"mean": None, "median": None, "min": None, "max": None, "count": 0}
    
    valid = depths[finite]
    return {
        "mean": round(float(np.mean(valid)), 3),
        "median": round(float(np.median(valid)), 3),
        "min": round(float(np.min(valid)), 3),
        "max": round(float(np.max(valid)), 3),
        "std": round(float(np.std(valid)), 3),
        "count": int(len(valid)),
    }


def create_visualization(image_bgr: np.ndarray, depth_map: np.ndarray,
                         segmented: List[Dict], output_path: Path):
    """Create a side-by-side visualization: original, depth, and annotated."""
    h, w = image_bgr.shape[:2]
    
    # Depth visualization (normalize for display)
    depth_vis = depth_map.copy()
    finite_mask = np.isfinite(depth_vis)
    if finite_mask.any():
        vmin = np.percentile(depth_vis[finite_mask], 2)
        vmax = np.percentile(depth_vis[finite_mask], 98)
        depth_vis = np.clip((depth_vis - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    else:
        depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Annotated with masks
    annotated = image_bgr.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (128, 0, 255),
    ]
    depth_labels = []
    for i, obj in enumerate(segmented):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        depth_info = obj.get("depth", {})
        mean_d = depth_info.get("mean", "?")
        label = f"{obj['class']} d={mean_d}m" if mean_d and mean_d != "?" else obj["class"]
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        depth_labels.append(f"  {obj['class']}: mean={mean_d}m, median={depth_info.get('median', '?')}m")
    
    # Add depth legend
    y_offset = h - 20
    for label in reversed(depth_labels[:6]):  # Max 6 objects in legend
        cv2.putText(annotated, label, (5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y_offset -= 14
    
    # Stack horizontally
    combined = np.hstack([image_bgr, depth_vis, annotated])
    cv2.imwrite(str(output_path), combined)


def main():
    parser = argparse.ArgumentParser(description="Depth + Segmentation analysis")
    parser.add_argument("--detections", default="output/detections/detections.json",
                        help="Path to YOLO-World detections JSON")
    parser.add_argument("--frames", default="output/frames",
                        help="Directory containing frame images")
    parser.add_argument("--output", default="output/analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--device", default="cpu", help="Device: cpu, mps, cuda")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth estimation")
    parser.add_argument("--skip-seg", action="store_true", help="Skip segmentation")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit number of frames (0 = all)")
    args = parser.parse_args()
    
    frames_dir = Path(args.frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # ── Load detections ──────────────────────────────────────────
    detections_path = Path(args.detections)
    if not detections_path.exists():
        print(f"ERROR: Detections file not found: {detections_path}")
        print("  Run: python scripts/detect_objects.py first")
        sys.exit(1)
    
    with open(detections_path) as f:
        detections_data = json.load(f)
    
    # ── Load models ──────────────────────────────────────────────
    depth_model = None
    depth_transform = None
    sam_model = None
    
    if not args.skip_depth and HAS_DEPTH_PRO:
        print("Loading Depth Pro model...")
        depth_model, depth_transform = load_depth_model(args.device)
    
    if not args.skip_seg and HAS_SAM:
        print("Loading segmentation model...")
        sam_model = load_sam_model(args.device)
    
    # ── Process frames ──────────────────────────────────────────
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))
    
    if args.max_frames > 0:
        frame_files = frame_files[:args.max_frames]
    
    print(f"\nProcessing {len(frame_files)} frames...")
    
    analysis_results = {}
    
    for i, frame_path in enumerate(frame_files):
        frame_name = frame_path.name
        print(f"  [{i+1}/{len(frame_files)}] {frame_name}")
        
        # Load image
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            print(f"    WARNING: Could not read {frame_path}")
            continue
        
        # Get detections for this frame
        frame_dets = detections_data.get(frame_name, {}).get("detections", [])
        print(f"    {len(frame_dets)} detections")
        
        # Run depth estimation
        depth_map = None
        if depth_model is not None:
            try:
                depth_map = estimate_depth(depth_model, depth_transform, img_bgr)
                print(f"    Depth map: {depth_map.shape}, range [{depth_map.min():.2f}, {depth_map.max():.2f}]m")
            except Exception as e:
                print(f"    Depth estimation failed: {e}")
        
        # Run segmentation
        segmented = []
        if sam_model is not None and frame_dets:
            try:
                segmented = segment_objects(sam_model, img_bgr, frame_dets)
                print(f"    Segmented: {len(segmented)} objects")
            except Exception as e:
                print(f"    Segmentation failed: {e}")
                segmented = []
        
        # Compute per-object depth
        if depth_map is not None and segmented:
            for obj in segmented:
                mask = obj.get("mask_np")
                if mask is not None and mask.any():
                    obj["depth"] = compute_object_depth(depth_map, mask)
                    # Remove numpy mask (not JSON serializable)
                    if "mask_np" in obj:
                        del obj["mask_np"]
        
        # Save visualization
        if depth_map is not None:
            viz_path = viz_dir / f"analysis_{frame_name}"
            create_visualization(img_bgr, depth_map, segmented, viz_path)
        
        analysis_results[frame_name] = {
            "num_detections": len(frame_dets),
            "num_segmented": len(segmented),
            "objects": segmented,
        }
    
    # ── Save results ────────────────────────────────────────────
    # Remove non-serializable fields
    for frame_name, frame_data in analysis_results.items():
        for obj in frame_data.get("objects", []):
            obj.pop("mask_np", None)
    
    results_path = output_dir / "analysis.json"
    with open(results_path, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis saved to {results_path}")
    print(f"Visualizations saved to {viz_dir}/")
    
    # ── Summary ────────────────────────────────────────────────
    total_objects = 0
    all_classes = set()
    depth_stats = []
    for frame_data in analysis_results.values():
        for obj in frame_data.get("objects", []):
            total_objects += 1
            all_classes.add(obj["class"])
            if "depth" in obj and obj["depth"].get("mean") is not None:
                depth_stats.append(obj["depth"]["mean"])
    
    print(f"\n━━━ Scene Analysis Summary ━━━")
    print(f"  Frames analyzed: {len(analysis_results)}")
    print(f"  Total objects detected: {total_objects}")
    print(f"  Unique classes: {', '.join(sorted(all_classes))}")
    if depth_stats:
        print(f"  Depth range: {min(depth_stats):.2f}m – {max(depth_stats):.2f}m")
        print(f"  Mean object depth: {np.mean(depth_stats):.2f}m")


if __name__ == "__main__":
    main()
