#!/usr/bin/env python3
"""
Run YOLO-World object detection on collected trajectory frames.

YOLO-World is an open-vocabulary detector that can detect objects
described by arbitrary text prompts. This script:

1. Loads a YOLO-World model (yolov8s-worldv2.pt by default)
2. Detects objects in each frame using a configurable vocabulary
3. Saves detection results (bboxes, classes, scores) as JSON
4. Saves annotated frames with bounding boxes drawn

Usage:
    python scripts/detect_objects.py --input output/frames --output output/detections
                                     [--vocabulary "chair,table,door,window,person,..."]
                                     [--model yolov8s-worldv2.pt]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

# Default open-vocabulary categories for indoor scene understanding
DEFAULT_VOCABULARY = [
    "chair", "table", "desk", "sofa", "couch", "bed", "cabinet", "shelf",
    "door", "window", "wall", "floor", "ceiling",
    "television", "tv", "monitor", "laptop", "computer",
    "lamp", "light", "refrigerator", "oven", "sink",
    "plant", "book", "bottle", "cup", "bowl",
    "backpack", "bag", "person", "picture", "painting",
    "stairs", "railing", "curtain", "pillow", "rug",
    "vase", "clock", "mirror"
]

def load_model(model_path: str, vocabulary: List[str], confidence: float = 0.25) -> YOLO:
    """Load YOLO-World model with custom vocabulary."""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Set custom classes for open-vocabulary detection
    model.set_classes(vocabulary)

    if confidence != 0.25:
        # Override confidence threshold if specifi ed
        pass

    print(f"  Vocabulary: {len(vocabulary)} classes")
    print(f"  Confidence threshold: {confidence}")
    return model


def detect_objects(
    model: YOLO,
    frame_path: Path,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 50,
) -> List[Dict]:
    """Run detection on a single frame and return structured results."""
    results = model.predict(
        source=str(frame_path),
        conf=confidence,
        iou=iou_threshold,
        max_det=max_detections,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] if cls_id < len(model.names) else f"class_{cls_id}"

            detections.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
            })

    return detections


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on an image."""
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (128, 0, 255),
        (255, 128, 0), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    ]
    img = image.copy()
    for det in detections:
        color = colors[det["class_id"] % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 5, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="YOLO-World object detection on frames")
    parser.add_argument("--input", default="output/frames", help="Input directory with frames")
    parser.add_argument("--output", default="output/detections", help="Output directory")
    parser.add_argument("--model", default="yolov8s-worldv2.pt", help="YOLO-World model path")
    parser.add_argument("--vocabulary", nargs="+", help="Custom vocabulary (space-separated)")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    vocab = args.vocabulary if args.vocabulary else DEFAULT_VOCABULARY

    # Load model
    model = load_model(args.model, vocab, args.confidence)

    # Scan for frame files
    frame_files = sorted(input_dir.glob("frame_*.jpg"))
    if not frame_files:
        frame_files = sorted(input_dir.glob("*.jpg"))
    if not frame_files:
        print(f"ERROR: No frame files found in {input_dir}")
        sys.exit(1)

    print(f"\nProcessing {len(frame_files)} frames...")

    all_results = {}
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    for i, frame_path in enumerate(frame_files):
        print(f"  [{i+1}/{len(frame_files)}] {frame_path.name}")

        detections = detect_objects(model, frame_path, args.confidence, args.iou)

        # Draw and save annotated frame
        img = cv2.imread(str(frame_path))
        if img is not None and detections:
            annotated = draw_detections(np.array(img), detections)
            cv2.imwrite(str(annotated_dir / frame_path.name), annotated)

        all_results[frame_path.name] = {
            "num_detections": len(detections),
            "detections": detections,
        }

        if detections:
            classes_found = set(d["class"] for d in detections)
            print(f"      Found {len(detections)} objects: {', '.join(sorted(classes_found))}")
        else:
            print(f"      No objects detected")

    # Save results
    results_path = output_dir / "detections.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Annotated frames saved to {annotated_dir}/")

    # Summary statistics
    total_objects = sum(v["num_detections"] for v in all_results.values())
    all_classes = set()
    for v in all_results.values():
        for d in v["detections"]:
            all_classes.add(d["class"])
    print(f"\nSummary:")
    print(f"  Frames processed: {len(frame_files)}")
    print(f"  Total objects detected: {total_objects}")
    print(f"  Unique classes found: {len(all_classes)}")
    print(f"  Classes: {', '.join(sorted(all_classes))}")


if __name__ == "__main__":
    main()
