# Interview Walkthrough

This is the presentation path I would use for the Track 2 spatial reasoning interview.

## Live Demo

Start the local web runner:

```bash
uv run python scripts/live_demo_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

The page starts the real pipeline from the browser. It writes each run to
`output/live_demo/<run_id>/` and updates the UI as artifacts appear:

1. `Habitat-Sim` renders egocentric RGB frames from the HM3D scene.
2. `YOLO-World` detects open-vocabulary indoor objects.
3. `Depth Pro + FastSAM` produces metric depth, masks, per-object depth statistics, and side-by-side visualizations.

For a live interview, use a short run first:

```text
Frames: 8-10
Depth frames: 3-4
Resolution: 960x540 or 1280x720
Confidence: 0.15
ML device: mps on Apple Silicon
```

This still executes the actual pipeline on real HM3D data, but it avoids spending the whole demo on Depth Pro inference. If time allows, mention that the full verified run used 26 frames and 340 detections.

Habitat-Sim renders through the native simulator environment and reports the Apple Metal-backed renderer on this machine. YOLO-World, Depth Pro, and FastSAM receive `--device mps` from the live demo.

## What To Say First

The assignment asked for first-person data from the fixed HM3D scene and a basic perception stack on top. I implemented a three-stage pipeline: simulate a human-like camera trajectory in Habitat-Sim, run open-vocabulary object detection with YOLO-World, then combine Depth Pro and FastSAM to compute per-object spatial depth statistics.

The important design choice is that each stage writes inspectable files: JPEG frames and JSON outputs. That makes the pipeline easy to debug, rerun stage-by-stage, and extend with a different detector or depth model later.

## Decision Map

| Decision | Why it was chosen | Code |
|---|---|---|
| Fixed HM3D scene `00800-TEEsavR23oF` | Matches the interview brief and gives realistic furnished interiors. | `scripts/collect_trajectory.py` scene discovery and `README.md` setup |
| 640 x 480 / 960 x 540 RGB frames | `640x480` is fast for debugging; the live demo supports `960x540` and `1280x720` for clearer presentation. | `scripts/collect_trajectory.py:69`, `docs/live-demo.html` |
| 1.5 m camera height | Approximate human eye height for first-person walking data. | `scripts/collect_trajectory.py:70` |
| 90 degree horizontal FOV | Wide enough to capture room context without becoming a fisheye view. | `scripts/collect_trajectory.py:69` |
| Navmesh-constrained waypoints | Prevents camera paths through walls and keeps movement human-like. | `scripts/collect_trajectory.py:102`, `scripts/collect_trajectory.py:126` |
| Open-vocabulary indoor prompts | Lets the detector look for room-scale objects without retraining. | `scripts/detect_objects.py:35` |
| Confidence near `0.15` for demo results | Trades some false positives for better recall in cluttered interiors. | `scripts/detect_objects.py:64` |
| FastSAM instead of NanoSAM | NanoSAM is Jetson/TensorRT-oriented; FastSAM gives the same bbox-to-mask role locally through ultralytics. | `scripts/analyze_scene.py:75` |
| Depth Pro | Produces metric monocular depth from RGB, which is useful for spatial reasoning without stereo or LiDAR. | `scripts/analyze_scene.py:91` |
| Per-object depth stats | Turns model outputs into interpretable spatial measurements. | `scripts/analyze_scene.py:169` |
| Separate conda and uv environments | Habitat-Sim native dependencies conflict with the newer ML stack. | `scripts/setup.sh`, `run_pipeline.sh` |

## Failure Cases To Acknowledge

YOLO-World sometimes confuses semantically close categories such as couch and sofa, or detects structural surfaces like wall and ceiling as objects because they are in the vocabulary. Thin objects and far-away details are less reliable.

FastSAM masks depend on YOLO boxes. If the box is bad, the mask and depth statistics inherit that error. Mask resizing can also introduce boundary artifacts.

Depth Pro is monocular, so it can be wrong at reflective surfaces, object boundaries, and ambiguous room geometry. It is also the slowest stage on CPU.

The current pipeline does not track object identity over time. A chair detected in two frames is counted as two observations, not one persistent object.

## Extension Intent

I would extend this in four steps.

First, add temporal association: match detections across frames using object class, box overlap, estimated depth, and camera pose. That would turn frame-level detections into persistent scene objects.

Second, build a simple spatial memory: fuse per-object depth estimates into a top-down map or object graph. The output would answer questions like "what objects are near the couch" or "which doorway leads toward the kitchen."

Third, improve model evaluation and substitution: benchmark YOLO-World confidence thresholds, replace FastSAM with NanoSAM if the target hardware supports it, and measure how much each specialist model contributes to the final spatial statistics.

Fourth, explore JEPA-style representation learning as a research extension. The current pipeline is a specialist baseline that produces explicit artifacts: boxes, labels, masks, metric depth, and JSON measurements. A JEPA model would not replace those outputs directly; its native output is a latent embedding field. The useful question is whether that latent field contains enough spatial structure to recover depth ordering, objectness, or trajectory dynamics through small trained heads.

The honest framing is that AI helped accelerate implementation, but the artifacts are inspectable: the scripts, JSON outputs, visualizations, and live runner show what each model contributes and where the weak assumptions are.

## JEPA Research Intent

The JEPA angle I want to communicate is: "I built the assignment with reliable specialist models first, then used that verified pipeline to define a clean representation-learning experiment."

The baseline is deliberately explicit. YOLO-World returns boxes and class labels, Depth Pro returns metric depth, and FastSAM returns object masks. That gives me interpretable pseudo-labels and evaluation targets. JEPA-family models, by contrast, learn by predicting hidden representations in embedding space. They may encode semantic layout and spatial context, but they do not natively output object labels, bounding boxes, masks, or metric depth.

So I would not claim "JEPA replaces the pipeline." I would say: "The baseline answers the interview task. JEPA is my proposed extension for testing whether a self-supervised representation can learn useful spatial structure from the same egocentric data."

### Proposed JEPA Experiment

The first experiment would use the verified pipeline as a teacher:

1. Collect HM3D trajectory frames and camera poses with Habitat-Sim.
2. Run the existing Depth Pro stage to produce metric depth maps.
3. Freeze an I-JEPA image encoder, such as `facebook/ijepa_vith16_1k`.
4. Train only a small decoder head from frozen JEPA features to the Depth Pro depth map.
5. Evaluate with pixel metrics and task metrics: absolute relative error, RMSE, and whether per-object depth ordering is preserved.

This is useful because the evaluation is grounded in the same artifacts already produced by the repo. I can compare the JEPA-derived depth estimate against the Depth Pro teacher and against the per-object depth statistics the pipeline computes.

If the depth probe works, the next experiment is an objectness probe: use YOLO-World boxes as supervision and train a small head to predict object regions from frozen JEPA features. I would not expect it to recover open-vocabulary labels without a language-conditioned head, but it might learn where objects tend to be in indoor scenes.

The third experiment is temporal: use V-JEPA or JEPA-WM style models to test whether embedding distance follows camera motion. In Habitat, I have known pose changes between frames, so I can ask whether representation changes correlate with forward motion, turns, and room transitions.

### How To Say This In The Interview

"I see the current system as a strong baseline rather than the final research idea. The implemented stack gives me reliable, inspectable spatial labels: boxes, masks, metric depth, and per-object measurements. My JEPA-style extension would ask whether a self-supervised latent representation can recover those same spatial quantities with small readout heads. If it works, it could reduce reliance on separate specialist models or improve temporal consistency. If it does not work, that is still a useful result: JEPA remains an auxiliary representation, not a replacement for the specialist perception stack."

### What Not To Overclaim

JEPA does not currently give object labels, masks, boxes, or metric depth out of the box.

The public I-JEPA checkpoint is a static image representation model, so it is not automatically a spatial-reasoning engine.

Depth Pro pseudo-labels are not ground truth. They are a practical teacher signal for a controlled experiment, but proper evaluation would ideally include Habitat ground-truth depth or multi-view consistency checks.

The extension should be framed as a testable research direction, not as a completed feature.
