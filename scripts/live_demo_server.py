#!/usr/bin/env python3
"""
Local web server for a live Habitat pipeline demo.

The server is deliberately dependency-free. It serves docs/live-demo.html and
starts the real pipeline scripts in a background thread when the browser posts
to /api/run. Generated artifacts are read from output/live_demo/<run_id>/ and
returned to the browser as they appear.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
RUNS_ROOT = ROOT / "output" / "live_demo"
DEFAULT_SCENE = ROOT / "data" / "hm3d" / "minival" / "00800-TEEsavR23oF" / "TEEsavR23oF.basis.glb"
DEFAULT_HABITAT_PYTHON = Path("/opt/homebrew/anaconda3/envs/habitat-sim/bin/python")

STATE_LOCK = threading.Lock()
STATE: Dict[str, Any] = {
    "running": False,
    "stage": "idle",
    "stageStatus": {"collect": "pending", "detect": "pending", "analyze": "pending"},
    "runId": None,
    "runDir": None,
    "startedAt": None,
    "finishedAt": None,
    "error": None,
    "log": [],
    "commands": [],
}
CURRENT_PROCESS: Optional[subprocess.Popen] = None
STOP_REQUESTED = False


def json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200, content_type: str = "text/plain") -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "%s; charset=utf-8" % content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def append_log(line: str) -> None:
    with STATE_LOCK:
        log = STATE.setdefault("log", [])
        log.append(line.rstrip())
        del log[:-300]


def update_state(**kwargs: Any) -> None:
    with STATE_LOCK:
        STATE.update(kwargs)


def update_stage(name: str, status: str) -> None:
    with STATE_LOCK:
        stage_status = dict(STATE.get("stageStatus", {}))
        stage_status[name] = status
        STATE["stageStatus"] = stage_status
        STATE["stage"] = name


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def cache_busted_url(run_id: str, rel_path: Path) -> str:
    abs_path = RUNS_ROOT / run_id / rel_path
    try:
        version = abs_path.stat().st_mtime_ns
    except FileNotFoundError:
        version = int(time.time() * 1000)
    return "/runs/%s/%s?v=%s" % (run_id, rel_path.as_posix(), version)


def frame_index(frame_name: str) -> int:
    try:
        return int(Path(frame_name).stem.split("_")[-1])
    except ValueError:
        return 0


def discover_frames(run_id: str, run_dir: Path) -> List[Dict[str, Any]]:
    frames_dir = run_dir / "frames"
    detections_dir = run_dir / "detections"
    analysis_dir = run_dir / "analysis"

    metadata = read_json(frames_dir / "metadata.json")
    detections = read_json(detections_dir / "detections.json")
    analysis = read_json(analysis_dir / "analysis.json")
    frame_meta = {
        item.get("file"): item
        for item in metadata.get("frames", [])
        if item.get("file")
    }

    names = set()
    for path in frames_dir.glob("frame_*.jpg"):
        names.add(path.name)
    for path in (detections_dir / "annotated").glob("frame_*.jpg"):
        names.add(path.name)
    for path in (analysis_dir / "visualizations").glob("analysis_frame_*.jpg"):
        names.add(path.name.replace("analysis_", ""))

    frames = []
    for name in sorted(names, key=frame_index):
        rgb_rel = Path("frames") / name
        det_rel = Path("detections") / "annotated" / name
        ana_rel = Path("analysis") / "visualizations" / ("analysis_%s" % name)
        depth_rel = Path("analysis") / "visualizations" / ("depth_%s" % name)
        spatial_rel = Path("analysis") / "visualizations" / ("spatial_%s" % name)

        item = {
            "index": frame_index(name),
            "name": name,
            "rgb": cache_busted_url(run_id, rgb_rel) if (run_dir / rgb_rel).exists() else None,
            "detection": cache_busted_url(run_id, det_rel) if (run_dir / det_rel).exists() else None,
            "analysis": cache_busted_url(run_id, ana_rel) if (run_dir / ana_rel).exists() else None,
            "depth": cache_busted_url(run_id, depth_rel) if (run_dir / depth_rel).exists() else None,
            "spatial": cache_busted_url(run_id, spatial_rel) if (run_dir / spatial_rel).exists() else None,
            "position": frame_meta.get(name, {}).get("position"),
            "rotation": frame_meta.get(name, {}).get("rotation"),
            "numDetections": detections.get(name, {}).get("num_detections", 0),
            "detections": detections.get(name, {}).get("detections", []),
            "objects": analysis.get(name, {}).get("objects", []),
        }
        frames.append(item)
    return frames


def summarize(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    classes = set()
    total_detections = 0
    depth_values = []
    positions = []

    for frame in frames:
        total_detections += int(frame.get("numDetections") or 0)
        for det in frame.get("detections", []):
            if det.get("class"):
                classes.add(det["class"])
        for obj in frame.get("objects", []):
            if obj.get("class"):
                classes.add(obj["class"])
            mean = obj.get("depth", {}).get("mean")
            if mean is not None:
                depth_values.append(float(mean))
        pos = frame.get("position")
        if pos and len(pos) >= 3:
            positions.append([float(pos[0]), float(pos[2])])

    return {
        "frameCount": len([f for f in frames if f.get("rgb")]),
        "detectionFrames": len([f for f in frames if f.get("detection")]),
        "analysisFrames": len([f for f in frames if f.get("analysis") or f.get("depth")]),
        "totalDetections": total_detections,
        "uniqueClasses": len(classes),
        "classes": sorted(classes),
        "depthMin": min(depth_values) if depth_values else None,
        "depthMax": max(depth_values) if depth_values else None,
        "positions": positions,
    }


def public_status() -> Dict[str, Any]:
    with STATE_LOCK:
        status = dict(STATE)
        status["stageStatus"] = dict(STATE.get("stageStatus", {}))
        status["log"] = list(STATE.get("log", []))
        status["commands"] = list(STATE.get("commands", []))

    run_id = status.get("runId")
    run_dir_value = status.get("runDir")
    frames: List[Dict[str, Any]] = []
    if run_id and run_dir_value:
        run_dir = Path(run_dir_value)
        frames = discover_frames(run_id, run_dir)

    status["frames"] = frames
    status["summary"] = summarize(frames)
    status["defaultScene"] = str(DEFAULT_SCENE.relative_to(ROOT)) if DEFAULT_SCENE.exists() else str(DEFAULT_SCENE)
    return status


def parse_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def normalized_options(payload: Dict[str, Any]) -> Dict[str, Any]:
    scene = str(payload.get("scene") or os.environ.get("HABITAT_SCENE") or DEFAULT_SCENE)
    scene_path = Path(scene)
    if not scene_path.is_absolute():
        scene_path = ROOT / scene_path

    frames = int(payload.get("frames") or 10)
    analysis_frames = int(payload.get("analysisFrames") or min(frames, 4))
    confidence = float(payload.get("confidence") or 0.15)
    device = str(payload.get("device") or "mps")
    resolution = str(payload.get("resolution") or "960x540").lower()
    if "x" in resolution:
        width_text, height_text = resolution.split("x", 1)
        width = int(width_text)
        height_px = int(height_text)
    else:
        width = int(payload.get("width") or 960)
        height_px = int(payload.get("heightPx") or 540)

    frames = max(1, min(frames, 80))
    analysis_frames = max(0, min(analysis_frames, frames))
    confidence = max(0.01, min(confidence, 0.95))
    width = max(320, min(width, 1920))
    height_px = max(240, min(height_px, 1080))
    if device not in {"cpu", "mps", "cuda"}:
        device = "cpu"

    return {
        "scene": scene_path,
        "frames": frames,
        "analysisFrames": analysis_frames,
        "confidence": confidence,
        "device": device,
        "width": width,
        "heightPx": height_px,
    }


def command_text(args: List[str]) -> str:
    return " ".join(args)


def clean_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    # Avoid leaking the uv virtualenv or local Python path into the Habitat conda
    # interpreter. This was the cause of confusing "habitat-sim not found" runs
    # when launched from the web process.
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    return env


def habitat_python_command() -> List[str]:
    configured = os.environ.get("HABITAT_PYTHON")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(DEFAULT_HABITAT_PYTHON)

    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    return ["conda", "run", "-n", "habitat-sim", "python"]


def run_command(stage_key: str, title: str, args: List[str]) -> None:
    global CURRENT_PROCESS, STOP_REQUESTED

    update_stage(stage_key, "running")
    append_log("")
    append_log("==> %s" % title)
    append_log("$ %s" % command_text(args))
    with STATE_LOCK:
        STATE.setdefault("commands", []).append({"stage": stage_key, "command": args})

    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=clean_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    CURRENT_PROCESS = proc

    assert proc.stdout is not None
    for line in proc.stdout:
        append_log(line)

    return_code = proc.wait()
    CURRENT_PROCESS = None

    if STOP_REQUESTED:
        raise RuntimeError("Run stopped")
    if return_code != 0:
        raise RuntimeError("%s failed with exit code %s" % (title, return_code))

    update_stage(stage_key, "done")


def pipeline_thread(options: Dict[str, Any], run_id: str, run_dir: Path) -> None:
    global STOP_REQUESTED

    try:
        update_state(
            running=True,
            stage="starting",
            stageStatus={"collect": "pending", "detect": "pending", "analyze": "pending"},
            runId=run_id,
            runDir=str(run_dir),
            startedAt=datetime.now().isoformat(timespec="seconds"),
            finishedAt=None,
            error=None,
            log=[],
            commands=[],
        )
        append_log("Live run: %s" % run_id)
        append_log("Scene: %s" % options["scene"])
        append_log("Frames: %s | analysis frames: %s | confidence: %.2f | device: %s | resolution: %sx%s" % (
            options["frames"],
            options["analysisFrames"],
            options["confidence"],
            options["device"],
            options["width"],
            options["heightPx"],
        ))

        (run_dir / "frames").mkdir(parents=True, exist_ok=True)
        (run_dir / "detections").mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis").mkdir(parents=True, exist_ok=True)

        collect = habitat_python_command() + [
            "scripts/collect_trajectory.py",
            "--scene", str(options["scene"]),
            "--output", str(run_dir / "frames"),
            "--frames", str(options["frames"]),
            "--height", "1.5",
            "--step", "0.5",
            "--width", str(options["width"]),
            "--height_px", str(options["heightPx"]),
        ]
        detect = [
            "uv", "run", "python", "scripts/detect_objects.py",
            "--input", str(run_dir / "frames"),
            "--output", str(run_dir / "detections"),
            "--model", "yolov8s-worldv2.pt",
            "--confidence", str(options["confidence"]),
            "--device", options["device"],
        ]
        analyze = [
            "uv", "run", "python", "scripts/analyze_scene.py",
            "--frames", str(run_dir / "frames"),
            "--detections", str(run_dir / "detections" / "detections.json"),
            "--output", str(run_dir / "analysis"),
            "--device", options["device"],
            "--max-frames", str(options["analysisFrames"]),
        ]

        run_command("collect", "Habitat-Sim trajectory collection", collect)
        run_command("detect", "YOLO-World open-vocabulary detection", detect)
        run_command("analyze", "Depth Pro + FastSAM spatial analysis", analyze)

        update_state(running=False, stage="complete", finishedAt=datetime.now().isoformat(timespec="seconds"))
        append_log("==> Run complete")
    except Exception as exc:
        with STATE_LOCK:
            stage = STATE.get("stage")
            stage_status = dict(STATE.get("stageStatus", {}))
            if stage in stage_status and stage_status.get(stage) == "running":
                stage_status[stage] = "error"
            STATE["stageStatus"] = stage_status
            STATE["running"] = False
            STATE["error"] = str(exc)
            STATE["finishedAt"] = datetime.now().isoformat(timespec="seconds")
        append_log("ERROR: %s" % exc)
    finally:
        STOP_REQUESTED = False


def start_run(payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], int]:
    with STATE_LOCK:
        if STATE.get("running"):
            return False, {"error": "A pipeline run is already active"}, HTTPStatus.CONFLICT

    options = normalized_options(payload)
    if not options["scene"].exists():
        return False, {
            "error": "Scene file not found",
            "scene": str(options["scene"]),
            "hint": "Download HM3D minival or pass a valid scene path.",
        }, HTTPStatus.BAD_REQUEST

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = "live_%s" % datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / run_id

    thread = threading.Thread(target=pipeline_thread, args=(options, run_id, run_dir), daemon=True)
    thread.start()
    return True, {"ok": True, "runId": run_id}, HTTPStatus.ACCEPTED


def stop_run() -> Dict[str, Any]:
    global STOP_REQUESTED, CURRENT_PROCESS
    STOP_REQUESTED = True
    proc = CURRENT_PROCESS
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        time.sleep(0.5)
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    update_state(running=False, stage="stopped", error="Run stopped by user", finishedAt=datetime.now().isoformat(timespec="seconds"))
    append_log("==> Stop requested")
    return {"ok": True}


def guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return "text/html"
    if suffix == ".css":
        return "text/css"
    if suffix == ".js":
        return "application/javascript"
    if suffix == ".json":
        return "application/json"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def serve_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    if not resolved.exists() or not resolved.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return

    body = resolved.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", guess_content_type(resolved))
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "HabitatLiveDemo/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route in {"/", "/live-demo.html"}:
            serve_file(self, DOCS_ROOT / "live-demo.html")
            return
        if route == "/api/status":
            json_response(self, public_status())
            return
        if route.startswith("/runs/"):
            parts = route.split("/", 3)
            if len(parts) < 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            run_id = unquote(parts[2])
            rel = unquote(parts[3])
            candidate = (RUNS_ROOT / run_id / rel).resolve()
            allowed_root = (RUNS_ROOT / run_id).resolve()
            if allowed_root not in candidate.parents and candidate != allowed_root:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            serve_file(self, candidate)
            return
        if route.startswith("/assets/"):
            candidate = (DOCS_ROOT / route.lstrip("/")).resolve()
            if DOCS_ROOT.resolve() not in candidate.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            serve_file(self, candidate)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            try:
                payload = parse_json_body(self)
            except json.JSONDecodeError:
                json_response(self, {"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
                return
            _, response, status = start_run(payload)
            json_response(self, response, status)
            return
        if parsed.path == "/api/stop":
            json_response(self, stop_run())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live Habitat web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = "http://%s:%s" % (args.host, args.port)
    print("Live Habitat demo server")
    print("  %s" % url)
    print("  Press Ctrl-C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_run()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
