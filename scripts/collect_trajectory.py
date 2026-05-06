#!/usr/bin/env python3
"""
Collect egocentric RGB frames from an AI Habitat scene, simulating
a human walking trajectory.

Usage:
    python scripts/collect_trajectory.py [--scene SCENE_PATH] [--output OUTPUT_DIR]
                                         [--frames N] [--height H] [--step S]

Scene priority:
    1. --scene CLI argument
    2. HABITAT_SCENE env var
    3. HM3D minival 00800-TEEsavR23oF (if downloaded)
    4. habitat-sim test scene (simple_room)

Output: A sequence of RGB frames saved as frame_0000.jpg, frame_0001.jpg, ...
        plus a metadata.json with camera parameters and agent positions.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

try:
    import habitat_sim
    from habitat_sim import nav
except ImportError:
    print("ERROR: habitat-sim not found. Install via:")
    print("  conda install -c aihabitat -c conda-forge habitat-sim")
    sys.exit(1)


def find_test_scene() -> Optional[str]:
    """Find a usable test scene from habitat-sim's test assets."""
    import habitat_sim as hs

    hs_path = Path(hs.__file__).parent
    # Check for cloned habitat-sim repo test assets
    candidates = [
        Path("/tmp/habitat-sim/data/test_assets/scenes/simple_room.glb"),
        hs_path.parent.parent / "data" / "test_assets" / "scenes" / "simple_room.glb",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def find_hm3d_scene() -> Optional[str]:
    """Find the HM3D minival scene 00800-TEEsavR23oF if downloaded."""
    hm3d_paths = [
        Path("data/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"),
        Path.home() / "data/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
        Path("/tmp/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"),
    ]
    for p in hm3d_paths:
        if p.exists():
            return str(p)
    return None


def make_sim_config(scene_path: str, width: int = 640, height: int = 480,
                    hfov: float = 90, sensor_height: float = 1.5) -> habitat_sim.Configuration:
    """Create a simulator configuration with a color sensor at human eye height."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False

    # RGB sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [height, width]
    rgb_sensor_spec.position = [0.0, sensor_height, 0.0]
    rgb_sensor_spec.hfov = hfov
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [height, width]
    depth_sensor_spec.position = [0.0, sensor_height, 0.0]
    depth_sensor_spec.hfov = hfov
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
    agent_cfg.height = sensor_height
    agent_cfg.radius = 0.1

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def generate_waypoints(sim, num_waypoints: int = 30, step_dist: float = 0.5) -> List[np.ndarray]:
    """Generate waypoints forming a walking path through the scene.

    Strategy:
    1. Try to use the navmesh for valid navigable points
    2. Fall back to a structured grid or spiral pattern around the origin

    Returns list of 3D positions (x, y, z).
    """
    pathfinder = sim.pathfinder
    if pathfinder.is_loaded:
        return _navmesh_waypoints(sim, num_waypoints, step_dist)
    else:
        print("  No navmesh loaded. Using heuristic waypoint generation.")
        return _heuristic_waypoints(sim, num_waypoints, step_dist)


def _to_np(v) -> np.ndarray:
    """Convert Magnum Vector3 or similar to numpy array."""
    if hasattr(v, '__iter__'):
        return np.array([float(v[0]), float(v[1]), float(v[2])])
    return np.array(v, dtype=float)


def _navmesh_waypoints(sim, num_waypoints: int, step_dist: float) -> List[np.ndarray]:
    """Generate waypoints by walking across the navmesh."""
    pathfinder = sim.pathfinder

    # Find a valid starting point
    bounds_raw = pathfinder.get_bounds()
    bounds = (_to_np(bounds_raw[0]), _to_np(bounds_raw[1]))
    center = (bounds[0] + bounds[1]) / 2

    # Start from a random navigable point
    start = _to_np(pathfinder.get_random_navigable_point())
    if np.all(start == 0):
        # Fallback: try a few positions
        for attempt in range(20):
            candidate = center + np.array([
                (np.random.random() - 0.5) * 5,
                0,
                (np.random.random() - 0.5) * 5,
            ])
            snapped = _to_np(pathfinder.snap_point(candidate))
            if np.any(snapped != 0) and pathfinder.is_navigable(snapped):
                start = snapped
                break
        else:
            return _heuristic_waypoints(sim, num_waypoints, step_dist)

    waypoints = [start]
    current = start.copy()
    directions = [
        np.array([1, 0, 0]),
        np.array([0, 0, 1]),
        np.array([-1, 0, 0]),
        np.array([0, 0, -1]),
        np.array([0.7, 0, 0.7]),
        np.array([-0.7, 0, 0.7]),
        np.array([0.7, 0, -0.7]),
        np.array([-0.7, 0, -0.7]),
    ]

    while len(waypoints) < num_waypoints:
        for d in directions:
            if len(waypoints) >= num_waypoints:
                break
            candidate = current + d * step_dist * (1 + np.random.random() * 0.5)
            if pathfinder.is_navigable(candidate):
                current = candidate
                waypoints.append(current.copy())

        # If no progress, try random perturbation
        current = current + np.array([
            (np.random.random() - 0.5) * 2,
            0,
            (np.random.random() - 0.5) * 2,
        ])
        if pathfinder.is_navigable(current):
            waypoints.append(current.copy())

    return waypoints


def _heuristic_waypoints(sim, num_waypoints: int, step_dist: float) -> List[np.ndarray]:
    """Generate waypoints without navmesh — spiral pattern from origin."""
    waypoints = [np.array([0.0, 0.0, 0.0])]
    angle_step = 2 * math.pi / 8
    radius = 1.0

    for i in range(1, num_waypoints):
        angle = i * angle_step * 0.7
        radius = 1.0 + i * 0.3
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        waypoints.append(np.array([x, 0.0, z]))

    return waypoints


def compute_look_at(from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
    """Compute a rotation quaternion that looks from from_pos toward to_pos.
    The agent faces forward in its local +z or -z depending on config.
    habitat-sim default: agent faces -z, up is +y.
    """
    direction = to_pos - from_pos
    if np.linalg.norm(direction) < 1e-6:
        return np.array([0, 0, 0, 1])  # identity quaternion

    # Normalize direction on xz plane
    direction[1] = 0
    direction = direction / np.linalg.norm(direction)

    # In habitat-sim, agent by default faces -Z axis
    # Compute yaw from forward vector
    yaw = math.atan2(direction[0], direction[2])

    # Convert yaw to quaternion (rotation around Y axis)
    half = yaw / 2
    q = np.array([0.0, math.sin(half), 0.0, math.cos(half)])
    # Note: habitat_sim uses quaternion format [x, y, z, w] for scene API
    return q


def capture_trajectory(sim, waypoints: List[np.ndarray],
                       output_dir: Path, look_ahead: bool = True) -> List[dict]:
    """Move the agent along waypoints and capture RGB frames."""
    agent = sim.initialize_agent(0)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for i, target_pos in enumerate(waypoints):
        # Set agent position
        agent_state = habitat_sim.AgentState()
        agent_state.position = target_pos

        # Compute rotation: look toward next waypoint
        if look_ahead and i < len(waypoints) - 1:
            agent_state.rotation = compute_look_at(target_pos, waypoints[i + 1])
        elif i > 0:
            agent_state.rotation = compute_look_at(waypoints[i - 1], target_pos)
        else:
            agent_state.rotation = np.array([0, 0, 0, 1])

        agent.set_state(agent_state)

        # Render
        observations = sim.get_sensor_observations()

        rgb = observations["color_sensor"]  # (H, W, 4) RGBA
        depth = observations.get("depth_sensor")  # (H, W, 1) float32

        # Save RGB as JPEG
        frame_path = output_dir / f"frame_{i:04d}.jpg"
        import cv2
        rgb_bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), rgb_bgr)

        # Convert rotation to list — handle Magnum quaternion
        rot = agent_state.rotation
        try:
            # Magnum Quaternion has .vector (Vector3) and .scalar (float)
            rot_list = [float(rot.vector.x), float(rot.vector.y), float(rot.vector.z), float(rot.scalar)]
        except Exception:
            rot_list = None

        # Convert position similarly
        try:
            pos_list = [float(target_pos[i]) for i in range(3)]
        except Exception:
            pos_list = [0.0, 0.0, 0.0]

        frames.append({
            "index": i,
            "file": str(frame_path.name),
            "position": pos_list,
            "rotation": rot_list,
        })

    return frames


def main():
    parser = argparse.ArgumentParser(description="Collect egocentric trajectory frames")
    parser.add_argument("--scene", help="Path to scene GLB file")
    parser.add_argument("--output", default="output/frames", help="Output directory for frames")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames to capture")
    parser.add_argument("--height", type=float, default=1.5, help="Sensor height (meters)")
    parser.add_argument("--step", type=float, default=0.5, help="Step distance between waypoints")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height_px", type=int, default=480)
    args = parser.parse_args()

    # Resolve scene path
    scene_path = args.scene or os.environ.get("HABITAT_SCENE")
    if not scene_path:
        scene_path = find_hm3d_scene()
        if scene_path:
            print(f"Using HM3D scene: {scene_path}")
    if not scene_path:
        scene_path = find_test_scene()
        if scene_path:
            print(f"Using test scene: {scene_path}")
    if not scene_path:
        print("ERROR: No scene found. Provide --scene, set HABITAT_SCENE, or download HM3D data.")
        print("To download HM3D minival:")
        print("  1. Register at https://matterport.com/habitat-matterport-3d-research-dataset")
        print("  2. Get API token from https://my.matterport.com/settings/account/devtools")
        print("  3. Run: python -m habitat_sim.utils.datasets_download \\")
        print("       --username <token-id> --password <token-secret> \\")
        print("       --uids hm3d_minival_v0.2 --data-path data/")
        sys.exit(1)

    output_dir = Path(args.output)

    print(f"Scene: {scene_path}")
    print(f"Output: {output_dir.resolve()}")

    # Configure simulator
    cfg = make_sim_config(scene_path, width=args.width, height=args.height_px,
                          sensor_height=args.height)

    print("Creating simulator...")
    sim = habitat_sim.Simulator(cfg)

    try:
        # Generate waypoints
        print(f"Generating {args.frames} waypoints...")
        waypoints = generate_waypoints(sim, num_waypoints=args.frames,
                                       step_dist=args.step)
        print(f"  Generated {len(waypoints)} waypoints")

        # Capture trajectory
        print("Capturing frames...")
        frame_meta = capture_trajectory(sim, waypoints, output_dir)

        # Save metadata
        metadata = {
            "scene": scene_path,
            "num_frames": len(frame_meta),
            "resolution": [args.width, args.height_px],
            "sensor_height": args.height,
            "frames": frame_meta,
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {meta_path}")

        print(f"\nDone! Collected {len(frame_meta)} frames in {output_dir}")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
