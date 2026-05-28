# CARLA Person Follower

Autonomous person-following demo in CARLA that integrates **Hybrid_reID** (YOLOv8 + DeepSORT + DINOv2) for person tracking and **OmniVLA-edge** (EfficientNet-B0 + Transformer VLA) for navigation — using only the onboard RGB camera, no CARLA oracle data.

---

## System Overview

```
CARLA RGB camera (640×480, 90° FOV, 3 Hz sync)
       │
       ▼
Thread 1 — Perception
  YOLODetector (YOLOv8n-seg, person-only)
       │
  HybridTracker (DeepSORT + DINOv2 re-ID)
       │
  SharedState ──────────────────────────────┐
                                            │
Thread 2 — Navigation                       │
  OmniVLA-edge (EfficientNet-B0 + Transformer)
  Input: 6-frame context (96×96) + bbox goal pose
  Output: 8-step waypoints → (vx, wz)       │
       │                                    │
       └──────────────── SharedState ───────┘
                                │
Main loop — Control (world.tick)
  vel_to_control → carla.VehicleControl
  WalkerControl  → pedestrian motion
  cv2.imshow     → live camera window
```

---

## Files

| File | Description |
|------|-------------|
| `carla_follower.py` | Main integration script — all logic in one file |
| `README.md` | This file |

**Dependencies (not in this dir — paths are hardcoded):**

| Path | Description |
|------|-------------|
| `~/OmniVLA/omnivla-edge/omnivla-edge.pth` | OmniVLA-edge model checkpoint |
| `~/OmniVLA/inference/model_omnivla_edge.py` | OmniVLA-edge model architecture |
| `~/OmniVLA/inference/utils_policy.py` | OmniVLA model loader |
| `~/Desktop/Thesis/Hybrid_reID/src/` | Hybrid_reID source (YOLODetector, HybridTracker) |
| `~/Desktop/Thesis/Hybrid_reID/src/yolov8n-seg.pt` | YOLOv8n-seg weights |

---

## Setup

### Prerequisites

- CARLA 0.9.16 installed at `~/CARLA_0.9.16/`
- Conda environment `omnivla` with required packages
- Hybrid_reID repo at `~/Desktop/Thesis/Hybrid_reID/`
- OmniVLA-edge checkpoint at `~/OmniVLA/omnivla-edge/omnivla-edge.pth`

### Install dependencies (if not already done)

```bash
conda activate omnivla
pip install carla==0.9.16 opencv-python ultralytics clip-by-openai
pip install "numpy<2"   # torch requires numpy < 2
```

---

## How to Run

### Step 1 — Start CARLA

Open a terminal:

```bash
DISPLAY=:1 ~/CARLA_0.9.16/CarlaUE4.sh -windowed -ResX=1280 -ResY=720 -quality-level=Low
```

Wait ~30 seconds for CARLA to fully initialise (until the Unreal window appears).

### Step 2 — Run the follower

```bash
cd ~/OmniVLA/carla_person_follower
DISPLAY=:1 conda run -n omnivla python carla_follower.py --device cpu --npcs 0 --town Town01
```

Or with the full Python path (avoids `conda run` overhead):

```bash
DISPLAY=:1 ~/miniconda3/envs/omnivla/bin/python carla_follower.py \
    --device cpu --npcs 0 --town Town01
```

Model loading takes ~2 minutes. Once done you will see:

```
[CARLA] Pedestrian walking ahead at 0.3 m/s.
  step=   0  lin=0.000  ang=0.000  LOST
  ...
  step=  25  lin=0.000  ang=0.000  id=1 bbox=[...] dist=3.4m
```

A live OpenCV window **"Person Follower"** shows the annotated camera feed.

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | `cuda` or `cpu` — OmniVLA inference device |
| `--town` | `Town01` | CARLA map to load |
| `--tick_hz` | `3` | Simulation tick rate (Hz) |
| `--npcs` | `10` | Number of NPC vehicles (use `0` for demo) |
| `--cam_w` | `640` | Camera width (px) |
| `--cam_h` | `480` | Camera height (px) |
| `--fov` | `90.0` | Camera horizontal FOV (degrees) |
| `--crop_goal` | off | Use person crop as OmniVLA goal image (modality 9) |
| `--no_display` | off | Headless mode — disables the OpenCV window |

---

## Architecture Details

### Perception thread
- **YOLOv8n-seg** detects persons in each 640×480 BGR frame (conf ≥ 0.3)
- **HybridTracker** (DeepSORT + DINOv2 re-ID) assigns a consistent ID across frames
- Primary target is locked on the pedestrian closest to the vehicle at startup
- Track format returned: `[x1, y1, x2, y2, track_id, feature_vector(768-dim)]`
- DINOv2 runs on CPU to leave GPU VRAM for OmniVLA

### Navigation thread
- **OmniVLA-edge** takes a 6-frame rolling context (96×96 RGB) + goal pose from the tracked bbox
- Goal pose: `[fwd/0.1, -rgt/0.1, cos θ, sin θ]` derived from monocular depth estimate
- Monocular distance: `clip((focal_px × 1.7) / bbox_height, 0.3, 20.0)` metres
- Outputs 8-step waypoints → converted to `(vx, wz)` velocities
- Speed cap: vehicle never exceeds **1.5 m/s**

### Pedestrian control
- Uses `carla.WalkerControl` directly (no AI controller) so the pedestrian always walks in the vehicle's forward direction — stays in camera FOV
- Speed: **0.3 m/s** (slow natural walk, no animation shaking)
- Respawned to 5 m ahead of the ego if it drifts more than 15 m away

### Warm-up sequence
1. Tick 15 times with pedestrian stationary — collect frames, lock tracker ID=1
2. Start `WalkerControl`, tick 5 more times — re-confirm track in walking pose
3. Start perception + navigation threads
4. Enter main loop

---

## Known Issues / Notes

- **Second run without restarting CARLA**: the script destroys all leftover actors and resets sync mode on startup — no need to restart CARLA between runs
- **`--device cpu` required** if GPU VRAM < ~4 GB (DINOv2 + OmniVLA + CARLA compete for VRAM)
- **`t[5]` is a DINOv2 feature vector**, not a class_id — filter tracks with `len(t) >= 5`, not `t[5] == 0`
- OmniVLA checkpoint was trained on robot data; waypoints may be near-zero initially until the context buffer fills with real frames (~6 steps)

---

## Stopping

Press **Q** in the OpenCV window, or **Ctrl+C** in the terminal. The cleanup handler disables CARLA sync mode and destroys all spawned actors.
