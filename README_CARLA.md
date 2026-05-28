# OmniVLA-edge + CARLA — Local Setup Guide

OmniVLA is a UC Berkeley / Toyota research model for autonomous robot navigation.
This guide sets it up locally and runs it inside the CARLA simulator.

---

## Requirements

- NVIDIA GPU (tested: RTX 4070, 8 GB VRAM)
- CARLA 0.9.16 installed at `~/CARLA_0.9.16/`
- Miniconda / Anaconda
- Display (X11) for CARLA GUI

---

## 1 — One-time setup

### Clone the repo
```bash
git clone https://github.com/NHirose/OmniVLA.git
cd OmniVLA
```

### Create conda environment
```bash
conda create -n omnivla python=3.10 -y
conda activate omnivla
```

### Install PyTorch 2.2 (CUDA 12.1 bundled — no separate CUDA install needed)
```bash
pip install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Install OmniVLA package
```bash
pip install -e .
```

### Install official CLIP (replaces the broken openai-clip 1.0.1)
```bash
pip uninstall openai-clip -y
pip install git+https://github.com/openai/CLIP.git
```

### Download OmniVLA-edge checkpoint (~414 MB)
```bash
wget -O omnivla-edge/omnivla-edge.pth \
    "https://huggingface.co/NHirose/omnivla-edge/resolve/main/omnivla-edge.pth"
```
> If the `omnivla-edge/` folder doesn't exist yet, clone it first:
> ```bash
> git clone https://huggingface.co/NHirose/omnivla-edge
> ```

---

## 2 — Run OmniVLA standalone (no CARLA)

Tests the model with bundled sample images. Outputs `inference/1_ex_omnivla_edge.jpg`.

```bash
conda activate omnivla
cd ~/OmniVLA
python inference/run_omnivla_edge.py
xdg-open inference/1_ex_omnivla_edge.jpg
```

---

## 3 — Run OmniVLA inside CARLA

### Step 1 — Start CARLA server (with GUI window)
```bash
DISPLAY=:1 ~/CARLA_0.9.16/CarlaUE4.sh -windowed -ResX=1280 -ResY=720 &
```

Wait ~20 seconds for CARLA to fully load before running the next command.

To start **without a GUI** (headless / faster):
```bash
DISPLAY=:1 ~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen &
```

### Step 2a — Bike following
```bash
conda activate omnivla
cd ~/OmniVLA
python inference/carla_omnivla.py
```

### Step 2b — Person following (recommended)
```bash
conda activate omnivla
cd ~/OmniVLA
python inference/carla_omnivla_person.py
```

### Step 2c — Lane driving without person or bike following
```bash
conda activate omnivla
cd ~/OmniVLA
python inference/carla_omnivla_lane.py
```

### Step 2d — Person following with visual tracker (no CARLA oracle)
```bash
conda activate omnivla
cd ~/OmniVLA
python inference/carla_omnivla_tracker.py
```

Uses **Hybrid_ReID** (YOLOv8 + DeepSORT + DINOv2) to detect and track the pedestrian
from camera images only — no GPS or world coordinates. Requires Hybrid_reID cloned to
`~/Desktop/Thesis/Hybrid_reID/`.

**Config flags** (top of the script):

| Flag | Default | Effect |
|---|---|---|
| `USE_TRACKER_POSE` | `True` | Visual bbox pose (real-world mode). `False` = oracle baseline. |
| `USE_CROP_GOAL` | `False` | Feed tracker crop as 96×96 goal image (modality 9). Phase 2 experiment. |

Press **Ctrl+C** to stop cleanly (destroys all spawned actors).

---

## 4 — What you see in the CARLA window

The spectator camera auto-follows the ego vehicle (Tesla Model 3) in a **3rd-person chase view** — 10 m behind, 5 m above.

**Bike script (`carla_omnivla.py`):**
- A Vespa scooter spawns 15 m ahead, drives on autopilot at reduced speed
- 19 NPC vehicles drive around on autopilot
- Each step saved to `inference/carla_run/step_XXXX.jpg`

**Person script (`carla_omnivla_person.py`):**
- A pedestrian spawns 8 m ahead, walks via WalkerAIController at 1.2 m/s
- 19 NPC vehicles drive around on autopilot
- Each step saved to `inference/carla_run_person/step_XXXX.jpg`
- Better results: pedestrian speed matches OmniVLA's training distribution

**Lane script (`carla_omnivla_lane.py`):**
- No pedestrian or bicycle is spawned
- 19 NPC vehicles drive around on autopilot
- Ego uses a lane waypoint ahead as the goal pose
- Each step saved to `inference/carla_run_lane/step_XXXX.jpg`

**Tracker script (`carla_omnivla_tracker.py`):**
- Pedestrian spawns 8 m ahead, walks via WalkerAIController at 0.5 m/s
- HybridTracker (YOLOv8 + DeepSORT + DINOv2) detects and tracks the pedestrian visually
- Goal pose derived from bounding box (pinhole model) — no CARLA world coords
- Green bbox + track ID overlaid on saved step images
- Both oracle distance and tracker distance logged each step for comparison
- Each step saved to `inference/carla_run_tracker/step_XXXX.jpg`

---

## 5 — Make a video from saved frames

```bash
# Bike run
ffmpeg -framerate 3 -i inference/carla_run/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_carla_bike.mp4
xdg-open omnivla_carla_bike.mp4

# Person run
ffmpeg -framerate 3 -i inference/carla_run_person/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_carla_person.mp4
xdg-open omnivla_carla_person.mp4

# Lane run
ffmpeg -framerate 3 -i inference/carla_run_lane/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_carla_lane.mp4
xdg-open omnivla_carla_lane.mp4
```

---

## 6 — Tuning / customisation

All options are at the top of either script:

| Variable | Bike default | Person default | Description |
|---|---|---|---|
| `TOWN` | `"Town01"` | `"Town01"` | CARLA map (`Town01`–`Town07`) |
| `TICK_HZ` | `3` | `3` | Inference rate (Hz) |
| `MAX_STEPS` | `9999` | `9999` | Steps before auto-stop |
| `LANGUAGE_GOAL` | `"move toward the bike ahead"` | `"follow the person ahead"` | Language prompt |
| `SAVE_DIR` | `./inference/carla_run` | `./inference/carla_run_person` | Output directory |

**Change map**:
```python
TOWN = "Town03"   # Town03 has a roundabout and more complex roads
```

---

## 7 — Kill CARLA

```bash
pkill -f CarlaUE4
```

---

## 8 — Person following vs bike following

| | Bike (`carla_omnivla.py`) | Person (`carla_omnivla_person.py`) |
|---|---|---|
| Ego vehicle | Tesla Model 3 | Microlino (smallest available) |
| Target speed | ~6 km/h (80% reduced autopilot) | ~1.2 m/s = 4.3 km/h (WalkerAI) |
| OmniVLA match | Moderate — slight speed mismatch | **Better** — matches robot training data |
| CARLA API | Traffic manager (vehicle autopilot) | WalkerAIController |
| Spawn distance | 15 m ahead | 8 m ahead |
| Throttle scaling | 0.80 | 0.40 |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'pkg_resources'` | `pip uninstall openai-clip -y && pip install git+https://github.com/openai/CLIP.git` |
| `Model weights not found` | Re-run the `wget` command in step 1 |
| CARLA timeout after many steps | Restart CARLA with `pkill -f CarlaUE4` then re-run |
| No car visible in CARLA window | Wait for the script to print `Spawned ego vehicle` — the spectator moves there automatically |
| CARLA won't start | Check `DISPLAY=:1` is correct for your machine (`echo $DISPLAY`) |
| Pedestrian won't move | CARLA requires one `world.tick()` before `controller.start()` — already handled in the script |
| Pedestrian spawns at (0,0,0) | Normal — CARLA reports location before first tick; it teleports to correct position after warm-up |
