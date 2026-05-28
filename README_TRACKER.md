# OmniVLA + Hybrid_ReID: Visual Person Following in CARLA

This document describes the architecture and implementation plan for replacing the CARLA oracle
(`pedestrian.get_location()`) with a purely visual tracking pipeline using Hybrid_ReID.

---

## The Core Problem

`carla_omnivla_person.py` uses `pedestrian.get_location()` — a CARLA API call that gives GPS-precise
world coordinates. A real robot has no access to this. It can only see the camera.

**Goal:** Replace that oracle with a visual tracker so the ego vehicle follows the person using
only what the camera sees — exactly as a real deployed robot would.

---

## System Architecture

```
CARLA Camera (400×300 px, 90° FOV)
        │
        ▼
HybridTracker.update(frame_bgr)          [every CARLA tick, ~3 Hz]
  ├─ YOLOv8  → person detections
  ├─ DeepSORT → Kalman filter + frame association
  ├─ DINOv2  → 768-dim ReID embedding (occlusion recovery)
  └─ → primary_bbox [x1, y1, x2, y2], track_id
        │
        ▼
bbox_to_goal_pose(bbox)                  [same tick]
  ├─ dist   = (focal_px × 1.7 m) / bbox_height
  ├─ angle  = atan2(cx − 200, 200)
  ├─ fwd    = dist × cos(angle)
  ├─ right  = dist × sin(angle)
  └─ → goal_pose [fwd/0.1, −right/0.1, cos(angle), sin(angle)]
        │  (latest goal_pose cached; oracle used as fallback)
        ▼
OmniVLA-edge inference                   [3 Hz, GPU]
  ├─ Modality 8: language + pose  (default)
  ├─ Modality 9: language + goal image   (USE_CROP_GOAL=True)
  ├─ Inputs: 6-frame context buffer, goal_pose, language embedding
  └─ → 8-waypoint predicted trajectory
        │
        ▼
Distance-keeping override                [same tick]
  ├─ dist < 1.5 m → reduce throttle (brake)
  ├─ dist > 7.0 m → increase throttle
  └─ target lost > 2 s → emergency brake
        │
        ▼
vehicle.apply_control(throttle, steer, brake)
```

---

## Camera Intrinsics

| Parameter     | Value                     |
|---------------|---------------------------|
| Resolution    | 400 × 300 px              |
| FOV           | 90°                       |
| Focal length  | `focal_px = 200 px`       |
| Mount point   | x=1.6 m, z=1.8 m on ego  |

**Distance formula (monocular):**
```
dist_m = (focal_px × person_height_m) / bbox_height_px
       = (200 × 1.7) / bbox_height_px
```

**Bearing formula:**
```
angle = atan2(bbox_center_x − 200, 200)
```

---

## OmniVLA goal_pose Format

```python
goal_pose = [
    fwd   / 0.1,    # forward distance in 0.1-m units (positive = ahead)
    -right / 0.1,   # left distance in 0.1-m units (positive = left of ego)
    cos(angle),     # heading cosine
    sin(angle),     # heading sine
]
```

Values are clamped to 8 m radius (OmniVLA's training distribution).

---

## Modality Selection

| ID | Mode              | Inputs                        | When to use                              |
|----|-------------------|-------------------------------|------------------------------------------|
| 8  | Language + Pose   | text tokens + goal_pose[4]    | Default — bbox-derived pose              |
| 9  | Language + Image  | text tokens + 96×96 crop      | Experimental — feed tracker crop as goal |
| 6  | Image Goal only   | 96×96 goal image              | No language needed                       |
| 4  | Pose only         | goal_pose[4]                  | No language signal                       |

---

## Config Flags (`carla_omnivla_tracker.py`)

| Flag              | Default | Effect                                                       |
|-------------------|---------|--------------------------------------------------------------|
| `USE_TRACKER_POSE`| `True`  | `True` = visual bbox pose; `False` = CARLA oracle (baseline) |
| `USE_CROP_GOAL`   | `False` | `True` = feed 96×96 tracker crop as goal image (modality 9) |

---

## What Each Module Provides

| Module          | What it adds                                                        |
|-----------------|---------------------------------------------------------------------|
| **HybridTracker** | Stable target identity across frames; occlusion recovery via ReID |
| **OmniVLA-edge**  | Language-conditioned path planning; handles turns and uncertainty  |
| **bbox_to_goal_pose** | No-GPS pose estimation from monocular camera                  |
| **Distance override** | Safety braking when too close; speed-up when target too far    |
| **Emergency stop**    | Stops ego if target lost for > 2 seconds                       |

**What will NOT improve without OmniVLA fine-tuning:**
- Speed matching to target (OmniVLA has a fixed max velocity from training)
- Maintaining precise 1–3 m gap
- Recovery after long occlusion (> 5 s)
- Following fast-moving targets (> 10 km/h)

---

## Comparison: Visual Tracker vs Oracle

| | Oracle (`carla_omnivla_person.py`) | Visual Tracker (`carla_omnivla_tracker.py`) |
|---|---|---|
| Pose source | `pedestrian.get_location()` | YOLOv8 bbox → pinhole model |
| Deployment | CARLA only | Real robot capable |
| Distance error | 0 m (exact) | ±0.3–1.0 m (monocular) |
| Occlusion handling | Perfect (GPS) | Hold last bbox; ReID recovery |
| VRAM overhead | None | ~2 GB (YOLOv8 + DINOv2) |

---

## Logging Output

Each step prints:
```
step=  10  lin=0.240  ang=-0.050  orc=6.2m  fwd=+6.1m  lat=-0.3m  mod=8  trk_dist=5.9m  id=1  bbox=[182,95,224,187]
```

| Field        | Meaning                                  |
|--------------|------------------------------------------|
| `lin / ang`  | OmniVLA linear / angular velocity output |
| `orc`        | Oracle ground-truth distance (m)         |
| `fwd / lat`  | Goal forward / lateral (m)               |
| `mod`        | OmniVLA modality ID (8 or 9)             |
| `trk_dist`   | Visual tracker distance estimate (m)     |
| `id`         | Assigned DeepSORT track ID               |
| `bbox`       | Tracker bounding box [x1,y1,x2,y2]       |

---

## Run Instructions

### Prerequisites

1. CARLA running:
   ```bash
   DISPLAY=:1 ~/CARLA_0.9.16/CarlaUE4.sh -windowed -ResX=1280 -ResY=720 &
   # wait ~20 seconds for CARLA to load
   ```

2. Hybrid_ReID cloned to `~/Desktop/Thesis/Hybrid_reID/`

3. OmniVLA checkpoint at `omnivla-edge/omnivla-edge.pth`

### Run the tracker script

```bash
conda activate omnivla
cd ~/OmniVLA
python inference/carla_omnivla_tracker.py
```

### Run oracle baseline for comparison

```bash
conda activate omnivla
cd ~/OmniVLA
python inference/carla_omnivla_person.py
```

### Make video from saved frames

```bash
# Tracker run
ffmpeg -framerate 3 -i inference/carla_run_tracker/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_tracker.mp4

# Oracle baseline
ffmpeg -framerate 3 -i inference/carla_run_person/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_person.mp4
```

---

## Phase Roadmap

### Phase 1 — Visual tracker as goal_pose source (NOW)
**Script:** `inference/carla_omnivla_tracker.py`

- HybridTracker replaces CARLA oracle
- `bbox_to_goal_pose()` using pinhole model
- Oracle still computed every step for quantitative comparison
- Both logged side-by-side: `orc=Xm  trk_dist=Ym`

**Success criteria:**
- `track_id=1` stable for 200+ steps (no ID switches)
- `ang` sign matches target lateral offset (steering toward target)
- `trk_dist` stays within 3–8 m (not diverging)
- Green bbox visible in saved frames
- `orc - trk_dist` error < 1.5 m on average

### Phase 2 — Goal image crop (modality 9)
**Script:** `inference/carla_omnivla_tracker.py` with `USE_CROP_GOAL=True`

- Feed 96×96 tracker crop as OmniVLA goal image
- Compare following quality: modality 8 vs 9 vs hybrid (8 fallback when no target)
- Expected: modality 9 gives stronger directional signal when crop is clean

### Phase 3 — Hierarchical control (AsyncVLA pattern)
**Script:** `inference/carla_omnivla_tracker_v2.py` (to be created)

- Decouple tracker (10–15 Hz) from OmniVLA (3–5 Hz)
- Local PID safety controller: distance keeping + emergency stop
- Cache OmniVLA waypoints; interpolate between calls
- Test occlusion recovery: walk pedestrian behind static object

Based on: **AsyncVLA** (arXiv:2602.13476) — 40% higher success rate vs OmniVLA alone.

### Phase 4 — TrackVLA as OmniVLA replacement
- **TrackVLA** (arXiv:2505.23189): VLA with `[Track]` token; 10 Hz; purpose-built for target following
- **ABot-N0** (arXiv:2602.11598): SOTA person-following; error 11.2 vs GNM's 16.2
- Replace OmniVLA in pipeline; compare following distance stability and occlusion recovery

### Phase 5 — Data collection + OmniVLA fine-tuning
- Record 500+ CARLA episodes: person / cyclist / car targets at various speeds
- Include occlusion events and re-identification events
- Fine-tune OmniVLA-edge with bbox-annotated observations + tracker crop goal image
- Target: following distance σ < 0.5 m; occlusion recovery > 80%

---

## Safety Rules

1. STOP if target not seen for > 2 seconds
2. STOP if distance < 1.0 m (collision avoidance)
3. MAX speed 0.3 m/s in person-following mode
4. MAX speed 0.8 m/s in cyclist-following mode
5. MAX speed 2.0 m/s in car-following mode
6. If target reappears after occlusion, confirm ReID similarity > 0.7 before resuming
7. Hard stop if ego velocity > 2× estimated target velocity

---

## Alternative Models (Literature)

| Model       | Hz  | Key Feature                                     | Paper             |
|-------------|-----|-------------------------------------------------|-------------------|
| **TrackVLA**    | 10  | `[Track]` token; 1.7M EVT-Bench samples     | arXiv:2505.23189  |
| **ABot-N0**     | ~10 | Qwen3-4B + Flow Matching; SOTA following    | arXiv:2602.11598  |
| **AsyncVLA**    | 20+ | Edge Adapter decouples OmniVLA latency      | arXiv:2602.13476  |
| **GNM/NoMaD**   | 5–10| Open-source; refresh goal image from crop  | Berkeley Lab      |

For new projects: **TrackVLA** or **ABot-N0** are better architectural fits than OmniVLA for tracking.
OmniVLA-edge advantage: multi-modal inputs, smaller footprint (~50M params), edge variant.

---

## Key Files

| File                                    | Purpose                              |
|-----------------------------------------|--------------------------------------|
| `inference/carla_omnivla_tracker.py`    | Phase 1: visual tracker + OmniVLA    |
| `inference/carla_omnivla_person.py`     | Oracle baseline (no change)          |
| `inference/carla_omnivla_lane.py`       | Lane driving baseline (no change)    |
| `~/Desktop/Thesis/Hybrid_reID/src/hybrid_tracker.py` | HybridTracker class    |
| `omnivla-edge/omnivla-edge.pth`         | OmniVLA checkpoint (~414 MB)         |
