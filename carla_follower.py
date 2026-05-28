"""
carla_follower.py — Hybrid_reID + OmniVLA person-following in CARLA
====================================================================
Three-thread integration. Zero CARLA cheats — only camera perception.

Data flow
---------
CARLA RGB camera
  → Thread 1 (perception)  : YOLODetector → HybridTracker → SharedState
  → Thread 2 (navigation)  : SharedState  → OmniVLA-edge  → SharedState
  → Main loop  (control)   : world.tick() → apply_control() → cv2.imshow

Display
-------
One OpenCV window, same style as Hybrid_reID/src/main.py:
  green bbox + ID label + track trail for the primary person.
  Status bar at the bottom (step, dist, lin, ang).

How to run
----------
  # Terminal 1 — CARLA server
  DISPLAY=:1 ~/CARLA_0.9.16/CarlaUE4.sh -windowed -ResX=1280 -ResY=720 &

  # Terminal 2 — this script
  cd ~/OmniVLA
  DISPLAY=:1 conda run -n omnivla python carla_follower.py

Optional flags
--------------
  --town      Town01 (default)
  --tick_hz   3      (default)
  --npcs      10     (default)
  --crop_goal        use person-crop as OmniVLA goal image (modality 9)
  --no_display       headless mode (no cv2 window)
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import carla
# Suppress Qt font/debug spam and point it at system fonts so window doesn't flicker
os.environ["QT_LOGGING_RULES"]  = "*.debug=false;qt.qpa.*=false"
os.environ["QT_QPA_FONTDIR"]    = "/usr/share/fonts/truetype"
import cv2
import numpy as np
from PIL import Image

# ── path setup ──────────────────────────────────────────────��─────────────────
# OmniVLA root so utils_policy is importable
_OMNIVLA_ROOT = os.path.dirname(os.path.abspath(__file__))
_OMNIVLA_INF  = os.path.join(_OMNIVLA_ROOT, "inference")
# Hybrid_reID source (contains yolo_detector, hybrid_tracker, etc.)
_HYBRID_SRC   = os.path.expanduser("~/Desktop/Thesis/Hybrid_reID/src")

# Hybrid_reID imports relative paths (profiling, utils …); chdir before import
_ORIG_CWD = os.getcwd()
os.chdir(_HYBRID_SRC)
sys.path.insert(0, _HYBRID_SRC)

from yolo_detector  import YOLODetector
from hybrid_tracker import HybridTracker

# restore cwd so OmniVLA can find its own assets
os.chdir(_OMNIVLA_ROOT)
for _p in (_OMNIVLA_ROOT, _OMNIVLA_INF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import clip
from utils_policy import load_model, transform_images_PIL_mask, transform_images_map

print("[Init] All imports OK.")


# ══════════════════════════════════════════════════════════════════════════════
# § 1  Thread-safe Shared State
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SharedState:
    """Cross-thread data — one lock guards everything."""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    running: bool = True

    # perception outputs (written by T1, read by T2 + main)
    bbox:            Optional[List[int]]   = None   # [x1, y1, x2, y2]
    crop_pil:        Optional[Image.Image] = None   # RGB person crop
    obs_frame_pil:   Optional[Image.Image] = None   # 96×96 RGB current frame for OmniVLA context
    track_id:        Optional[int]         = None
    dist_m:          float                 = 0.0
    target_visible:  bool                  = False
    track_history:   dict = field(default_factory=dict)  # id → deque[(cx,cy)]

    # navigation output (written by T2, read by main)
    vx: float = 0.0   # linear  velocity
    wz: float = 0.0   # angular velocity

    # latest annotated frame for display (written by T1, read by main)
    display_frame: Optional[np.ndarray] = None

    def write_track(self, bbox, crop, obs_frame, track_id, dist_m, history: dict) -> None:
        with self._lock:
            self.bbox           = bbox
            self.crop_pil       = crop
            self.obs_frame_pil  = obs_frame
            self.track_id       = track_id
            self.dist_m         = dist_m
            self.target_visible = True
            self.track_history  = history

    def write_lost(self, obs_frame=None) -> None:
        with self._lock:
            self.bbox           = None
            self.crop_pil       = None
            if obs_frame is not None:
                self.obs_frame_pil = obs_frame
            self.track_id       = None
            self.dist_m         = 0.0
            self.target_visible = False

    def read_perception(self):
        with self._lock:
            return (self.bbox, self.crop_pil, self.obs_frame_pil,
                    self.track_id, self.dist_m, self.target_visible,
                    dict(self.track_history))

    def write_velocity(self, vx: float, wz: float) -> None:
        with self._lock:
            self.vx = vx
            self.wz = wz

    def read_velocity(self) -> Tuple[float, float]:
        with self._lock:
            return self.vx, self.wz

    def write_display(self, frame: np.ndarray) -> None:
        with self._lock:
            self.display_frame = frame

    def read_display(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.display_frame


# ══════════════════════════════════════════════════════════════════════════════
# § 2  Geometry helpers  (visual-only — no CARLA coordinates)
# ══════════════════════════════════════════════════════════════════════════════

def _scalar(v, default=0) -> int:
    """Safely convert a numpy scalar/array or None to int."""
    if v is None:
        return default
    try:
        return int(np.asarray(v).flat[0])
    except Exception:
        return default


def bbox_to_goal_pose(
    bbox: List[int],
    cam_w: int,
    focal_px: float,
    person_h_m: float = 1.7,
    spacing: float = 0.1,
) -> np.ndarray:
    """
    Pinhole-camera monocular depth + bearing → OmniVLA goal_pose.

    Returns float32[4]: [fwd/spacing, -rgt/spacing, cos θ, sin θ]
    No CARLA data used — only the bounding box.
    """
    x1, y1, x2, y2 = bbox
    bh    = max(float(y2 - y1), 1.0)
    bcx   = (float(x1) + float(x2)) / 2.0
    dist  = float(np.clip((focal_px * person_h_m) / bh, 0.5, 8.0))
    angle = math.atan2(bcx - cam_w / 2.0, focal_px)
    fwd   = dist * math.cos(angle)
    rgt   = dist * math.sin(angle)
    return np.array([fwd / spacing, -rgt / spacing,
                     math.cos(angle), math.sin(angle)], dtype=np.float32)


def bbox_dist_m(bbox: List[int], focal_px: float,
                person_h_m: float = 1.7) -> float:
    bh = max(float(bbox[3]) - float(bbox[1]), 1.0)
    return float(np.clip((focal_px * person_h_m) / bh, 0.3, 20.0))


def _clip_angle(theta: float) -> float:
    while theta >  math.pi: theta -= 2 * math.pi
    while theta < -math.pi: theta += 2 * math.pi
    return theta


def waypoints_to_vel(waypoints: np.ndarray, tick_hz: float = 3.0) -> Tuple[float, float]:
    """
    Convert OmniVLA waypoint[4] (index 4) to (linear_vel, angular_vel).
    Applies the same velocity-coupling constraint used across all OmniVLA CARLA scripts.
    """
    DT     = 1.0 / tick_hz
    chosen = waypoints[4].copy()
    chosen[:2] *= 0.1   # denormalise: spacing = 0.1 m
    dx, dy, hx, hy = chosen

    EPS = 1e-8
    if abs(dx) < EPS and abs(dy) < EPS:
        lin = 0.0
        ang = _clip_angle(float(np.arctan2(hy, hx))) / DT
    elif abs(dx) < EPS:
        lin = 0.0
        ang = float(np.sign(dy)) * math.pi / (2 * DT)
    else:
        lin = dx / DT
        ang = float(np.arctan(dy / dx)) / DT

    lin = float(np.clip(lin, 0.0, 0.5))
    ang = float(np.clip(ang, -1.0, 1.0))

    maxv = maxw = 0.3
    if abs(lin) <= maxv:
        if abs(ang) <= maxw:
            vx, wz = lin, ang
        else:
            rd = lin / (ang + 1e-9)
            vx = maxw * float(np.sign(lin)) * abs(rd)
            wz = maxw * float(np.sign(ang))
    else:
        if abs(ang) <= 0.001:
            vx, wz = maxv * float(np.sign(lin)), 0.0
        else:
            rd = lin / ang
            if abs(rd) >= maxv / maxw:
                vx = maxv * float(np.sign(lin))
                wz = maxv * float(np.sign(ang)) / abs(rd)
            else:
                vx = maxw * float(np.sign(lin)) * abs(rd)
                wz = maxw * float(np.sign(ang))

    return float(vx), float(wz)


def vel_to_control(vx: float, wz: float, dist_m: float,
                   min_dist: float = 1.5, max_dist: float = 7.0):
    """(vx, wz, dist) → carla.VehicleControl with distance-keeping override."""
    if dist_m > 0:
        if dist_m < min_dist:
            vx = max(0.0, vx - 0.15)
        elif dist_m > max_dist:
            vx = min(0.2, vx + 0.05)
    throttle = float(np.clip(vx / 0.3 * 0.45, 0.0, 1.0))
    brake    = 0.3 if vx < 0.02 else 0.0
    steer    = float(np.clip(-wz / 0.3, -1.0, 1.0))
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


# ══════════════════════════════════════════════════════════════════════════════
# § 3  Thread 1 — Perception  (YOLO + HybridTracker)
# ══════════════════════════════════════════════════════════════════════════════

def perception_thread(
    state:     SharedState,
    frame_q:   queue.Queue,   # BGR frames from CARLA camera
    detector:  YOLODetector,
    tracker:   HybridTracker,
    cam_w:     int,
    cam_h:     int,
    focal_px:  float,
    use_crop:  bool,
    show_display: bool,
) -> None:
    """
    Reads CARLA frames, runs YOLO + HybridTracker, writes to SharedState.

    Annotation is drawn here (same style as Hybrid_reID/src/main.py) and
    stored in state.display_frame for the main thread's cv2.imshow.
    """
    print("[Perception] Thread started.")
    _LOST_COLOR    = (0,   0, 200)   # red
    _PRIMARY_COLOR = (0, 255,   0)   # green  — same as Hybrid_reID main.py

    while state.running:
        try:
            frame_bgr = frame_q.get(timeout=1.0)
        except queue.Empty:
            continue

        # Build 96×96 PIL of this frame for OmniVLA context buffer
        obs_96 = Image.fromarray(
            cv2.resize(frame_bgr, (96, 96))[:, :, ::-1].copy()
        )

        # ── YOLO detection ────────────────────────────────────────────────────
        try:
            os.chdir(_HYBRID_SRC)
            detections = detector.detect(frame_bgr)
            os.chdir(_OMNIVLA_ROOT)
        except Exception as exc:
            os.chdir(_OMNIVLA_ROOT)
            print(f"[Perception] detect() error: {exc}")
            state.write_lost(obs_96)
            continue

        # ── HybridTracker update ──────────────────────────────────────────────
        try:
            os.chdir(_HYBRID_SRC)
            tracks = tracker.update(frame_bgr, detections)
            os.chdir(_OMNIVLA_ROOT)
        except Exception as exc:
            os.chdir(_OMNIVLA_ROOT)
            print(f"[Perception] tracker.update() error: {exc}")
            state.write_lost(obs_96)
            continue

        # ── extract primary person track (ID == 1, or lowest ID) ─────────────
        # Track format: [x1,y1,x2,y2, track_id, feature_vector(768-dim)]
        # t[5] is DINOv2 features, not a class_id. Since YOLO is person-only,
        # all confirmed tracks are persons — no class filter needed.
        persons = [t for t in tracks if len(t) >= 5]

        display = frame_bgr.copy()
        vx, wz  = state.read_velocity()

        if persons:
            # prefer track ID == 1 (primary object set during auto-select)
            primary_list = [t for t in persons if _scalar(t[4]) == 1]
            chosen       = primary_list[0] if primary_list else min(
                               persons, key=lambda t: _scalar(t[4]))

            x1 = _scalar(chosen[0]); y1 = _scalar(chosen[1])
            x2 = _scalar(chosen[2]); y2 = _scalar(chosen[3])
            tid = _scalar(chosen[4])

            # person crop for OmniVLA goal image
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(cam_w, x2), min(cam_h, y2)
            crop_pil = None
            if use_crop and cx2 > cx1 and cy2 > cy1:
                crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
                crop_pil = Image.fromarray(crop_bgr[:, :, ::-1])

            dist = bbox_dist_m([x1, y1, x2, y2], focal_px)
            state.write_track([x1, y1, x2, y2], crop_pil, obs_96, tid, dist,
                              dict(tracker.track_history)
                              if hasattr(tracker, "track_history") else {})

            # ── draw: bbox + ID label (Hybrid_reID style) ─────────────────────
            cv2.rectangle(display, (x1, y1), (x2, y2), _PRIMARY_COLOR, 3)
            cv2.putText(display, f"ID: {tid}", (x1, max(y1 - 15, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, _PRIMARY_COLOR, 3)

            # track trail
            if tid in getattr(tracker, "track_history", {}):
                pts = list(tracker.track_history[tid])
                for i in range(1, len(pts)):
                    cv2.line(display,
                             (int(pts[i-1][0]), int(pts[i-1][1])),
                             (int(pts[i][0]),   int(pts[i][1])),
                             _PRIMARY_COLOR, 3)

            # dist overlay
            cv2.putText(display, f"{dist:.1f}m", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _PRIMARY_COLOR, 2)

        else:
            state.write_lost(obs_96)
            cv2.putText(display, "SEARCHING …", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, _LOST_COLOR, 2)

        # ── status bar at bottom (same as Hybrid_reID main.py style) ─────────
        if show_display:
            _, _, _, _, dist_s, vis, _ = state.read_perception()
            status = (f"lin={vx:+.2f}  ang={wz:+.2f}  "
                      f"dist={dist_s:.1f}m  "
                      f"{'TRACKING' if vis else 'LOST'}")
            cv2.rectangle(display,
                          (0, cam_h - 22), (cam_w, cam_h), (0, 0, 0), -1)
            cv2.putText(display, status, (6, cam_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 220, 0), 1)

            state.write_display(display)

    print("[Perception] Thread exited.")


# ══════════════════════════════════════════════════════════════════════════════
# § 4  Thread 2 — Navigation  (OmniVLA-edge inference)
# ══════════════════════════════════════════════════════════════════════════════

def navigation_thread(
    state:      SharedState,
    model,
    device,
    feat_text,
    cam_w:      int,
    focal_px:   float,
    tick_hz:    int,
    use_crop:   bool,
) -> None:
    """
    Reads SharedState, runs OmniVLA-edge inference at tick_hz, writes vx/wz.

    Inputs to OmniVLA (all from camera perception — no CARLA data):
      obs_images : 6-frame rolling context (96×96)
      goal_pose  : [fwd/0.1, -rgt/0.1, cos θ, sin θ]  from bbox
      goal_image : person crop (modality 9) or grey (modality 8)
      language   : "follow the person ahead"  (pre-encoded)
    """
    print("[Navigation] Thread started.")
    DT      = 1.0 / tick_hz
    IMGSIZE = (96, 96)
    CLIPSIZE= (224, 224)

    mask96  = np.ones((96,  96,  3), dtype=np.float32)
    mask224 = np.ones((224, 224, 3), dtype=np.float32)
    gray96  = Image.new("RGB", IMGSIZE,  (128, 128, 128))
    gray224 = Image.new("RGB", CLIPSIZE, (128, 128, 128))

    ctx_buf: deque = deque(maxlen=6)
    for _ in range(6):
        ctx_buf.append(gray96)

    last_bbox = None   # hold last known bbox if target temporarily lost

    while state.running:
        t0 = time.perf_counter()

        bbox, crop_pil, obs_frame_pil, _, dist_m, visible, _ = state.read_perception()

        if not visible:
            # hold last pose briefly; stop if never seen anyone
            if last_bbox is None:
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, DT - elapsed))
                continue
            bbox = last_bbox
        else:
            last_bbox = bbox

        # use actual camera frame in context buffer (or grey if not yet available)
        ctx_frame = obs_frame_pil if obs_frame_pil is not None else gray96

        try:
            goal_pose_np = bbox_to_goal_pose(bbox, cam_w, focal_px)
            goal_pose_t  = (torch.tensor(goal_pose_np, dtype=torch.float32)
                            .unsqueeze(0).to(device))

            cur_large_pil = obs_frame_pil.resize((224, 224)) if obs_frame_pil is not None else gray224
            cur_large = transform_images_PIL_mask(cur_large_pil, mask224).to(device)

            # rolling context — use actual camera frame so OmniVLA sees the scene
            ctx_buf.append(ctx_frame)
            buf = list(ctx_buf) if len(ctx_buf) >= 6 else [gray96] * (6 - len(ctx_buf)) + list(ctx_buf)
            obs_t    = transform_images_PIL_mask(buf, mask96)
            parts    = torch.split(obs_t.to(device), 3, dim=1)
            obs_cur  = parts[-1]
            obs_all  = torch.cat(parts, dim=1)

            sat   = Image.new("RGB", (352, 352), (0, 0, 0))
            map_t = torch.cat((
                transform_images_map(sat).to(device),
                transform_images_map(sat).to(device),
                obs_cur,
            ), dim=1)

            use_crop_now = use_crop and crop_pil is not None
            gimg     = crop_pil.resize(IMGSIZE) if use_crop_now else gray96
            gimg_t   = transform_images_PIL_mask(gimg, mask96).to(device)
            mod_t    = torch.tensor([9 if use_crop_now else 8],
                                    dtype=torch.long).to(device)

            with torch.no_grad():
                pred_actions, _, _ = model(
                    obs_all, goal_pose_t, map_t,
                    gimg_t, mod_t, feat_text, cur_large,
                )

            waypoints = pred_actions.float().cpu().numpy()
            vx, wz    = waypoints_to_vel(waypoints[0], tick_hz)
            state.write_velocity(vx, wz)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print("[Navigation] CUDA OOM — switching to CPU.")
                device = torch.device("cpu")
                model  = model.cpu()
                feat_text = feat_text.cpu()
            else:
                print(f"[Navigation] forward error: {exc}")
        except Exception as exc:
            print(f"[Navigation] error: {exc}")

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, DT - elapsed))

    print("[Navigation] Thread exited.")


# ══════════════════════════════════════════════════════════════════════════════
# § 5  CARLA world setup  (no oracle data passed to perception/nav)
# ══════════════════════════════════════════════════════════════════════════════

def _spawn_ego(world, bp_lib, spawn_points):
    preferred = ["vehicle.micro.microlino", "vehicle.nissan.micra",
                 "vehicle.seat.leon", "vehicle.audi.tt"]
    all4 = sorted([b for b in bp_lib.filter("vehicle.*")
                   if b.has_attribute("number_of_wheels")
                   and int(b.get_attribute("number_of_wheels")) == 4],
                  key=lambda b: b.id)
    bps = []
    for vid in preferred:
        try: bps.append(bp_lib.find(vid))
        except IndexError: pass
    bps += all4
    for sp in spawn_points:
        for bp in bps:
            v = world.try_spawn_actor(bp, sp)
            if v:
                return v
    raise RuntimeError("No free spawn point for ego vehicle.")


def _spawn_pedestrian(world, bp_lib, ego_tf, lead_dist: float = 5.0):
    fwd = ego_tf.get_forward_vector()
    # right-perpendicular vector (for lateral offsets)
    rgt = carla.Vector3D(x=-fwd.y, y=fwd.x, z=0.0)
    loc = ego_tf.location
    ped_bp = random.choice(list(bp_lib.filter("walker.pedestrian.*")))

    # Try positions: directly ahead first, then small lateral fallbacks
    ped = None
    candidates = []
    for dist in [5.0, 6.0, 4.0, 7.0, 8.0]:
        for lat in [0.0, 1.0, -1.0, 2.0, -2.0]:
            candidates.append((
                loc.x + fwd.x * dist + rgt.x * lat,
                loc.y + fwd.y * dist + rgt.y * lat,
                loc.z + 0.5,
            ))

    for cx, cy, cz in candidates:
        sp = carla.Transform(
            carla.Location(x=cx, y=cy, z=cz),
            ego_tf.rotation,
        )
        ped = world.try_spawn_actor(ped_bp, sp)
        if ped is not None:
            print(f"[CARLA] Pedestrian spawned at ({cx:.1f}, {cy:.1f}) "
                  f"— dist={math.hypot(cx - loc.x, cy - loc.y):.1f}m ahead of ego")
            break

    if ped is None:
        # Last resort: random nav mesh — warn loudly so user knows YOLO may not see it
        nav_loc = world.get_random_location_from_navigation()
        ped = world.spawn_actor(ped_bp, carla.Transform(nav_loc))
        print(f"[WARN] Pedestrian random-spawn at {nav_loc} — "
              "may NOT be visible in camera; detection will be 0 until pedestrian enters FOV")

    world.tick()
    return ped, None   # No AI controller — movement via WalkerControl in main loop


# ══════════════════════════════════════════════════════════════════════════════
# § 6  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Person-following: Hybrid_reID + OmniVLA in CARLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int, default=2000)
    p.add_argument("--town",       default="Town01")
    p.add_argument("--tick_hz",    type=int, default=3)
    p.add_argument("--npcs",       type=int, default=10)
    p.add_argument("--max_steps",  type=int, default=9999)
    p.add_argument("--cam_w",      type=int, default=640)
    p.add_argument("--cam_h",      type=int, default=480)
    p.add_argument("--fov",        type=float, default=90.0)
    p.add_argument("--checkpoint",
                   default=os.path.join(_OMNIVLA_ROOT, "omnivla-edge", "omnivla-edge.pth"))
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--crop_goal",  action="store_true",
                   help="Use person crop as OmniVLA goal image (modality 9)")
    p.add_argument("--no_display", action="store_true",
                   help="Disable OpenCV window (headless/server mode)")
    p.add_argument("--yolo_path",
                   default=os.path.join(_HYBRID_SRC, "yolov8n-seg.pt"))
    return p.parse_args()


def main():
    args = parse_args()

    # ── OmniVLA model ─────────────────────────────────────────────────────────
    device = (torch.device("cuda:0") if args.device == "cuda"
              and torch.cuda.is_available() else torch.device("cpu"))
    print(f"[OmniVLA] Device: {device}")

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # Force CLIP to load on the same device as OmniVLA.
    # utils_policy.py calls clip.load() without a device arg → always uses CUDA,
    # causing OOM even when --device cpu is passed. Monkey-patch before load_model().
    import clip as _clip_mod
    _orig_clip_load = _clip_mod.load
    _clip_device = str(device)
    def _device_clip_load(name, device=None, *args, **kwargs):
        return _orig_clip_load(name, device=_clip_device, *args, **kwargs)
    _clip_mod.load = _device_clip_load

    # model_omnivla_edge.py line 241 does: device = obs_img.get_device()
    # get_device() returns -1 for CPU tensors → self.all_masks.to(-1) raises
    # "Device index must not be negative". Patch it to return torch.device('cpu').
    _orig_get_device = torch.Tensor.get_device
    def _safe_get_device(self):
        if not self.is_cuda:
            return torch.device("cpu")
        return _orig_get_device(self)
    torch.Tensor.get_device = _safe_get_device

    model_params = {
        "model_type": "omnivla-edge", "len_traj_pred": 8,
        "learn_angle": True, "context_size": 5,
        "obs_encoder": "efficientnet-b0", "encoding_size": 256,
        "obs_encoding_size": 1024, "goal_encoding_size": 1024,
        "late_fusion": False, "mha_num_attention_heads": 4,
        "mha_num_attention_layers": 4, "mha_ff_dim_factor": 4,
        "clip_type": "ViT-B/32",
    }
    LANG_GOAL = "follow the person ahead"
    print(f"[OmniVLA] Loading checkpoint {ckpt} on {device} …")
    model, text_enc, _ = load_model(ckpt, model_params, device)
    text_enc = text_enc.to(device).eval()
    model    = model.to(device).eval()
    _tokens  = clip.tokenize(LANG_GOAL, truncate=True).to(device)
    with torch.no_grad():
        feat_text = text_enc.encode_text(_tokens).float()
    print(f'[OmniVLA] Ready. Language goal: "{LANG_GOAL}"')

    # ── Hybrid_reID tracker ───────────────────────────────────────────────────
    print("[Hybrid_reID] Initialising YOLODetector …")
    os.chdir(_HYBRID_SRC)
    detector = YOLODetector(
        model_path=args.yolo_path,
        conf_threshold=0.3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    tracker = HybridTracker(
        max_cosine_distance=0.15,
        nn_budget=1000,
        max_age=30,
        min_confidence=0.3,
        re_id_interval=999,   # DINOv2 only on track loss — frees GPU for OmniVLA
        gallery_size=500,
    )
    # DINOv2 → CPU so OmniVLA has full GPU VRAM
    try:
        fe = tracker.feature_extractor
        fe.model  = fe.model.cpu()
        fe.device = torch.device("cpu")
        print("[Hybrid_reID] DINOv2 moved to CPU — GPU reserved for OmniVLA.")
    except AttributeError:
        print("[Hybrid_reID] WARNING: could not move DINOv2 to CPU.")
    os.chdir(_OMNIVLA_ROOT)
    print("[Hybrid_reID] Ready.")

    # ── camera intrinsics ─────────────────────────────────────────────────────
    focal_px = args.cam_w / (2.0 * math.tan(math.radians(args.fov / 2.0)))

    # ── CARLA world ───────────────────────────────────────────────────────────
    print(f"[CARLA] Connecting to {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    # Port 2000 opens before CARLA is fully ready — retry until responsive.
    world = None
    for attempt in range(30):
        try:
            world = client.get_world()
            break
        except RuntimeError:
            print(f"[CARLA] Not ready yet ({attempt+1}/30), retrying …")
            time.sleep(3.0)
    if world is None:
        raise RuntimeError("CARLA did not become ready after 90 s")

    client.set_timeout(120.0)

    cur_map = world.get_map().name
    if args.town not in cur_map:
        print(f"[CARLA] Loading {args.town} …")
        world = client.load_world(args.town)
        time.sleep(2.0)
        world = client.get_world()   # refresh reference after map reload
    else:
        print(f"[CARLA] Already on {cur_map}.")
    print(f"[CARLA] Map: {world.get_map().name}")

    # Reset world to async mode first — this unsticks any leftover sync/TM state
    # from a previous run that was killed with pkill -9 (cleanup never ran).
    settings = world.get_settings()
    settings.synchronous_mode    = False
    settings.fixed_delta_seconds = 0.0
    world.apply_settings(settings)
    time.sleep(0.5)

    # Reset TM to async before we try to use it (avoids 120 s deadlock)
    try:
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(False)
    except Exception as e:
        print(f"[CARLA] TM reset warning (safe to ignore if --npcs 0): {e}")
        tm = None

    # Destroy leftover actors from a previous run
    for actor in world.get_actors():
        if actor.type_id.startswith(("vehicle.", "walker.", "controller.")):
            try:
                actor.destroy()
            except Exception:
                pass
    time.sleep(0.5)

    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / args.tick_hz
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    if tm is None:
        tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    bp_lib       = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spectator    = world.get_spectator()
    actors       = []

    # ── spawn ego + pedestrian ────────────────────────────────────────────────
    vehicle = _spawn_ego(world, bp_lib, spawn_points)
    actors.append(vehicle)
    print(f"[CARLA] Ego: {vehicle.type_id}")

    world.tick()
    ego_tf = vehicle.get_transform()

    ped, _ = _spawn_pedestrian(world, bp_lib, ego_tf)
    actors.append(ped)
    print(f"[CARLA] Pedestrian: {ped.type_id}  at {ped.get_location()}")

    # ── NPC traffic ───────────────────────────────────────────────────────────
    ego_loc   = vehicle.get_location()
    npc_count = 0
    for sp in spawn_points:
        if npc_count >= args.npcs:
            break
        if sp.location.distance(ego_loc) < 5.0:
            continue
        cands = [b for b in bp_lib.filter("vehicle.*")
                 if b.has_attribute("number_of_wheels")
                 and int(b.get_attribute("number_of_wheels")) == 4]
        npc = world.try_spawn_actor(random.choice(cands), sp)
        if npc:
            npc.set_autopilot(True, 8000)
            actors.append(npc)
            npc_count += 1
    print(f"[CARLA] {npc_count} NPC vehicles.")

    # ── RGB camera ────────────────────────────────────────────────────────────
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.cam_w))
    cam_bp.set_attribute("image_size_y", str(args.cam_h))
    cam_bp.set_attribute("fov",          str(args.fov))
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.6, z=1.8)),
        attach_to=vehicle,
    )
    actors.append(camera)

    img_queue   = queue.Queue(maxsize=4)
    frame_queue = queue.Queue(maxsize=4)   # BGR frames for perception thread

    def _on_image(raw):
        arr = np.frombuffer(raw.raw_data, dtype=np.uint8)
        bgr = arr.reshape(raw.height, raw.width, 4)[:, :, :3].copy()
        if img_queue.full():
            try: img_queue.get_nowait()
            except queue.Empty: pass
        img_queue.put(bgr)

    camera.listen(_on_image)

    # ── cleanup ───────────────────────────────────────────────────────────────
    def cleanup(sig=None, _f=None):
        print("\n[Main] Cleaning up …")
        state.running = False
        time.sleep(0.3)   # let threads notice
        settings.synchronous_mode = False
        tm.set_synchronous_mode(False)
        world.apply_settings(settings)
        # stop pedestrian
        try:
            wc = carla.WalkerControl(); wc.speed = 0.0
            ped.apply_control(wc)
        except Exception: pass
        for a in actors:
            try:
                if a.is_alive: a.destroy()
            except Exception: pass
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[Main] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    # ── shared state ──────────────────────────────────────────────────────────
    state = SharedState()

    # ── warm-up: collect frames + lock tracker ID=1 ──────────────────────────
    # DeepSORT requires N_INIT=3 consecutive detections to confirm a track.
    # We tick 15 times and run detect+track on each frame so ID=1 is confirmed
    # before the perception thread starts.
    print("[Init] Warm-up (collecting frames, pedestrian is stationary) …")
    warmup_frames = []
    for _ in range(15):
        world.tick()
        try:
            warmup_frames.append(img_queue.get(timeout=1.0))
        except queue.Empty:
            pass

    if warmup_frames:
        # auto-select on the last frame (freshest view of the stationary pedestrian)
        first_bgr = warmup_frames[-1]
        print("[Hybrid_reID] Auto-selecting primary target …")
        os.chdir(_HYBRID_SRC)
        init_dets = detector.detect(first_bgr)
        os.chdir(_OMNIVLA_ROOT)
        persons = [d for d in init_dets if len(d) >= 6 and int(d[5]) == 0]
        if persons:
            best = max(persons, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            tracker.set_primary_object_by_bbox(best[:4])
            print(f"[Hybrid_reID] Primary target bbox={[int(v) for v in best[:4]]}")

            # Run tracker.update() on the last 5 warm-up frames so DeepSORT
            # accumulates the 3 consecutive hits it needs to confirm the track.
            for i, wf in enumerate(warmup_frames[-5:]):
                os.chdir(_HYBRID_SRC)
                wf_dets     = detector.detect(wf)
                wf_tracks   = tracker.update(wf, wf_dets)
                os.chdir(_OMNIVLA_ROOT)
                id1 = [t for t in wf_tracks if len(t) >= 5 and _scalar(t[4]) == 1]
                print(f"[Hybrid_reID] warm-up {i+1}/5  ID=1={bool(id1)}  tracks={len(wf_tracks)}")
        else:
            print("[Hybrid_reID] No person visible in warm-up — will auto-select on first live frame.")
            if hasattr(tracker, "waiting_for_selection"):
                tracker.waiting_for_selection = False

    # ── pedestrian direct control — walk straight ahead of vehicle ────────────
    # WalkerControl (no AI controller): pedestrian walks in vehicle's forward
    # direction, guaranteed to stay in camera FOV.
    # 0.3 m/s = visibly slow natural walk without animation shaking.
    _PED_SPEED = 0.3

    def _ped_forward_dir():
        """Vehicle forward direction — used as pedestrian walking direction."""
        return vehicle.get_transform().get_forward_vector()

    def _apply_walker_control(direction, speed):
        wc   = carla.WalkerControl()
        norm = math.sqrt(direction.x**2 + direction.y**2 + 1e-9)
        wc.direction = carla.Vector3D(x=direction.x/norm, y=direction.y/norm, z=0.0)
        wc.speed = speed
        wc.jump  = False
        ped.apply_control(wc)

    _ped_dir = _ped_forward_dir()
    _apply_walker_control(_ped_dir, _PED_SPEED)

    # Post-start warmup: tick 5 more times with pedestrian already walking so
    # DeepSORT re-confirms the track in the walking pose before the main loop.
    print("[Init] Post-start warmup (re-confirming track while pedestrian walks) …")
    for i in range(5):
        world.tick()
        try:
            wf = img_queue.get(timeout=1.0)
            _apply_walker_control(_ped_dir, _PED_SPEED)
            os.chdir(_HYBRID_SRC)
            wf_dets   = detector.detect(wf)
            wf_tracks = tracker.update(wf, wf_dets)
            os.chdir(_OMNIVLA_ROOT)
            id1 = [t for t in wf_tracks if len(t) >= 5 and _scalar(t[4]) == 1]
            print(f"[Hybrid_reID] post-warmup {i+1}/5  ID=1={bool(id1)}  tracks={len(wf_tracks)}")
        except queue.Empty:
            pass

    print(f"[CARLA] Pedestrian walking ahead at {_PED_SPEED} m/s.")

    # ── start perception and navigation threads ───────────────────────────────
    t_perc = threading.Thread(
        target=perception_thread,
        args=(state, frame_queue, detector, tracker,
              args.cam_w, args.cam_h, focal_px,
              args.crop_goal, not args.no_display),
        name="perception", daemon=True,
    )
    t_nav = threading.Thread(
        target=navigation_thread,
        args=(state, model, device, feat_text,
              args.cam_w, focal_px, args.tick_hz, args.crop_goal),
        name="navigation", daemon=True,
    )
    t_perc.start()
    t_nav.start()

    # Test whether cv2 display is available; auto-disable if not (headless conda)
    _can_display = False
    if not args.no_display:
        try:
            # startWindowThread keeps Qt event loop alive between waitKey calls,
            # preventing the flickering that occurs at low tick rates (3 Hz).
            cv2.startWindowThread()
            cv2.namedWindow("Person Follower", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Person Follower", args.cam_w, args.cam_h)
            _can_display = True
        except (cv2.error, Exception):
            print("[Main] OpenCV has no display backend — running headless.")
            args.no_display = True

    print(f"\n[Main] Running — press Q to quit.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Main loop: CARLA tick → feed frames → apply control → display
    # ══════════════════════════════════════════════════════════════════════════
    _last_disp: Optional[np.ndarray] = None   # hold last valid frame — prevents blank flicker
    _last_ped_retarget = 0                     # step of last pedestrian re-target

    for step in range(args.max_steps):
        if not state.running:
            break

        # ── CARLA simulation tick ─────────────────────────────────────────────
        world.tick()

        # ── get camera frame ──────────────────────────────────────────────────
        try:
            bgr = img_queue.get(timeout=2.0)
        except queue.Empty:
            print(f"  step={step:4d}  [WARN] camera timeout")
            continue

        # push to perception thread (drop oldest if it can't keep up)
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(bgr)

        # ── read latest navigation output ─────────────────────────────────────
        vx, wz = state.read_velocity()

        # ── read perception for dist + visibility ─────────────────────────────
        bbox, _, _, track_id, dist_m, visible, _ = state.read_perception()

        # ── apply control (visual perception only — no CARLA cheats) ─────────
        cur_v   = vehicle.get_velocity()
        cur_spd = math.sqrt(cur_v.x**2 + cur_v.y**2)

        if not visible:
            # Don't brake when LOST — use last OmniVLA direction to keep
            # creeping toward where the pedestrian was last seen.
            if vx > 0.01:
                ctrl = vel_to_control(vx * 0.5, wz, 10.0)
                ctrl.throttle = max(ctrl.throttle, 0.12)
            else:
                ctrl = carla.VehicleControl(throttle=0.12, brake=0.0)
            if cur_spd > 1.0:
                ctrl.throttle = 0.0
                ctrl.brake    = 0.2
            vehicle.apply_control(ctrl)
        else:
            ctrl = vel_to_control(vx, wz, dist_m)
            # When OmniVLA outputs near-zero (still warming up), nudge forward slowly.
            if ctrl.throttle < 0.05 and dist_m > 2.0:
                ctrl.throttle = 0.20
                ctrl.brake    = 0.0
            # Hard speed cap: never exceed 1.5 m/s (walking pace).
            if cur_spd > 1.5:
                ctrl.throttle = 0.0
                ctrl.brake    = 0.2
            vehicle.apply_control(ctrl)

        # ── pedestrian: apply WalkerControl every step ───────────────────────────
        # Always walk in the vehicle's forward direction — this keeps the pedestrian
        # centered in the camera's FOV.
        # Every 30 steps: respawn close to ego if it drifted too far away.
        try:
            _ped_dir = _ped_forward_dir()
            _apply_walker_control(_ped_dir, _PED_SPEED)
        except Exception:
            pass

        if step - _last_ped_retarget >= 30:
            try:
                ped_loc  = ped.get_location()
                ego_loc2 = vehicle.get_location()
                if ped_loc.distance(ego_loc2) > 15.0:
                    ego_tf2 = vehicle.get_transform()
                    fwd2    = ego_tf2.get_forward_vector()
                    ped.set_location(carla.Location(
                        x=ego_tf2.location.x + fwd2.x * 5.0,
                        y=ego_tf2.location.y + fwd2.y * 5.0,
                        z=ego_tf2.location.z + 0.5,
                    ))
            except Exception:
                pass
            _last_ped_retarget = step

        # ── spectator follows ego (CARLA window) ──────────────────────────────
        ego_tf = vehicle.get_transform()
        fwd    = ego_tf.get_forward_vector()
        spectator.set_transform(carla.Transform(
            carla.Location(
                x=ego_tf.location.x - fwd.x * 10.0,
                y=ego_tf.location.y - fwd.y * 10.0,
                z=ego_tf.location.z + 5.0,
            ),
            carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw),
        ))

        # ── console log ───────────────────────────────────────────────────────
        trk = (f"id={track_id} bbox={bbox} dist={dist_m:.1f}m"
               if visible else "LOST")
        print(f"  step={step:4d}  lin={vx:.3f}  ang={wz:.3f}  {trk}")

        # ── OpenCV display (same window as Hybrid_reID would open) ────────────
        if not args.no_display:
            disp = state.read_display()
            if disp is not None:
                _last_disp = disp
            # Always show last valid frame — prevents blank/flicker between perception updates
            if _last_disp is not None:
                cv2.imshow("Person Follower", _last_disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cleanup()


if __name__ == "__main__":
    main()
