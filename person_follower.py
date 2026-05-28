"""
person_follower.py — Real-time Person-Following Robot
=====================================================
Three-thread architecture:
  T1 Perception: Camera → YOLOv8 → DeepSORT+DINOv2 → SharedState
  T2 Navigation: SharedState → OmniVLA-edge → ActionBuffer
  T3 Control   : ActionBuffer → send_velocity(linear, angular)

Usage:
    python person_follower.py --video 0
    python person_follower.py --video path/to/video.mp4 --device cpu
    python person_follower.py --video 0 \\
        --checkpoint ~/OmniVLA/omnivla-edge \\
        --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
_HYBRID_SRC   = os.path.expanduser("~/Desktop/Thesis/Hybrid_reID/src")
_OMNIVLA_INF  = os.path.expanduser("~/OmniVLA/inference")
_OMNIVLA_ROOT = os.path.expanduser("~/OmniVLA")

for _p in (_HYBRID_SRC, _OMNIVLA_INF, _OMNIVLA_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


# ══════════════════════════════════════════════════════════════════════════════
# § 1  Thread-safe Shared State
# ══════════════════════════════════════════════════════════════════════════════

class SharedState:
    """All cross-thread data protected by a single lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running:         bool                     = True
        # latest full camera frame (BGR numpy) + PIL version for nav
        self._frame_bgr:      Optional[np.ndarray]     = None
        self._frame_pil:      Optional[Image.Image]    = None
        # tracker outputs
        self._bbox:           Optional[List[int]]      = None   # [x1,y1,x2,y2]
        self._goal_image:     Optional[Image.Image]    = None   # RGB crop of person
        self._track_lost:     bool                     = True
        # control output
        self._velocity:       Tuple[float, float]      = (0.0, 0.0)
        # per-thread FPS (written without lock — fine for display)
        self.fps_perc:        float                    = 0.0
        self.fps_nav:         float                    = 0.0
        self.fps_ctrl:        float                    = 0.0

    # ── writers ───────────────────────────────────────────────────────────────

    def write_frame(self, bgr: np.ndarray) -> None:
        pil = Image.fromarray(bgr[:, :, ::-1])   # BGR→RGB
        with self._lock:
            self._frame_bgr = bgr
            self._frame_pil = pil

    def write_track(self, bbox: List[int], crop: Image.Image) -> None:
        with self._lock:
            self._bbox        = bbox
            self._goal_image  = crop
            self._track_lost  = False

    def write_lost(self) -> None:
        with self._lock:
            self._bbox       = None
            self._goal_image = None
            self._track_lost = True

    def write_velocity(self, linear: float, angular: float) -> None:
        with self._lock:
            self._velocity = (linear, angular)

    # ── readers ───────────────────────────────────────────────────────────────

    def read_frame(self) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
        with self._lock:
            return self._frame_bgr, self._frame_pil

    def read_track(self) -> Tuple[Optional[List[int]], Optional[Image.Image], bool]:
        with self._lock:
            return self._bbox, self._goal_image, self._track_lost

    def read_velocity(self) -> Tuple[float, float]:
        with self._lock:
            return self._velocity


# ══════════════════════════════════════════════════════════════════════════════
# § 2  Action Chunk Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ActionBuffer:
    """Thread-safe FIFO of (linear_vel, angular_vel) pairs."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._buf:  Deque[Tuple[float, float]] = deque()
        self._ready = threading.Event()

    def put_chunk(self, chunk: List[Tuple[float, float]]) -> None:
        with self._lock:
            self._buf.clear()
            self._buf.extend(chunk)
        self._ready.set()

    def get_next(self, timeout: float = 0.1) -> Optional[Tuple[float, float]]:
        self._ready.wait(timeout)
        with self._lock:
            if self._buf:
                item = self._buf.popleft()
                if not self._buf:
                    self._ready.clear()
                return item
        return None

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
        self._ready.clear()


# ══════════════════════════════════════════════════════════════════════════════
# § 3  Robot Interface stub — replace body of send_velocity with real API
# ══════════════════════════════════════════════════════════════════════════════

def send_velocity(linear: float, angular: float) -> None:
    """Send velocity command to robot.  Replace with real robot API."""
    print(f"[Robot] linear={linear:+.3f}  angular={angular:+.3f}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# § 4  Perception Thread (T1)
# ══════════════════════════════════════════════════════════════════════════════

def _init_tracker(model_path: str, yolo_device: str) -> Tuple[object, object]:
    """
    Build YOLODetector + HybridTracker.

    Settings:
      - re_id_interval=999  → DINOv2 only runs on explicit track-loss re-ID,
                               not on every frame — keeps GPU free for OmniVLA
      - gallery_size=500    → lighter than default 3 000
      DINOv2 feature extractor is moved to CPU so OmniVLA can initialise
      its own CUBLAS handles on the GPU without OOM.
    """
    from yolo_detector  import YOLODetector
    from hybrid_tracker import HybridTracker
    import torch

    yolo_device = yolo_device if yolo_device in ("cuda", "cpu") else "cpu"

    # Try TensorRT acceleration; fall back gracefully if unavailable
    try:
        detector = YOLODetector(
            model_path=model_path,
            conf_threshold=0.3,
            device=yolo_device,
            use_tensorrt=True,
        )
        print("[Perception] YOLODetector with TensorRT.")
    except TypeError:
        detector = YOLODetector(
            model_path=model_path,
            conf_threshold=0.3,
            device=yolo_device,
        )
        print("[Perception] YOLODetector (no TensorRT).")

    tracker = HybridTracker(
        max_cosine_distance=0.15,
        nn_budget=1000,
        max_age=30,
        min_confidence=0.3,
        re_id_interval=999,    # disable periodic re-ID; trigger only on loss
        gallery_size=500,
    )

    # DINOv2 → CPU so OmniVLA has the full GPU
    try:
        fe        = tracker.feature_extractor
        fe.model  = fe.model.cpu()
        fe.device = torch.device("cpu")
        print("[Perception] DINOv2 moved to CPU — GPU reserved for OmniVLA.")
    except AttributeError:
        print("[Perception] WARNING: could not move DINOv2 to CPU.")

    return detector, tracker


def _scalar(v) -> int:
    """Convert a numpy scalar-or-array track field to a plain Python int."""
    return int(np.asarray(v).flat[0])


def perception_thread(
    state:      SharedState,
    cap:        cv2.VideoCapture,
    model_path: str,
    device:     str,
) -> None:
    """
    T1 — reads frames from cap, runs YOLOv8 + HybridTracker.

    Writes every frame to SharedState (for display).
    On person detection: writes bbox + RGB crop.
    On loss: sets track_lost flag.
    """
    print("[Perception] Initialising …")
    try:
        detector, tracker = _init_tracker(model_path, device)
        available = True
        print("[Perception] Ready.")
    except Exception as exc:
        print(f"[Perception] Init failed ({exc}) — dummy mode.")
        available = detector = tracker = False

    fps_buf: Deque[float] = deque(maxlen=60)

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        state.write_frame(frame)

        if not available:
            state.write_lost()
            continue

        h, w = frame.shape[:2]

        try:
            detections = detector.detect(frame)
            tracks     = tracker.update(frame, detections)
        except Exception as exc:
            print(f"[Perception] error: {exc}")
            state.write_lost()
            continue

        # filter to persons only (COCO class 0)
        persons = [t for t in tracks if len(t) >= 6 and _scalar(t[5]) == 0]

        if not persons:
            state.write_lost()
        else:
            # primary = lowest (most stable) track ID
            primary = min(persons, key=lambda t: _scalar(t[4]))
            x1 = _scalar(primary[0]); y1 = _scalar(primary[1])
            x2 = _scalar(primary[2]); y2 = _scalar(primary[3])

            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            if cx2 > cx1 and cy2 > cy1:
                crop_bgr = frame[cy1:cy2, cx1:cx2]
                crop_pil = Image.fromarray(crop_bgr[:, :, ::-1])   # RGB
            else:
                crop_pil = Image.new("RGB", (96, 96), (128, 128, 128))

            state.write_track([x1, y1, x2, y2], crop_pil)

        # FPS
        fps_buf.append(time.perf_counter())
        if len(fps_buf) > 1:
            state.fps_perc = (len(fps_buf) - 1) / (fps_buf[-1] - fps_buf[0] + 1e-6)

    print("[Perception] Thread exited.")


# ══════════════════════════════════════════════════════════════════════════════
# § 5  Navigation Thread (T2) — OmniVLA-edge
# ══════════════════════════════════════════════════════════════════════════════

def _clip_angle(theta: float) -> float:
    while theta >  math.pi: theta -= 2 * math.pi
    while theta < -math.pi: theta += 2 * math.pi
    return theta


def _waypoints_to_chunk(
    waypoints: np.ndarray,
    tick_hz:   float = 3.0,
) -> List[Tuple[float, float]]:
    """
    Convert OmniVLA 8-step waypoint array [8, 4] to a list of
    (linear_vel, angular_vel) pairs.

    Applies the same PD control + velocity-coupling constraint
    (maxv = maxw = 0.3) used in carla_omnivla_hybrid.py.
    """
    DT    = 1.0 / tick_hz
    chunk: List[Tuple[float, float]] = []

    for wp in waypoints:
        pt = wp.copy()
        pt[:2] *= 0.1    # denormalise: metric_waypoint_spacing = 0.1 m
        dx, dy, hx, hy = pt

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

        lin = float(np.clip(lin,  0.0,  0.5))
        ang = float(np.clip(ang, -1.0,  1.0))

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

        chunk.append((float(vx), float(wz)))

    return chunk


def _bbox_to_goal_pose(
    bbox:    List[int],
    frame_w: int,
    frame_h: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """
    Monocular goal_pose from bbox.

    Returns float32[4]: [fwd/0.1, -rgt/0.1, cos θ, sin θ]
    matching OmniVLA training convention.
    """
    focal_px = frame_w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    x1, y1, x2, y2 = bbox
    bh   = max(float(y2 - y1), 1.0)
    bcx  = (float(x1) + float(x2)) / 2.0
    dist = float(np.clip((focal_px * 1.7) / bh, 0.5, 8.0))
    ang  = math.atan2(bcx - frame_w / 2.0, focal_px)
    fwd  = dist * math.cos(ang)
    rgt  = dist * math.sin(ang)
    return np.array([fwd / 0.1, -rgt / 0.1, math.cos(ang), math.sin(ang)],
                    dtype=np.float32)


def _load_omnivla(
    checkpoint: str,
    device:     str,
) -> Tuple[object, object, object]:
    """
    Load OmniVLA-edge + CLIP text encoder.

    Tries local checkpoint first; falls back to HuggingFace
    'NHirose/omnivla-original' if not found locally.

    Returns (model, torch.device, feat_text_tensor).
    """
    import torch
    import clip
    from utils_policy import load_model

    EDGE_PARAMS = {
        "model_type":               "omnivla-edge",
        "len_traj_pred":            8,
        "learn_angle":              True,
        "context_size":             5,
        "obs_encoder":              "efficientnet-b0",
        "encoding_size":            256,
        "obs_encoding_size":        1024,
        "goal_encoding_size":       1024,
        "late_fusion":              False,
        "mha_num_attention_heads":  4,
        "mha_num_attention_layers": 4,
        "mha_ff_dim_factor":        4,
        "clip_type":                "ViT-B/32",
    }
    LANG_GOAL = "follow the person ahead"

    # resolve checkpoint path
    ckpt = checkpoint
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "omnivla-edge.pth")
    if not os.path.exists(ckpt):
        print(f"[Navigation] '{ckpt}' not found — downloading from HuggingFace …")
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id="NHirose/omnivla-original",
                                filename="omnivla-edge.pth")

    use_cuda = (device == "cuda") and torch.cuda.is_available()
    dev      = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"[Navigation] Loading OmniVLA-edge on {dev} …")

    try:
        model, text_enc, _ = load_model(ckpt, EDGE_PARAMS, dev)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("[Navigation] CUDA OOM — falling back to CPU.")
            dev   = torch.device("cpu")
            model, text_enc, _ = load_model(ckpt, EDGE_PARAMS, dev)
        else:
            raise

    text_enc = text_enc.to(dev).eval()
    model    = model.to(dev).eval()

    tokens = clip.tokenize(LANG_GOAL, truncate=True).to(dev)
    with torch.no_grad():
        feat_text = text_enc.encode_text(tokens).float()

    print(f"[Navigation] OmniVLA-edge ready.  goal=\"{LANG_GOAL}\"")
    return model, dev, feat_text


def _omnivla_forward(
    model:        object,
    dev:          object,
    feat_text:    object,
    current_pil:  Image.Image,
    goal_img:     Optional[Image.Image],
    goal_pose_np: np.ndarray,
    ctx:          Deque[Image.Image],
    mask96:       np.ndarray,
    mask224:      np.ndarray,
) -> np.ndarray:
    """
    One OmniVLA-edge forward pass.  Returns waypoints [8, 4].

    Matches the tensor construction in carla_omnivla_hybrid.py exactly.
    """
    import torch
    from utils_policy import transform_images_PIL_mask, transform_images_map

    GRAY96  = Image.new("RGB", (96,  96),  (128, 128, 128))
    GRAY224 = Image.new("RGB", (224, 224), (128, 128, 128))

    goal_pose_t = (torch.tensor(goal_pose_np, dtype=torch.float32)
                   .unsqueeze(0).to(dev))

    img96  = current_pil.resize((96,  96))
    img224 = current_pil.resize((224, 224))

    # rolling context buffer → obs tensor [1, 18, 96, 96]
    ctx.append(img96)
    buf = (list(ctx) if len(ctx) >= 6
           else [img96] * (6 - len(ctx)) + list(ctx))

    obs_t    = transform_images_PIL_mask(buf, mask96)
    parts    = torch.split(obs_t.to(dev), 3, dim=1)
    obs_cur  = parts[-1]
    obs_all  = torch.cat(parts, dim=1)

    cur_large = transform_images_PIL_mask(img224, mask224).to(dev)

    # satellite map: black placeholder (no GPS)
    sat   = Image.new("RGB", (352, 352), (0, 0, 0))
    map_t = torch.cat((
        transform_images_map(sat).to(dev),
        transform_images_map(sat).to(dev),
        obs_cur,
    ), dim=1)

    gimg   = goal_img if goal_img is not None else GRAY96
    gimg_t = transform_images_PIL_mask(gimg.resize((96, 96)), mask96).to(dev)

    use_crop = goal_img is not None
    mod_t    = torch.tensor([9 if use_crop else 8], dtype=torch.long).to(dev)

    with torch.no_grad():
        predicted_actions, _, _ = model(
            obs_all, goal_pose_t, map_t,
            gimg_t, mod_t, feat_text, cur_large,
        )

    return predicted_actions.float().cpu().numpy()   # [1, 8, 4]


def navigation_thread(
    state:      SharedState,
    buf:        ActionBuffer,
    checkpoint: str,
    device:     str,
    frame_w:    int,
    frame_h:    int,
) -> None:
    """
    T2 — runs OmniVLA-edge at 3 Hz, pushes 8-step action chunks
    into ActionBuffer.

    If track is lost, pushes an all-zero chunk (stop signal).
    On CUDA OOM, retries on CPU.
    """
    print("[Navigation] Initialising …")
    try:
        model, dev, feat_text = _load_omnivla(checkpoint, device)
        available = True
    except Exception as exc:
        print(f"[Navigation] Init failed ({exc}) — dummy mode.")
        available = False
        import torch as _torch
        dev = _torch.device("cpu")
        feat_text = model = None

    TICK_HZ = 3.0
    DT      = 1.0 / TICK_HZ

    ctx     = deque(maxlen=6)
    mask96  = np.ones((96,  96,  3), dtype=np.float32)
    mask224 = np.ones((224, 224, 3), dtype=np.float32)
    gray96  = Image.new("RGB", (96, 96), (128, 128, 128))

    # seed context buffer with grey frames
    for _ in range(6):
        ctx.append(gray96)

    fps_buf: Deque[float] = deque(maxlen=30)

    while state.running:
        t0 = time.perf_counter()

        bbox, goal_img, track_lost = state.read_track()

        # stop while track is lost
        if track_lost or bbox is None or not available:
            buf.put_chunk([(0.0, 0.0)] * 8)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, DT - elapsed))
            continue

        _, frame_pil = state.read_frame()
        if frame_pil is None:
            time.sleep(0.01)
            continue

        goal_pose_np = _bbox_to_goal_pose(bbox, frame_w, frame_h)

        try:
            waypoints = _omnivla_forward(
                model, dev, feat_text,
                frame_pil, goal_img, goal_pose_np,
                ctx, mask96, mask224,
            )
            chunk = _waypoints_to_chunk(waypoints[0], TICK_HZ)
            buf.put_chunk(chunk)
            if chunk:
                state.write_velocity(*chunk[0])

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                import torch as _torch
                print("[Navigation] CUDA OOM — switching to CPU …")
                dev       = _torch.device("cpu")
                model     = model.cpu()
                feat_text = feat_text.cpu()
            else:
                print(f"[Navigation] forward error: {exc}")
            buf.put_chunk([(0.0, 0.0)] * 8)
        except Exception as exc:
            print(f"[Navigation] error: {exc}")
            buf.put_chunk([(0.0, 0.0)] * 8)

        fps_buf.append(time.perf_counter())
        if len(fps_buf) > 1:
            state.fps_nav = (len(fps_buf) - 1) / (fps_buf[-1] - fps_buf[0] + 1e-6)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, DT - elapsed))

    print("[Navigation] Thread exited.")


# ══════════════════════════════════════════════════════════════════════════════
# § 6  Control Thread (T3)
# ══════════════════════════════════════════════════════════════════════════════

def control_thread(state: SharedState, buf: ActionBuffer) -> None:
    """
    T3 — drains ActionBuffer at ~37 ms/step (3 Hz × 8 steps).

    Sends stop command immediately when track_lost is True.
    """
    STEP_DT = 1.0 / (3.0 * 8)   # ≈ 37 ms
    fps_buf: Deque[float] = deque(maxlen=60)

    print("[Control] Started.")

    while state.running:
        t0 = time.perf_counter()

        _, _, track_lost = state.read_track()
        if track_lost:
            send_velocity(0.0, 0.0)
            state.write_velocity(0.0, 0.0)
            buf.clear()
            time.sleep(STEP_DT)
            continue

        cmd = buf.get_next(timeout=STEP_DT * 2)
        if cmd is not None:
            linear, angular = cmd
            send_velocity(linear, angular)
            state.write_velocity(linear, angular)
        else:
            send_velocity(0.0, 0.0)

        fps_buf.append(time.perf_counter())
        if len(fps_buf) > 1:
            state.fps_ctrl = (len(fps_buf) - 1) / (fps_buf[-1] - fps_buf[0] + 1e-6)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, STEP_DT - elapsed))

    print("[Control] Thread exited.")


# ══════════════════════════════════════════════════════════════════════════════
# § 7  FPS Logger
# ══════════════════════════════════════════════════════════════════════════════

def _fps_logger(state: SharedState, interval: float = 5.0) -> None:
    """Logs per-thread FPS to console every `interval` seconds."""
    while state.running:
        time.sleep(interval)
        if not state.running:
            break
        print(
            f"[FPS]  perc={state.fps_perc:5.1f} Hz  "
            f"nav={state.fps_nav:5.1f} Hz  "
            f"ctrl={state.fps_ctrl:5.1f} Hz",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# § 8  Debug Visualizer (OpenCV overlay)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_debug(frame: np.ndarray, state: SharedState) -> np.ndarray:
    """
    Annotate frame with:
      - bounding box (green = tracking, red = lost)
      - track status and velocity
      - per-thread FPS bar at the bottom
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    bbox, _, track_lost = state.read_track()
    linear, angular     = state.read_velocity()

    # ── bounding box ──────────────────────────────────────────────────────────
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = (0, 220, 0) if not track_lost else (0, 60, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, "ID:1", (x1 + 2, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ── status + velocity (top-left) ──────────────────────────────────────────
    if track_lost:
        status, sc = "LOST",     (0,  60, 255)
    elif bbox is not None:
        status, sc = "TRACKING", (0, 220,   0)
    else:
        status, sc = "SEARCHING",(0, 200, 255)

    cv2.putText(vis, status, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, sc, 2, cv2.LINE_AA)
    cv2.putText(vis, f"lin={linear:+.2f}  ang={angular:+.2f}", (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    # ── per-thread FPS (bottom) ───────────────────────────────────────────────
    fps_text = (f"perc {state.fps_perc:4.1f}Hz  "
                f"nav {state.fps_nav:4.1f}Hz  "
                f"ctrl {state.fps_ctrl:4.1f}Hz")
    cv2.putText(vis, fps_text, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# § 9  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time person-following robot: OmniVLA-edge + Hybrid_reID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",      default="0",
                   help="Camera index (int) or path to video file")
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"],
                   help="Compute device for OmniVLA (YOLO runs on same device)")
    p.add_argument("--model-path", default="yolov8n.pt", dest="model_path",
                   help="YOLO weights (.pt or TensorRT .engine)")
    p.add_argument("--checkpoint", default=os.path.expanduser("~/OmniVLA/omnivla-edge"),
                   help="OmniVLA-edge checkpoint directory or .pth file")
    p.add_argument("--no-display", action="store_true", dest="no_display",
                   help="Disable OpenCV debug window (headless mode)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── open video source ─────────────────────────────────────────────────────
    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Main] Cannot open source: {src}")
        sys.exit(1)

    ret, probe = cap.read()
    if not ret:
        print("[Main] No frames from source.")
        sys.exit(1)
    frame_h, frame_w = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"[Main] Source: {src}  |  Frame: {frame_w}×{frame_h}")

    # resolve YOLO model path (look next to this script if bare filename)
    model_path = args.model_path
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        candidate = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(candidate):
            model_path = candidate

    # ── shared objects ────────────────────────────────────────────────────────
    state = SharedState()
    buf   = ActionBuffer()

    # ── start worker threads ──────────────────────────────────────────────────
    threads = [
        threading.Thread(target=perception_thread,
                         args=(state, cap, model_path, args.device),
                         name="perception", daemon=True),
        threading.Thread(target=navigation_thread,
                         args=(state, buf, args.checkpoint, args.device,
                               frame_w, frame_h),
                         name="navigation", daemon=True),
        threading.Thread(target=control_thread,
                         args=(state, buf),
                         name="control", daemon=True),
        threading.Thread(target=_fps_logger,
                         args=(state,),
                         name="fps_logger", daemon=True),
    ]
    for t in threads:
        t.start()

    print("[Main] All threads started.  Press 'q' / Escape to quit.\n")

    if not args.no_display:
        cv2.namedWindow("Person Follower", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Person Follower", 800, 600)

    try:
        while state.running:
            frame_bgr, _ = state.read_frame()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            if not args.no_display:
                vis = _draw_debug(frame_bgr, state)
                cv2.imshow("Person Follower", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl-C received.")
    finally:
        print("[Main] Shutting down …")
        state.running = False
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        # give daemon threads a moment to print their exit messages
        time.sleep(0.5)
        print("[Main] Done.")


if __name__ == "__main__":
    main()
