"""
OmniVLA-edge + CARLA + Hybrid_ReID — Correct integration.

Pipeline:
  Camera frame
    → YOLODetector.detect(frame)       [person detections]
    → HybridTracker.update(frame, dets) [DeepSORT + DINOv2 ReID → stable bbox + ID]
    → bbox_to_goal_pose(bbox)           [monocular pose, no GPS]
    → OmniVLA-edge inference            [language + pose → 8-waypoint trajectory]
    → vehicle.apply_control()

Config flags at the top let you swap between:
  USE_TRACKER_POSE = True   → visual tracker (real-world capable)
  USE_TRACKER_POSE = False  → CARLA oracle pose (baseline comparison)
  USE_CROP_GOAL    = True   → feed tracker crop as goal image (modality 9, Phase 2)
  USE_CROP_GOAL    = False  → gray placeholder + modality 8 (default)
"""

import sys, os, time, math, queue, signal, random
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HYBRID_REID_SRC = os.path.expanduser("~/Desktop/Thesis/Hybrid_reID/src")
sys.path.insert(0, HYBRID_REID_SRC)
os.chdir(HYBRID_REID_SRC)  # hybrid_tracker imports relative paths (profiling, utils, etc.)

import numpy as np
import subprocess
import torch
import carla
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# restore cwd so OmniVLA imports work
_OMNIVLA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_OMNIVLA_DIR)
sys.path.insert(0, _OMNIVLA_DIR)

from utils_policy import transform_images_map, load_model, transform_images_PIL_mask

# ── config ────────────────────────────────────────────────────────────────────
CARLA_HOST    = "localhost"
CARLA_PORT    = 2000
TOWN          = "Town01"
TICK_HZ       = 3
MAX_STEPS     = 9999
SAVE_DIR      = "./inference/carla_run_hybrid"
MODEL_PATH    = "./omnivla-edge/omnivla-edge.pth"
LANGUAGE_GOAL = "follow the person ahead"

IMGSIZE       = (96, 96)
IMGSIZE_CLIP  = (224, 224)
METRIC_WAYPOINT_SPACING = 0.1

# ── mode flags ────────────────────────────────────────────────────────────────
USE_TRACKER_POSE = True   # False = CARLA oracle (baseline)
USE_CROP_GOAL    = False  # True  = modality 9, tracker crop as goal image

# ── camera intrinsics (must match cam_bp settings below) ─────────────────────
CAM_W, CAM_H = 400, 300
FOV_DEG      = 90.0
FOCAL_PX     = CAM_W / (2.0 * math.tan(math.radians(FOV_DEG / 2.0)))  # 200 px

PERSON_HEIGHT_M  = 1.7   # average standing adult
MIN_FOLLOW_DIST  = 1.5   # m — brake when closer than this
MAX_FOLLOW_DIST  = 7.0   # m — accelerate when farther than this
LOST_STOP_SECS   = 2.0   # s — emergency stop after this many seconds without detection
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# ── load Hybrid_ReID modules ───────────────────────────────────────────────────
_TRACKER_AVAILABLE = False
YOLODetector = None
HybridTracker = None

try:
    os.chdir(HYBRID_REID_SRC)
    from yolo_detector import YOLODetector
    from hybrid_tracker import HybridTracker
    os.chdir(_OMNIVLA_DIR)
    _TRACKER_AVAILABLE = True
    print("[Hybrid_ReID] YOLODetector + HybridTracker imported OK")
except ImportError as e:
    os.chdir(_OMNIVLA_DIR)
    print(f"[WARNING] Hybrid_ReID import failed: {e}")
    print("[WARNING] Running in ORACLE mode only (USE_TRACKER_POSE forced False)")

use_tracker = USE_TRACKER_POSE and _TRACKER_AVAILABLE


def _scalar(val, default=0):
    """Convert any numeric type (int, float, numpy scalar/array, None) to a Python int."""
    if val is None:
        return default
    try:
        return int(np.asarray(val).flat[0])
    except Exception:
        return default


# ── geometry helpers ───────────────────────────────────────────────────────────
def clip_angle(theta):
    while theta >  math.pi: theta -= 2 * math.pi
    while theta < -math.pi: theta += 2 * math.pi
    return theta


def carla_to_goal_pose(ego_tf, target_location, spacing=METRIC_WAYPOINT_SPACING):
    """CARLA oracle pose: world-space location → OmniVLA goal_pose."""
    fwd   = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    dx = target_location.x - ego_tf.location.x
    dy = target_location.y - ego_tf.location.y
    fwd_d = dx * fwd.x   + dy * fwd.y
    rgt_d = dx * right.x + dy * right.y
    r = math.sqrt(fwd_d**2 + rgt_d**2)
    if r > 8.0:
        fwd_d *= 8.0 / r
        rgt_d *= 8.0 / r
    bear  = math.atan2(dy, dx)
    d_yaw = bear - math.radians(ego_tf.rotation.yaw)
    return np.array([fwd_d / spacing, -rgt_d / spacing,
                     math.cos(d_yaw), math.sin(d_yaw)], dtype=np.float32)


def bbox_to_goal_pose(bbox, spacing=METRIC_WAYPOINT_SPACING):
    """
    Visual pose: bounding box → OmniVLA goal_pose (no GPS).
    Uses pinhole camera model to estimate forward dist and bearing.
    """
    x1, y1, x2, y2 = bbox
    bh  = max(float(y2 - y1), 1.0)
    bcx = (float(x1) + float(x2)) / 2.0
    dist  = float(np.clip((FOCAL_PX * PERSON_HEIGHT_M) / bh, 0.5, 8.0))
    angle = math.atan2(bcx - CAM_W / 2.0, FOCAL_PX)
    fwd = dist * math.cos(angle)
    rgt = dist * math.sin(angle)
    return np.array([fwd / spacing, -rgt / spacing,
                     math.cos(angle), math.sin(angle)], dtype=np.float32)


def bbox_dist_m(bbox):
    bh = max(float(bbox[3]) - float(bbox[1]), 1.0)
    return float(np.clip((FOCAL_PX * PERSON_HEIGHT_M) / bh, 0.3, 20.0))


# ── load OmniVLA-edge ──────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[OmniVLA] Device: {device}")

model_params = {
    "model_type": "omnivla-edge",
    "len_traj_pred": 8,
    "learn_angle": True,
    "context_size": 5,
    "obs_encoder": "efficientnet-b0",
    "encoding_size": 256,
    "obs_encoding_size": 1024,
    "goal_encoding_size": 1024,
    "late_fusion": False,
    "mha_num_attention_heads": 4,
    "mha_num_attention_layers": 4,
    "mha_ff_dim_factor": 4,
    "clip_type": "ViT-B/32",
}

print("[OmniVLA] Loading model …")
model, text_encoder, _ = load_model(MODEL_PATH, model_params, device)
text_encoder = text_encoder.to(device).eval()
model        = model.to(device).eval()
print("[OmniVLA] Model loaded.")

mask_96  = np.ones((96,  96,  3), dtype=np.float32)
mask_224 = np.ones((224, 224, 3), dtype=np.float32)
_gray_goal     = Image.new("RGB", IMGSIZE, color=(128, 128, 128))
goal_image_PIL = _gray_goal
context_buffer = deque(maxlen=6)

# pre-encode language once
_lang_tokens = clip.tokenize(LANGUAGE_GOAL, truncate=True).to(device)
with torch.no_grad():
    _feat_text = text_encoder.encode_text(_lang_tokens).float()


# ── init Hybrid_ReID ──────────────────────────────────────────────────────────
detector = None
tracker  = None
if use_tracker:
    print("[Hybrid_ReID] Initialising YOLODetector …")
    detector = YOLODetector(conf_threshold=0.30,
                            device="cuda" if torch.cuda.is_available() else "cpu")
    print("[Hybrid_ReID] Initialising HybridTracker …")
    tracker = HybridTracker(max_cosine_distance=0.15,
                            min_confidence=0.30,
                            max_age=30)
    print("[Hybrid_ReID] Both models ready.")
else:
    print("[Hybrid_ReID] Skipped (oracle mode).")


# ── OmniVLA inference ──────────────────────────────────────────────────────────
def omnivla_infer(current_pil, goal_pose_np, goal_img_pil, use_crop):
    DT = 1.0 / TICK_HZ
    goal_pose_t = torch.tensor(goal_pose_np, dtype=torch.float32).unsqueeze(0).to(device)

    img96  = current_pil.resize(IMGSIZE)
    img224 = current_pil.resize(IMGSIZE_CLIP)

    context_buffer.append(img96)
    buf = (list(context_buffer) if len(context_buffer) >= 6
           else [img96] * (6 - len(context_buffer)) + list(context_buffer))
    obs_images = transform_images_PIL_mask(buf, mask_96)
    obs_images = torch.split(obs_images.to(device), 3, dim=1)
    obs_image_cur = obs_images[-1].to(device)
    obs_images    = torch.cat(obs_images, dim=1).to(device)
    cur_large     = transform_images_PIL_mask(img224, mask_224).to(device)

    sat = Image.new("RGB", (352, 352), color=(0, 0, 0))
    map_images = torch.cat((transform_images_map(sat).to(device),
                            transform_images_map(sat).to(device),
                            obs_image_cur), dim=1)

    goal_img_t = transform_images_PIL_mask(goal_img_pil.resize(IMGSIZE), mask_96).to(device)
    modality   = torch.tensor([9 if use_crop else 8], dtype=torch.long).to(device)

    with torch.no_grad():
        predicted_actions, _, _ = model(
            obs_images, goal_pose_t, map_images,
            goal_img_t, modality, _feat_text, cur_large,
        )

    waypoints = predicted_actions.float().cpu().numpy()
    chosen    = waypoints[0][4].copy()
    chosen[:2] *= METRIC_WAYPOINT_SPACING
    dx, dy, hx, hy = chosen

    EPS = 1e-8
    if abs(dx) < EPS and abs(dy) < EPS:
        lin = 0.0
        ang = clip_angle(np.arctan2(hy, hx)) / DT
    elif abs(dx) < EPS:
        lin = 0.0
        ang = np.sign(dy) * math.pi / (2 * DT)
    else:
        lin = dx / DT
        ang = math.atan(dy / dx) / DT

    lin = float(np.clip(lin, 0, 0.5))
    ang = float(np.clip(ang, -1.0, 1.0))
    maxv, maxw = 0.3, 0.3

    if abs(lin) <= maxv:
        if abs(ang) <= maxw:
            lin_out, ang_out = lin, ang
        else:
            rd = lin / (ang + 1e-9)
            lin_out = maxw * np.sign(lin) * abs(rd)
            ang_out = maxw * np.sign(ang)
    else:
        if abs(ang) <= 0.001:
            lin_out, ang_out = maxv * np.sign(lin), 0.0
        else:
            rd = lin / ang
            if abs(rd) >= maxv / maxw:
                lin_out = maxv * np.sign(lin)
                ang_out = maxv * np.sign(ang) / abs(rd)
            else:
                lin_out = maxw * np.sign(lin) * abs(rd)
                ang_out = maxw * np.sign(ang)

    return lin_out, ang_out, waypoints[0]


def make_control(lin, ang, dist_m):
    """lin/ang → CARLA VehicleControl with distance-keeping override."""
    if dist_m > 0:
        if dist_m < MIN_FOLLOW_DIST:
            lin = max(0.0, lin - 0.15)
        elif dist_m > MAX_FOLLOW_DIST:
            lin = min(0.3, lin + 0.05)
    throttle = float(np.clip(lin / 0.3 * 0.40, 0.0, 1.0))
    brake    = 0.3 if lin < 0.02 else 0.0
    steer    = float(np.clip(-ang / 0.3, -1.0, 1.0))
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


# ── CARLA setup ───────────────────────────────────────────────────────────────
print("[CARLA] Connecting …")
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(20.0)
world  = client.load_world(TOWN)
print(f"[CARLA] Loaded {TOWN}")

settings = world.get_settings()
settings.synchronous_mode   = True
settings.fixed_delta_seconds = 1.0 / TICK_HZ
world.apply_settings(settings)
world.set_weather(carla.WeatherParameters.ClearNoon)

bp_lib       = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)

# ── pedestrian 8 m ahead ──────────────────────────────────────────────────────
ped_bp    = random.choice(list(bp_lib.filter("walker.pedestrian.*")))
ego_spawn = spawn_points[0]
ego_fwd   = ego_spawn.get_forward_vector()

ped_spawn = carla.Transform(
    carla.Location(x=ego_spawn.location.x + ego_fwd.x * 8.0,
                   y=ego_spawn.location.y + ego_fwd.y * 8.0,
                   z=ego_spawn.location.z + 0.5),
    ego_spawn.rotation,
)
pedestrian = world.try_spawn_actor(ped_bp, ped_spawn)
if pedestrian is None:
    pedestrian = world.spawn_actor(ped_bp,
                                   carla.Transform(world.get_random_location_from_navigation()))

ctrl_bp  = bp_lib.find("controller.ai.walker")
ped_ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=pedestrian)
world.tick()
ped_ctrl.start()
ped_walk_target = carla.Location(x=ego_spawn.location.x + ego_fwd.x * 60.0,
                                  y=ego_spawn.location.y + ego_fwd.y * 60.0,
                                  z=ego_spawn.location.z)
ped_ctrl.go_to_location(ped_walk_target)
ped_ctrl.set_max_speed(0.5)
print(f"[CARLA] Pedestrian [{ped_bp.id}] spawned")

# ── NPC background traffic ────────────────────────────────────────────────────
npcs = []
for sp in spawn_points[1:20]:
    bp = random.choice([b for b in bp_lib.filter("vehicle.*")
                        if b.has_attribute("number_of_wheels")
                        and int(b.get_attribute("number_of_wheels")) == 4])
    npc = world.try_spawn_actor(bp, sp)
    if npc:
        npc.set_autopilot(True, 8000)
        npcs.append(npc)
print(f"[CARLA] Spawned {len(npcs)} NPC cars")

# ── ego vehicle (smallest) ────────────────────────────────────────────────────
_small = ["vehicle.micro.microlino", "vehicle.nissan.micra",
          "vehicle.seat.leon", "vehicle.audi.tt", "vehicle.mini.cooper_s"]
vehicle_bp = None
for _vid in _small:
    try:
        vehicle_bp = bp_lib.find(_vid)
        break
    except IndexError:
        continue
if vehicle_bp is None:
    vehicle_bp = sorted([b for b in bp_lib.filter("vehicle.*")
                         if b.has_attribute("number_of_wheels")
                         and int(b.get_attribute("number_of_wheels")) == 4],
                        key=lambda b: b.id)[0]

vehicle = world.spawn_actor(vehicle_bp, ego_spawn)
print(f"[CARLA] Ego [{vehicle_bp.id}] spawned")

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(CAM_W))
cam_bp.set_attribute("image_size_y", str(CAM_H))
cam_bp.set_attribute("fov", str(FOV_DEG))
camera = world.spawn_actor(cam_bp,
                           carla.Transform(carla.Location(x=1.6, z=1.8)),
                           attach_to=vehicle)

img_queue = queue.Queue()
camera.listen(img_queue.put)

spectator      = world.get_spectator()
actors_spawned = [vehicle, camera, pedestrian, ped_ctrl] + npcs


def cleanup(sig=None, frame=None):
    print("\n[CARLA] Cleaning up …")
    settings.synchronous_mode = False
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(settings)
    try: ped_ctrl.stop()
    except: pass
    for a in actors_spawned:
        try:
            if a.is_alive: a.destroy()
        except: pass
    try: _tk_root.destroy()
    except: pass
    print("[CARLA] Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)


# ── warm-up: fill context buffer + auto-select tracker target ─────────────────
print("[INIT] Warm-up frames …")
first_frame_bgr = None

for wu in range(10):
    world.tick()
    try:
        raw = img_queue.get(timeout=1.0)
    except queue.Empty:
        continue
    arr = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    pil_wu = Image.fromarray(arr[:, :, :3][:, :, ::-1])
    context_buffer.append(pil_wu.resize(IMGSIZE))
    # keep the last warm-up frame for tracker initialisation
    first_frame_bgr = np.array(pil_wu)[:, :, ::-1]

# auto-select the first detected person as the tracking target
if use_tracker and first_frame_bgr is not None:
    print("[Hybrid_ReID] Auto-selecting primary target …")
    init_dets = detector.detect(first_frame_bgr)
    persons   = [d for d in init_dets if len(d) >= 6 and int(d[5]) == 0]
    if persons:
        # pick person with largest bbox area (most prominent in frame)
        best = max(persons, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        tracker.set_primary_object_by_bbox(best[:4])
        print(f"[Hybrid_ReID] Primary target set: bbox={[int(v) for v in best[:4]]}  "
              f"conf={best[4]:.2f}")
    else:
        print("[Hybrid_ReID] No person detected in warm-up frame — will auto-select on first detection.")
        tracker.waiting_for_selection = False   # let tracker auto-assign on first person seen

mode_str = ("VISUAL TRACKER" if use_tracker else "ORACLE") + \
           (" + CROP GOAL" if USE_CROP_GOAL and use_tracker else "")
print(f"\n[RUN] Mode: {mode_str}")
print(f"[RUN] Language goal: \"{LANGUAGE_GOAL}\"\n")


# ── tracking state ────────────────────────────────────────────────────────────
primary_bbox   = None   # last detected/confirmed person bbox
primary_id     = None
last_seen_step = 0    # set to 0; emergency stop only fires after _lost_stop_steps
_last_retarget = 0
_lost_stop_steps = int(LOST_STOP_SECS * TICK_HZ)

# ── Tkinter live viewer window ────────────────────────────────────────────────
_tk_root = tk.Tk()
_tk_root.title("Hybrid_ReID + OmniVLA — live tracker view")
_tk_root.resizable(False, False)
_tk_label = tk.Label(_tk_root)
_tk_label.pack()
_tk_root.update()

# ── main loop ─────────────────────────────────────────────────────────────────
for step in range(MAX_STEPS):
    world.tick()

    try:
        raw = img_queue.get(timeout=2.0)
    except queue.Empty:
        print(f"  step={step:3d}  [WARNING] camera timeout")
        continue

    arr = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    pil = Image.fromarray(arr[:, :, :3][:, :, ::-1])   # BGRA → RGB PIL

    # ── oracle reference (always computed for logging/baseline comparison) ──
    ego_tf     = vehicle.get_transform()
    ped_loc    = pedestrian.get_location()
    oracle_pose = carla_to_goal_pose(ego_tf, ped_loc)
    oracle_dist = math.sqrt((ego_tf.location.x - ped_loc.x)**2 +
                            (ego_tf.location.y - ped_loc.y)**2)

    # ── re-target pedestrian every 30 steps so it keeps walking ─────────────
    if step - _last_retarget >= 30:
        ped_fwd = pedestrian.get_transform().get_forward_vector()
        new_tgt = carla.Location(x=ped_loc.x + ped_fwd.x * 15.0,
                                  y=ped_loc.y + ped_fwd.y * 15.0,
                                  z=ped_loc.z)
        try: ped_ctrl.go_to_location(new_tgt)
        except: pass
        _last_retarget = step

    # ── visual tracker ────────────────────────────────────────────────────────
    tracker_dist_m = -1.0
    goal_pose_np   = oracle_pose   # default fallback
    goal_image_PIL = _gray_goal
    modality_used  = 8

    if use_tracker:
        frame_bgr = np.array(pil)[:, :, ::-1]

        # Step 1: YOLO detection
        detections = detector.detect(frame_bgr)

        # Step 2: DeepSORT + DINOv2 ReID tracking
        tracks = tracker.update(frame_bgr, detections)

        # Step 3: extract primary track (ID == 1, or first person)
        # t[4]=track_id, t[5]=class_id — can be numpy array/scalar/None; use _scalar()
        if step == 0 and tracks:
            print(f"  [DEBUG] track[0] types: {[type(v).__name__ for v in tracks[0]]}")
            print(f"  [DEBUG] track[0] values: {[str(v)[:20] for v in tracks[0]]}")
        person_tracks = [t for t in tracks if len(t) >= 6 and _scalar(t[5]) == 0]

        if person_tracks:
            # prefer primary object (ID 1), else first person
            primary_tracks = [t for t in person_tracks if _scalar(t[4]) == 1]
            chosen = primary_tracks[0] if primary_tracks else person_tracks[0]

            primary_bbox   = [_scalar(chosen[0]), _scalar(chosen[1]),
                               _scalar(chosen[2]), _scalar(chosen[3])]
            primary_id     = _scalar(chosen[4])
            last_seen_step = step
            tracker_dist_m = bbox_dist_m(primary_bbox)
            goal_pose_np   = bbox_to_goal_pose(primary_bbox)

            if USE_CROP_GOAL:
                x1, y1, x2, y2 = [max(0, primary_bbox[0]), max(0, primary_bbox[1]),
                                   min(CAM_W, primary_bbox[2]), min(CAM_H, primary_bbox[3])]
                if x2 > x1 and y2 > y1:
                    goal_image_PIL = pil.crop((x1, y1, x2, y2))
                modality_used = 9

        elif primary_bbox is not None:
            # target temporarily lost — hold last known pose
            goal_pose_np   = bbox_to_goal_pose(primary_bbox)
            tracker_dist_m = bbox_dist_m(primary_bbox)

        else:
            # never seen any person — fall back to oracle
            goal_pose_np = oracle_pose

        # emergency stop if target lost for > LOST_STOP_SECS seconds
        # (only after step > _lost_stop_steps to avoid triggering at startup)
        steps_lost = step - last_seen_step
        if step > _lost_stop_steps and steps_lost > _lost_stop_steps and primary_bbox is None:
            print(f"  step={step:3d}  [LOST {steps_lost} steps — EMERGENCY STOP]")
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            continue

    else:
        goal_pose_np = oracle_pose

    # ── OmniVLA inference ─────────────────────────────────────────────────────
    use_crop = USE_CROP_GOAL and use_tracker and (primary_bbox is not None)
    lin, ang, waypoints = omnivla_infer(pil, goal_pose_np, goal_image_PIL, use_crop)

    # ── apply control ─────────────────────────────────────────────────────────
    ctrl_dist = tracker_dist_m if use_tracker else oracle_dist
    vehicle.apply_control(make_control(lin, ang, ctrl_dist))

    # ── logging ───────────────────────────────────────────────────────────────
    fwd_m = goal_pose_np[0] * METRIC_WAYPOINT_SPACING
    lat_m = goal_pose_np[1] * METRIC_WAYPOINT_SPACING
    trk_str = (f"trk={tracker_dist_m:.1f}m id={primary_id} bbox={primary_bbox}"
               if (use_tracker and primary_bbox is not None) else "trk=NONE")
    print(f"  step={step:3d}  lin={lin:.3f}  ang={ang:.3f}  "
          f"orc={oracle_dist:.1f}m  fwd={fwd_m:+.1f}m  lat={lat_m:+.1f}m  "
          f"mod={modality_used}  {trk_str}")

    # ── save visualization ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    vis = pil.copy()
    if primary_bbox is not None:
        draw = ImageDraw.Draw(vis)
        x1, y1, x2, y2 = primary_bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        lbl_y = max(y1 - 14, 0)
        draw.rectangle([x1, lbl_y, x1 + 52, lbl_y + 14], fill=(0, 255, 0))
        draw.text((x1 + 2, lbl_y), f"ID {primary_id}", fill=(0, 0, 0))

    axes[0].imshow(vis)
    mode_label = f"mod={modality_used}  id={primary_id}"
    axes[0].set_title(f'"{LANGUAGE_GOAL}"  [{mode_label}]', fontsize=9)
    axes[0].axis("off")

    xs = [-w[1] * 0.1 for w in waypoints]
    ys = [ w[0] * 0.1 for w in waypoints]
    axes[1].plot(xs, ys, "b.-", markersize=8, label="OmniVLA path")
    axes[1].plot(0, 0, "go", markersize=10, label="ego")
    gx = -goal_pose_np[1] * METRIC_WAYPOINT_SPACING
    gy =  goal_pose_np[0] * METRIC_WAYPOINT_SPACING
    scale = min(9.0 / (abs(gy) + 1e-3), 1.0)
    axes[1].plot(gx * scale, gy * scale, "r*", markersize=15, label="goal")
    trk_lbl = f"trk={tracker_dist_m:.1f}m" if tracker_dist_m > 0 else "trk=N/A"
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(0, 10)
    axes[1].set_title(f"lin={lin:.2f}  ang={ang:.2f}  {trk_lbl}  orc={oracle_dist:.1f}m",
                      fontsize=9)
    axes[1].set_xlabel("lateral (m)")
    axes[1].set_ylabel("forward (m)")
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(SAVE_DIR, f"step_{step:04d}.jpg"), bbox_inches="tight", dpi=100)
    plt.close(fig)

    # ── live GUI window: tracker overlay (Tkinter, works on X11) ─────────────
    gui_pil = pil.copy()
    draw_gui = ImageDraw.Draw(gui_pil)
    if primary_bbox is not None:
        x1, y1, x2, y2 = primary_bbox
        draw_gui.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        lbl = f"ID:{primary_id}  {tracker_dist_m:.1f}m"
        draw_gui.rectangle([x1, max(y1-18, 0), x1+110, y1], fill=(0, 200, 0))
        draw_gui.text((x1+2, max(y1-16, 0)), lbl, fill=(0, 0, 0))
    else:
        draw_gui.rectangle([0, 0, 120, 22], fill=(200, 0, 0))
        draw_gui.text((4, 4), "NO TARGET", fill=(255, 255, 255))

    status_txt = f"step={step}  orc={oracle_dist:.1f}m  lin={lin:.2f}  ang={ang:.2f}"
    draw_gui.rectangle([0, CAM_H-18, CAM_W, CAM_H], fill=(0, 0, 0))
    draw_gui.text((4, CAM_H-16), status_txt, fill=(255, 220, 0))

    # scale up 2× for easier viewing
    gui_large = gui_pil.resize((CAM_W * 2, CAM_H * 2), Image.NEAREST)
    _tk_img = ImageTk.PhotoImage(gui_large)
    _tk_label.configure(image=_tk_img)
    _tk_label.image = _tk_img
    _tk_root.update()

    # ── spectator chase-cam ───────────────────────────────────────────────────
    fwd = ego_tf.get_forward_vector()
    spectator.set_transform(carla.Transform(
        carla.Location(x=ego_tf.location.x - fwd.x * 10.0,
                       y=ego_tf.location.y - fwd.y * 10.0,
                       z=ego_tf.location.z + 5.0),
        carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw),
    ))

cleanup()
