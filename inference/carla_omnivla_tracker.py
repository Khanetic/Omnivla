"""
OmniVLA-edge + CARLA + Hybrid_ReID — Visual Person Following.

Replaces the CARLA oracle (pedestrian.get_location()) with a purely visual
tracking pipeline:
  Camera frame → HybridTracker (YOLOv8 + DeepSORT + DINOv2 ReID)
              → primary bbox → bbox_to_goal_pose()
              → OmniVLA goal_pose (no GPS/world coords)

The CARLA pedestrian is still spawned to give us something to follow, but
the ego vehicle navigates using only the camera image — just as a real robot would.

Flags at the top of this file let you compare modes:
  USE_TRACKER_POSE : True  → visual bbox pose (real-world mode)
                     False → CARLA oracle pose (baseline)
  USE_CROP_GOAL    : True  → feed 96×96 tracker crop as goal_image (modality 9)
                     False → use gray placeholder + modality 8 (original mode)
"""

import sys, os, time, math, queue, signal, random
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser("~/Desktop/Thesis/Hybrid_reID/src"))

import numpy as np
import torch
import carla
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from utils_policy import transform_images_map, load_model, transform_images_PIL, transform_images_PIL_mask

try:
    from hybrid_tracker import HybridTracker
    _TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] HybridTracker import failed: {e}")
    print("[WARNING] Falling back to oracle pose (USE_TRACKER_POSE forced to False)")
    _TRACKER_AVAILABLE = False

# ── config ────────────────────────────────────────────────────────────────────
CARLA_HOST   = "localhost"
CARLA_PORT   = 2000
TOWN         = "Town01"
TICK_HZ      = 3
MAX_STEPS    = 9999
SAVE_DIR     = "./inference/carla_run_tracker"
MODEL_PATH   = "./omnivla-edge/omnivla-edge.pth"

LANGUAGE_GOAL = "follow the person ahead"

IMGSIZE      = (96, 96)
IMGSIZE_CLIP = (224, 224)

METRIC_WAYPOINT_SPACING = 0.1

# ── tracker / goal mode ───────────────────────────────────────────────────────
USE_TRACKER_POSE = True   # True = visual bbox pose; False = CARLA oracle
USE_CROP_GOAL    = False  # True = feed tracker crop as goal image (modality 9)

# camera intrinsics (must match cam_bp attributes below)
CAM_W, CAM_H = 400, 300
FOV_DEG      = 90.0
FOCAL_PX     = CAM_W / (2.0 * math.tan(math.radians(FOV_DEG / 2.0)))  # = 200 px

# physical target height for monocular distance estimation
PERSON_HEIGHT_M = 1.7   # average adult standing height

# following safety distances
MIN_FOLLOW_DIST = 1.5   # m — brake if closer than this
MAX_FOLLOW_DIST = 7.0   # m — accelerate if farther than this
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)


def clip_angle(theta):
    while theta > math.pi:  theta -= 2 * math.pi
    while theta < -math.pi: theta += 2 * math.pi
    return theta


def carla_to_goal_pose(ego_tf, target_location, metric_spacing=0.1):
    """CARLA oracle: convert world-space location to OmniVLA goal_pose."""
    fwd   = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    dx = target_location.x - ego_tf.location.x
    dy = target_location.y - ego_tf.location.y

    forward_dist = dx * fwd.x + dy * fwd.y
    right_dist   = dx * right.x + dy * right.y

    radius = math.sqrt(forward_dist**2 + right_dist**2)
    if radius > 8.0:
        scale = 8.0 / radius
        forward_dist *= scale
        right_dist   *= scale

    bear  = math.atan2(dy, dx)
    d_yaw = bear - math.radians(ego_tf.rotation.yaw)

    return np.array([
        forward_dist / metric_spacing,
        -right_dist  / metric_spacing,
        math.cos(d_yaw),
        math.sin(d_yaw),
    ], dtype=np.float32)


def bbox_to_goal_pose(bbox, metric_spacing=0.1):
    """
    Visual pose: convert tracker bounding box to OmniVLA goal_pose format.
    Uses pinhole camera geometry (no GPS or world coordinates).

    goal_pose[0] = forward distance / metric_spacing
    goal_pose[1] = left distance  / metric_spacing  (positive = left of ego)
    goal_pose[2] = cos(bearing angle)
    goal_pose[3] = sin(bearing angle)
    """
    x1, y1, x2, y2 = bbox
    bh  = max(float(y2 - y1), 1.0)
    bcx = (float(x1) + float(x2)) / 2.0

    # monocular distance from bbox height
    dist = float(np.clip((FOCAL_PX * PERSON_HEIGHT_M) / bh, 0.5, 8.0))

    # horizontal bearing (positive angle = target to the right of image centre)
    angle = math.atan2(bcx - CAM_W / 2.0, FOCAL_PX)

    fwd = dist * math.cos(angle)
    rgt = dist * math.sin(angle)   # positive = right

    return np.array([
        fwd / metric_spacing,
        -rgt / metric_spacing,   # flip: OmniVLA uses left-positive
        math.cos(angle),
        math.sin(angle),
    ], dtype=np.float32)


def bbox_distance_m(bbox):
    """Estimate target distance in metres from bbox height."""
    bh = max(float(bbox[3]) - float(bbox[1]), 1.0)
    return float(np.clip((FOCAL_PX * PERSON_HEIGHT_M) / bh, 0.3, 20.0))


# ── load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

print("Loading OmniVLA-edge …")
model, text_encoder, preprocess = load_model(MODEL_PATH, model_params, device)
text_encoder = text_encoder.to(device).eval()
model        = model.to(device).eval()
print("Model loaded.")

mask_96  = np.ones((96,  96,  3), dtype=np.float32)
mask_224 = np.ones((224, 224, 3), dtype=np.float32)

_gray_goal    = Image.new("RGB", IMGSIZE, color=(128, 128, 128))
goal_image_PIL = _gray_goal   # updated in main loop when tracker has a crop
context_buffer = deque(maxlen=6)

# pre-encode language goal once (saves ~5ms per step)
_lang_tokens    = clip.tokenize(LANGUAGE_GOAL, truncate=True).to(device)
with torch.no_grad():
    _feat_text_lan  = text_encoder.encode_text(_lang_tokens).float()


def omnivla_infer(current_pil, goal_pose_np, goal_img_pil, use_crop, step):
    """Run one OmniVLA forward pass.

    Args:
        current_pil : PIL.Image — current camera frame (full resolution)
        goal_pose_np: np.float32[4] — [fwd/0.1, -right/0.1, cosθ, sinθ]
        goal_img_pil: PIL.Image — 96×96 goal image (tracker crop or gray)
        use_crop    : bool — True → modality 9 (lang+image), False → modality 8 (lang+pose)
        step        : int
    Returns:
        (lin_out, ang_out) : float — linear/angular velocity commands
        waypoints          : np.ndarray [8, 4]
    """
    DT = 1.0 / TICK_HZ

    goal_pose_torch = torch.tensor(goal_pose_np, dtype=torch.float32).unsqueeze(0).to(device)

    img96  = current_pil.resize(IMGSIZE)
    img224 = current_pil.resize(IMGSIZE_CLIP)

    context_buffer.append(img96)
    context_queue = (list(context_buffer) if len(context_buffer) >= 6
                     else [img96] * (6 - len(context_buffer)) + list(context_buffer))
    obs_images = transform_images_PIL_mask(context_queue, mask_96)
    obs_images = torch.split(obs_images.to(device), 3, dim=1)
    obs_image_cur = obs_images[-1].to(device)
    obs_images    = torch.cat(obs_images, dim=1).to(device)

    cur_large_img = transform_images_PIL_mask(img224, mask_224).to(device)

    sat = Image.new("RGB", (352, 352), color=(0, 0, 0))
    map_images = torch.cat((
        transform_images_map(sat).to(device),
        transform_images_map(sat).to(device),
        obs_image_cur,
    ), dim=1)

    goal_image_t = transform_images_PIL_mask(goal_img_pil.resize(IMGSIZE), mask_96).to(device)

    # modality 9 = language + goal image; modality 8 = language + pose
    modality_id = 9 if use_crop else 8
    modality_id_select = torch.tensor([modality_id], dtype=torch.long).to(device)

    with torch.no_grad():
        predicted_actions, _, _ = model(
            obs_images,
            goal_pose_torch,
            map_images,
            goal_image_t,
            modality_id_select,
            _feat_text_lan,
            cur_large_img,
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
        ang = np.sign(dy) * np.pi / (2 * DT)
    else:
        lin = dx / DT
        ang = np.arctan(dy / dx) / DT

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


def _draw_bbox(pil_img, bbox, track_id, color=(0, 255, 0)):
    """Return a copy of pil_img with a green bbox + ID label drawn on it."""
    vis  = pil_img.copy()
    draw = ImageDraw.Draw(vis)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    label_y = max(y1 - 14, 0)
    draw.rectangle([x1, label_y, x1 + 50, label_y + 14], fill=color)
    draw.text((x1 + 2, label_y), f"ID {track_id}", fill=(0, 0, 0))
    return vis


def _save_vis(current_pil, waypoints, lin, ang, step, goal_pose_np,
              primary_bbox, primary_id, tracker_dist_m, oracle_dist_m, modality):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # left panel: annotated camera frame
    if primary_bbox is not None:
        vis_img = _draw_bbox(current_pil, primary_bbox, primary_id)
    else:
        vis_img = current_pil
    axes[0].imshow(vis_img)
    mode_str = f"mod={modality}  id={primary_id}"
    axes[0].set_title(f'"{LANGUAGE_GOAL}"  [{mode_str}]', fontsize=9)
    axes[0].axis("off")

    # right panel: waypoint trajectory + goal marker
    xs = [-w[1] * 0.1 for w in waypoints]
    ys = [ w[0] * 0.1 for w in waypoints]
    axes[1].plot(xs, ys, "b.-", markersize=8, label="predicted path")
    axes[1].plot(0, 0, "go", markersize=10, label="ego")

    gx = -goal_pose_np[1] * METRIC_WAYPOINT_SPACING
    gy =  goal_pose_np[0] * METRIC_WAYPOINT_SPACING
    scale = min(9.0 / (abs(gy) + 1e-3), 1.0)
    axes[1].plot(gx * scale, gy * scale, "r*", markersize=15, label="target goal")

    dist_label = (f"trk={tracker_dist_m:.1f}m"
                  if tracker_dist_m > 0 else "trk=N/A")
    oracle_label = f"orc={oracle_dist_m:.1f}m" if oracle_dist_m >= 0 else ""
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(0, 10)
    axes[1].set_title(f"lin={lin:.2f}  ang={ang:.2f}  {dist_label}  {oracle_label}", fontsize=9)
    axes[1].set_xlabel("lateral (m)"); axes[1].set_ylabel("forward (m)")
    axes[1].legend(fontsize=8)

    fig.savefig(os.path.join(SAVE_DIR, f"step_{step:04d}.jpg"), bbox_inches="tight", dpi=100)
    plt.close(fig)


def linear_angular_to_carla(lin, ang, tracker_dist_m):
    """Convert (lin, ang) to CARLA VehicleControl with distance-keeping override."""
    # safety: slow down if too close, force forward if too far
    if tracker_dist_m > 0:
        if tracker_dist_m < MIN_FOLLOW_DIST:
            lin = max(0.0, lin - 0.15)
        elif tracker_dist_m > MAX_FOLLOW_DIST:
            lin = min(0.3, lin + 0.05)

    throttle = float(np.clip(lin / 0.3 * 0.40, 0.0, 1.0))
    brake    = 0.3 if lin < 0.02 else 0.0
    steer    = float(np.clip(-ang / 0.3, -1.0, 1.0))
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


# ── CARLA setup ───────────────────────────────────────────────────────────────
print("Connecting to CARLA …")
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(20.0)
world  = client.load_world(TOWN)
print(f"Loaded {TOWN}")

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / TICK_HZ
world.apply_settings(settings)

world.set_weather(carla.WeatherParameters.ClearNoon)

bp_lib       = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)

# ── spawn target PEDESTRIAN 8 m ahead of ego ──────────────────────────────────
ped_bps   = list(bp_lib.filter("walker.pedestrian.*"))
ped_bp    = random.choice(ped_bps)

ego_spawn = spawn_points[0]
ego_fwd   = ego_spawn.get_forward_vector()
ped_spawn = carla.Transform(
    carla.Location(
        x=ego_spawn.location.x + ego_fwd.x * 8.0,
        y=ego_spawn.location.y + ego_fwd.y * 8.0,
        z=ego_spawn.location.z + 0.5,
    ),
    ego_spawn.rotation,
)
pedestrian = world.try_spawn_actor(ped_bp, ped_spawn)
if pedestrian is None:
    fb_loc = world.get_random_location_from_navigation()
    pedestrian = world.spawn_actor(ped_bp, carla.Transform(fb_loc))

ctrl_bp  = bp_lib.find("controller.ai.walker")
ped_ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=pedestrian)
world.tick()
ped_ctrl.start()
ped_walk_target = carla.Location(
    x=ego_spawn.location.x + ego_fwd.x * 60.0,
    y=ego_spawn.location.y + ego_fwd.y * 60.0,
    z=ego_spawn.location.z,
)
ped_ctrl.go_to_location(ped_walk_target)
ped_ctrl.set_max_speed(0.5)
print(f"Spawned pedestrian [{ped_bp.id}] at {pedestrian.get_location()}")

# ── spawn background NPC cars ─────────────────────────────────────────────────
npcs = []
for sp in spawn_points[1:20]:
    bp = random.choice([b for b in bp_lib.filter("vehicle.*")
                        if b.has_attribute("number_of_wheels")
                        and int(b.get_attribute("number_of_wheels")) == 4])
    npc = world.try_spawn_actor(bp, sp)
    if npc:
        npc.set_autopilot(True, 8000)
        npcs.append(npc)
print(f"Spawned {len(npcs)} NPC cars")

# ── ego vehicle (smallest available) ─────────────────────────────────────────
_small_ids = ["vehicle.micro.microlino", "vehicle.nissan.micra", "vehicle.seat.leon",
              "vehicle.audi.tt", "vehicle.mini.cooper_s"]
vehicle_bp = None
for _vid in _small_ids:
    try:
        vehicle_bp = bp_lib.find(_vid)
        break
    except IndexError:
        continue
if vehicle_bp is None:
    _all4 = [b for b in bp_lib.filter("vehicle.*")
             if b.has_attribute("number_of_wheels")
             and int(b.get_attribute("number_of_wheels")) == 4]
    vehicle_bp = sorted(_all4, key=lambda b: b.id)[0]

vehicle = world.spawn_actor(vehicle_bp, ego_spawn)
print(f"Spawned ego [{vehicle_bp.id}] at {ego_spawn.location}")

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(CAM_W))
cam_bp.set_attribute("image_size_y", str(CAM_H))
cam_bp.set_attribute("fov", str(FOV_DEG))
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.8)), attach_to=vehicle)

img_queue = queue.Queue()
camera.listen(img_queue.put)

spectator      = world.get_spectator()
actors_spawned = [vehicle, camera, pedestrian, ped_ctrl] + npcs


def cleanup(sig=None, frame=None):
    print("\nCleaning up …")
    settings.synchronous_mode = False
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(settings)
    try: ped_ctrl.stop()
    except: pass
    for a in actors_spawned:
        try:
            if a.is_alive: a.destroy()
        except Exception:
            pass
    print("Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

# ── init tracker ──────────────────────────────────────────────────────────────
tracker      = None
use_tracker  = USE_TRACKER_POSE and _TRACKER_AVAILABLE
if use_tracker:
    print("Initialising HybridTracker …")
    tracker = HybridTracker(
        max_cosine_distance=0.15,
        min_confidence=0.3,
        max_age=30,
    )
    print("HybridTracker ready.")
else:
    print("Running in ORACLE mode (no visual tracker).")

primary_bbox     = None
primary_id       = None
last_seen_step   = -999   # step when target was last detected

# ── warm up ───────────────────────────────────────────────────────────────────
for _ in range(10):
    world.tick()
    try:
        raw = img_queue.get(timeout=1.0)
        arr = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(raw.height, raw.width, 4)
        pil_wu = Image.fromarray(arr[:, :, :3][:, :, ::-1])
        context_buffer.append(pil_wu.resize(IMGSIZE))
    except:
        pass

mode_desc = ("VISUAL TRACKER" if use_tracker else "ORACLE") + (
    " + CROP GOAL" if USE_CROP_GOAL and use_tracker else "")
print(f"\nRunning — mode: {mode_desc}")
print(f"Language goal: \"{LANGUAGE_GOAL}\"\n")

# ── main loop ─────────────────────────────────────────────────────────────────
_last_retarget = 0

for step in range(MAX_STEPS):
    world.tick()

    try:
        raw = img_queue.get(timeout=2.0)
    except queue.Empty:
        continue

    arr = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    pil = Image.fromarray(arr[:, :, :3][:, :, ::-1])   # BGRA → RGB PIL

    # ── oracle reference (always computed for logging/comparison) ──────────────
    ego_tf      = vehicle.get_transform()
    ped_loc     = pedestrian.get_location()
    oracle_pose = carla_to_goal_pose(ego_tf, ped_loc)
    oracle_dist = math.sqrt((ego_tf.location.x - ped_loc.x)**2 +
                            (ego_tf.location.y - ped_loc.y)**2)

    # ── re-target pedestrian AI every 30 steps so it keeps walking ahead ───────
    if step - _last_retarget >= 30:
        ped_fwd = pedestrian.get_transform().get_forward_vector()
        new_target = carla.Location(
            x=ped_loc.x + ped_fwd.x * 15.0,
            y=ped_loc.y + ped_fwd.y * 15.0,
            z=ped_loc.z,
        )
        try:
            ped_ctrl.go_to_location(new_target)
        except Exception:
            pass
        _last_retarget = step

    # ── visual tracker ─────────────────────────────────────────────────────────
    tracker_dist_m = -1.0   # -1 = not detected this frame
    goal_pose_np   = oracle_pose   # default: oracle fallback

    if use_tracker:
        frame_bgr = np.array(pil)[:, :, ::-1]  # PIL RGB → BGR for OpenCV/YOLO
        tracks    = tracker.update(frame_bgr)

        # filter to person class (class_id == 0 in COCO)
        persons = [t for t in tracks if len(t) >= 6 and int(t[5]) == 0]

        if persons:
            # primary = track with lowest / most stable ID
            primary = min(persons, key=lambda t: t[4])
            primary_bbox = [int(primary[0]), int(primary[1]),
                            int(primary[2]), int(primary[3])]
            primary_id   = int(primary[4])
            last_seen_step = step
            tracker_dist_m = bbox_distance_m(primary_bbox)

            # visual pose from bbox
            goal_pose_np = bbox_to_goal_pose(primary_bbox)

            # optionally update goal image with tracker crop (Phase 2)
            if USE_CROP_GOAL:
                x1, y1, x2, y2 = primary_bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(CAM_W, x2), min(CAM_H, y2)
                if x2 > x1 and y2 > y1:
                    goal_image_PIL = pil.crop((x1, y1, x2, y2))
                else:
                    goal_image_PIL = _gray_goal
            else:
                goal_image_PIL = _gray_goal

        elif primary_bbox is not None:
            # target temporarily lost — hold last known bbox pose
            goal_pose_np   = bbox_to_goal_pose(primary_bbox)
            tracker_dist_m = bbox_distance_m(primary_bbox)
            goal_image_PIL = _gray_goal
        else:
            # never seen — fall back to oracle
            goal_pose_np   = oracle_pose
            goal_image_PIL = _gray_goal

        # emergency stop if lost for more than 2 s
        steps_lost = step - last_seen_step
        if steps_lost > int(2.0 * TICK_HZ) and primary_bbox is None:
            print(f"  step={step:3d}  [LOST TARGET for {steps_lost} steps — STOPPING]")
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            continue

    else:
        # oracle mode
        goal_pose_np   = oracle_pose
        goal_image_PIL = _gray_goal

    # ── OmniVLA inference ──────────────────────────────────────────────────────
    use_crop = USE_CROP_GOAL and use_tracker and (primary_bbox is not None)
    lin, ang, waypoints = omnivla_infer(pil, goal_pose_np, goal_image_PIL, use_crop, step)

    # ── safety distance override ───────────────────────────────────────────────
    ctrl = linear_angular_to_carla(lin, ang, tracker_dist_m if use_tracker else oracle_dist)
    vehicle.apply_control(ctrl)

    # ── logging ────────────────────────────────────────────────────────────────
    fwd_m = goal_pose_np[0] * METRIC_WAYPOINT_SPACING
    lat_m = goal_pose_np[1] * METRIC_WAYPOINT_SPACING
    modality = 9 if use_crop else 8
    trk_str  = f"trk_dist={tracker_dist_m:.1f}m  id={primary_id}  bbox={primary_bbox}" \
               if use_tracker and primary_bbox is not None else "trk=LOST"
    print(f"  step={step:3d}  lin={lin:.3f}  ang={ang:.3f}  "
          f"orc={oracle_dist:.1f}m  fwd={fwd_m:+.1f}m  lat={lat_m:+.1f}m  "
          f"mod={modality}  {trk_str}")

    # ── visualization ──────────────────────────────────────────────────────────
    _save_vis(pil, waypoints, lin, ang, step, goal_pose_np,
              primary_bbox, primary_id, tracker_dist_m, oracle_dist, modality)

    # ── spectator chase-cam ────────────────────────────────────────────────────
    fwd = ego_tf.get_forward_vector()
    spectator.set_transform(carla.Transform(
        carla.Location(
            x=ego_tf.location.x - fwd.x * 10.0,
            y=ego_tf.location.y - fwd.y * 10.0,
            z=ego_tf.location.z + 5.0,
        ),
        carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw)
    ))

cleanup()
