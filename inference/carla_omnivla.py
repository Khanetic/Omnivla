"""
OmniVLA-edge + CARLA bridge.
Spawns a Tesla ego vehicle + a target bicycle in Town01.
OmniVLA uses the bicycle's live position as the goal pose each step
(pose-conditioned following) combined with the language prompt.
"""

import sys, os, time, math, queue, signal
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import carla
import utm
import clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utils_policy import transform_images_map, load_model, transform_images_PIL, transform_images_PIL_mask

# ── config ────────────────────────────────────────────────────────────────────
CARLA_HOST   = "localhost"
CARLA_PORT   = 2000
TOWN         = "Town01"
TICK_HZ      = 3
MAX_STEPS    = 9999       # run until Ctrl+C
SAVE_DIR     = "./inference/carla_run"
MODEL_PATH   = "./omnivla-edge/omnivla-edge.pth"

LANGUAGE_GOAL = "move toward the bike ahead"   # ← language prompt

IMGSIZE      = (96, 96)
IMGSIZE_CLIP = (224, 224)

METRIC_WAYPOINT_SPACING = 0.1
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

def clip_angle(theta):
    while theta > math.pi:  theta -= 2 * math.pi
    while theta < -math.pi: theta += 2 * math.pi
    return theta


def carla_to_goal_pose(ego_tf, target_location, metric_spacing=0.1):
    """
    Convert a CARLA world-space target location into OmniVLA goal_pose format:
      [forward/spacing, left/spacing, cos(dYaw), sin(dYaw)]
    goal_pose[0] = FORWARD distance (positive = ahead of ego)
    goal_pose[1] = LEFT distance (positive = to the left of ego)
    Matches run_omnivla_edge.py: relative_y=forward, -relative_x=left.
    """
    fwd   = ego_tf.get_forward_vector()   # unit vector pointing ahead
    right = ego_tf.get_right_vector()     # unit vector pointing right

    dx = target_location.x - ego_tf.location.x
    dy = target_location.y - ego_tf.location.y

    # project world displacement onto ego-local axes
    forward_dist = dx * fwd.x   + dy * fwd.y    # positive = ahead
    right_dist   = dx * right.x + dy * right.y  # positive = to the right

    # clamp to 8 m — keeps goal_pose within training distribution
    radius = math.sqrt(forward_dist**2 + right_dist**2)
    if radius > 8.0:
        scale = 8.0 / radius
        forward_dist *= scale
        right_dist   *= scale

    # bearing from ego to target
    bear  = math.atan2(dy, dx)
    d_yaw = bear - math.radians(ego_tf.rotation.yaw)

    return np.array([
        forward_dist / metric_spacing,   # goal_pose[0] = FORWARD
        -right_dist  / metric_spacing,   # goal_pose[1] = LEFT (flip right→left)
        math.cos(d_yaw),
        math.sin(d_yaw),
    ], dtype=np.float32)


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

print(f"Loading OmniVLA-edge …")
model, text_encoder, preprocess = load_model(MODEL_PATH, model_params, device)
text_encoder = text_encoder.to(device).eval()
model        = model.to(device).eval()
print("Model loaded.")

mask_96  = np.ones((96,  96,  3), dtype=np.float32)
mask_224 = np.ones((224, 224, 3), dtype=np.float32)
goal_image_PIL = Image.new("RGB", IMGSIZE, color=(128, 128, 128))
context_buffer = deque(maxlen=6)   # rolling buffer of last 6 frames


def omnivla_infer(current_pil, goal_pose_np, step, real_dist_m):
    """Run one forward pass. goal_pose_np is a (4,) array from carla_to_goal_pose."""
    DT = 1.0 / TICK_HZ

    goal_pose_torch = torch.tensor(goal_pose_np, dtype=torch.float32).unsqueeze(0).to(device)

    img96  = current_pil.resize(IMGSIZE)
    img224 = current_pil.resize(IMGSIZE_CLIP)

    context_buffer.append(img96)
    context_queue = list(context_buffer) if len(context_buffer) >= 6 else [img96] * (6 - len(context_buffer)) + list(context_buffer)
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

    goal_image_t  = transform_images_PIL_mask(goal_image_PIL, mask_96).to(device)
    obj_inst_lan  = clip.tokenize(LANGUAGE_GOAL, truncate=True).to(device)
    feat_text_lan = text_encoder.encode_text(obj_inst_lan).float()

    # modality 8 = pose + language (from run_omnivla_edge.py mapping)
    modality_id_select = torch.tensor([8], dtype=torch.long).to(device)

    with torch.no_grad():
        predicted_actions, _, _ = model(
            obs_images,
            goal_pose_torch,
            map_images,
            goal_image_t,
            modality_id_select,
            feat_text_lan,
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

    _save_vis(current_pil, waypoints[0], lin_out, ang_out, step, goal_pose_np)
    print(f"  step={step:3d}  lin={lin_out:.3f}  ang={ang_out:.3f}  bike_dist={real_dist_m:.1f}m")
    return lin_out, ang_out


def _save_vis(current_pil, waypoints, lin, ang, step, goal_pose_np):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(current_pil)
    axes[0].set_title(f'CARLA front camera  —  "{LANGUAGE_GOAL}"')
    axes[0].axis("off")

    xs = [-w[1] * 0.1 for w in waypoints]
    ys = [ w[0] * 0.1 for w in waypoints]
    axes[1].plot(xs, ys, "b.-", markersize=8, label="predicted path")
    axes[1].plot(0, 0, "go", markersize=10, label="ego")

    # mark bike goal direction
    gx = -goal_pose_np[1] * METRIC_WAYPOINT_SPACING
    gy =  goal_pose_np[0] * METRIC_WAYPOINT_SPACING
    scale = min(9.0 / (abs(gy) + 1e-3), 1.0)
    axes[1].plot(gx * scale, gy * scale, "r*", markersize=15, label="bike goal")

    axes[1].set_xlim(-3, 3); axes[1].set_ylim(0, 10)
    axes[1].set_title(f"lin={lin:.2f}  ang={ang:.2f}")
    axes[1].set_xlabel("lateral (m)"); axes[1].set_ylabel("forward (m)")
    axes[1].legend(fontsize=8)

    fig.savefig(os.path.join(SAVE_DIR, f"step_{step:04d}.jpg"), bbox_inches="tight", dpi=100)
    plt.close(fig)


def linear_angular_to_carla(lin, ang):
    # Scale up so the Tesla can keep pace with the slow bike
    throttle = float(np.clip(lin / 0.3 * 0.80, 0.0, 1.0))
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

bp_lib      = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)

# ── spawn target BICYCLE 15 m ahead of ego ────────────────────────────────────
bike_bps = [bp for bp in bp_lib.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 2]
print(f"Available 2-wheel blueprints: {[b.id for b in bike_bps]}")

ego_spawn  = spawn_points[0]
ego_fwd    = ego_spawn.get_forward_vector()
bike_spawn = carla.Transform(
    carla.Location(
        x=ego_spawn.location.x + ego_fwd.x * 15.0,
        y=ego_spawn.location.y + ego_fwd.y * 15.0,
        z=ego_spawn.location.z + 0.3,
    ),
    ego_spawn.rotation,
)
bike_bp = bike_bps[0] if bike_bps else bp_lib.find("vehicle.bh.crossbike")
bike = world.try_spawn_actor(bike_bp, bike_spawn)
if bike is None:
    bike = world.spawn_actor(bike_bp, spawn_points[5])
bike.set_autopilot(True, 8000)
traffic_manager.vehicle_percentage_speed_difference(bike, 80)   # bike goes 80% slower than speed limit
print(f"Spawned bike [{bike_bp.id}] at {bike.get_location()}")

# ── spawn background NPC cars ─────────────────────────────────────────────────
npcs = []
for sp in spawn_points[1:20]:
    bp = np.random.choice([b for b in bp_lib.filter("vehicle.*")
                           if b.has_attribute("number_of_wheels")
                           and int(b.get_attribute("number_of_wheels")) == 4])
    npc = world.try_spawn_actor(bp, sp)
    if npc:
        npc.set_autopilot(True, 8000)
        npcs.append(npc)
print(f"Spawned {len(npcs)} NPC cars")

# ── ego vehicle ───────────────────────────────────────────────────────────────
vehicle_bp = bp_lib.find("vehicle.tesla.model3")
vehicle    = world.spawn_actor(vehicle_bp, ego_spawn)
print(f"Spawned ego Tesla at {ego_spawn.location}")

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "400")
cam_bp.set_attribute("image_size_y", "300")
cam_bp.set_attribute("fov", "90")
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.8)), attach_to=vehicle)

img_queue = queue.Queue()
camera.listen(img_queue.put)

spectator    = world.get_spectator()
actors_spawned = [vehicle, camera, bike] + npcs


def cleanup(sig=None, frame=None):
    print("\nCleaning up …")
    settings.synchronous_mode = False
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(settings)
    for a in actors_spawned:
        try:
            if a.is_alive: a.destroy()
        except Exception:
            pass
    print("Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

# warm up
for _ in range(10):
    world.tick()
    try: img_queue.get(timeout=1.0)
    except: pass

print(f"\nRunning — language goal: \"{LANGUAGE_GOAL}\"")
print(f"Ego follows the bike using live pose + language modality.\n")

# ── main loop ─────────────────────────────────────────────────────────────────
for step in range(MAX_STEPS):
    world.tick()

    try:
        raw = img_queue.get(timeout=2.0)
    except queue.Empty:
        continue

    arr = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    pil = Image.fromarray(arr[:, :, :3][:, :, ::-1])
    pil.save("./inference/current_img.jpg")

    # compute live goal pose toward the bike
    ego_tf       = vehicle.get_transform()
    bike_loc     = bike.get_location()
    goal_pose_np = carla_to_goal_pose(ego_tf, bike_loc)
    real_dist    = math.sqrt((ego_tf.location.x - bike_loc.x)**2 + (ego_tf.location.y - bike_loc.y)**2)

    lin, ang = omnivla_infer(pil, goal_pose_np, step, real_dist)

    vehicle.apply_control(linear_angular_to_carla(lin, ang))

    # spectator chase-cam behind ego
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
