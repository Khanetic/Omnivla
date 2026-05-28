"""
carla_person_follower.py — Person-Following Robot in CARLA Simulator
====================================================================
Plugs a CARLA RGB camera into the three-thread architecture from
person_follower.py, adds a pygame GUI window.

Usage:
    DISPLAY=:1 conda run -n omnivla python carla_person_follower.py
    DISPLAY=:1 conda run -n omnivla python carla_person_follower.py \\
        --town Town01 --npcs 10 --tick_hz 3
"""
from __future__ import annotations

import argparse
import os
import queue
import random
import signal
import sys
import threading
import time
from collections import deque
from typing import Optional, Tuple

import carla
import cv2
import numpy as np
import pygame
from PIL import Image

# ── import shared components from person_follower.py ─────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from person_follower import (
    SharedState,
    ActionBuffer,
    _draw_debug,
    navigation_thread,
    control_thread,
    _fps_logger,
    _scalar,
    _init_tracker,
    send_velocity,
)

# ══════════════════════════════════════════════════════════════════════════════
# § 1  CARLA Camera → cv2-compatible Capture wrapper
# ══════════════════════════════════════════════════════════════════════════════

class CARLACameraCapture:
    """
    Wraps a CARLA RGB camera sensor with the same interface as
    cv2.VideoCapture so perception_thread can use it unchanged.
    """

    def __init__(self, camera_actor: carla.Actor,
                 width: int = 640, height: int = 480) -> None:
        self._q:      queue.Queue = queue.Queue(maxsize=4)
        self._width  = width
        self._height = height
        camera_actor.listen(self._on_image)

    def _on_image(self, img: carla.Image) -> None:
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))   # BGRA
        bgr = arr[:, :, :3].copy()                       # drop alpha
        if self._q.full():
            try: self._q.get_nowait()
            except queue.Empty: pass
        self._q.put(bgr)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            return True, self._q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def isOpened(self) -> bool:
        return True

    def set(self, *_) -> None:
        pass   # no-op (no frame seeking in live CARLA stream)

    def release(self) -> None:
        pass

    @property
    def frame_w(self) -> int:
        return self._width

    @property
    def frame_h(self) -> int:
        return self._height


# ══════════════════════════════════════════════════════════════════════════════
# § 2  CARLA Velocity → VehicleControl  (same as old carla_runner.py)
# ══════════════════════════════════════════════════════════════════════════════

MIN_DIST = 1.0   # m — brake if closer than this
MAX_DIST = 2.5   # m — accelerate if farther than this

def _vel_to_control(vx: float, wz: float, dist_m: float) -> carla.VehicleControl:
    import numpy as _np
    if dist_m > 0:
        if dist_m < MIN_DIST:
            vx = max(0.0, vx - 0.15)
        elif dist_m > MAX_DIST:
            vx = min(0.3, vx + 0.05)
    throttle = float(_np.clip(vx / 0.3 * 0.40, 0.0, 1.0))
    brake    = 0.3 if vx < 0.02 else 0.0
    steer    = float(_np.clip(-wz / 0.3, -1.0, 1.0))
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


# ══════════════════════════════════════════════════════════════════════════════
# § 3  CARLA Perception Thread  (adds spectator follow + CARLA tick sync)
# ══════════════════════════════════════════════════════════════════════════════

def carla_perception_thread(
    state:      SharedState,
    cap:        CARLACameraCapture,
    model_path: str,
    device:     str,
    vehicle:    carla.Actor,
    spectator:  carla.Actor,
    world:      carla.World,
) -> None:
    """
    Identical to person_follower.perception_thread but also moves the
    spectator camera behind the ego each step so the CARLA GUI follows
    the action.
    """
    print("[Perception] Initialising …")
    try:
        detector, tracker = _init_tracker(model_path, device)
        available = True
        print("[Perception] Ready.")
    except Exception as exc:
        print(f"[Perception] Init failed ({exc}) — dummy mode.")
        available = detector = tracker = False

    fps_buf = deque(maxlen=60)

    while state.running:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        state.write_frame(frame)

        # ── tracker ───────────────────────────────────────────────────────────
        if available:
            try:
                detections = detector.detect(frame)
                tracks     = tracker.update(frame, detections)
                persons    = [t for t in tracks
                              if len(t) >= 6 and _scalar(t[5]) == 0]
            except Exception as exc:
                print(f"[Perception] error: {exc}")
                persons = []

            if not persons:
                state.write_lost()
            else:
                primary = min(persons, key=lambda t: _scalar(t[4]))
                x1 = _scalar(primary[0]); y1 = _scalar(primary[1])
                x2 = _scalar(primary[2]); y2 = _scalar(primary[3])
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    crop_pil = Image.fromarray(frame[cy1:cy2, cx1:cx2][:, :, ::-1])
                else:
                    crop_pil = Image.new("RGB", (96, 96), (128, 128, 128))
                state.write_track([x1, y1, x2, y2], crop_pil)
        else:
            state.write_lost()

        # ── spectator follows ego ─────────────────────────────────────────────
        try:
            tf  = vehicle.get_transform()
            fwd = tf.get_forward_vector()
            spectator.set_transform(carla.Transform(
                carla.Location(
                    x=tf.location.x - fwd.x * 10.0,
                    y=tf.location.y - fwd.y * 10.0,
                    z=tf.location.z + 5.0,
                ),
                carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw),
            ))
        except Exception:
            pass

        fps_buf.append(time.perf_counter())
        if len(fps_buf) > 1:
            state.fps_perc = (len(fps_buf) - 1) / (fps_buf[-1] - fps_buf[0] + 1e-6)

    print("[Perception] Thread exited.")


# ══════════════════════════════════════════════════════════════════════════════
# § 4  CARLA Control Thread  (sends VehicleControl instead of stub)
# ══════════════════════════════════════════════════════════════════════════════

def carla_control_thread(
    state:   SharedState,
    buf:     ActionBuffer,
    vehicle: carla.Actor,
) -> None:
    """
    Drains ActionBuffer at 37 ms/step, converts (linear, angular) to
    carla.VehicleControl, and applies it to the ego vehicle.
    """
    STEP_DT = 1.0 / (3.0 * 8)
    fps_buf = deque(maxlen=60)
    print("[Control] Started.")

    while state.running:
        t0 = time.perf_counter()

        _, goal_img, track_lost = state.read_track()
        # rough distance from goal_image size heuristic (good enough for braking)
        bbox, _, _ = state.read_track()
        dist_m = 0.0
        if bbox is not None:
            import math
            bh = max(float(bbox[3] - bbox[1]), 1.0)
            focal_px = 640 / (2.0 * math.tan(math.radians(90.0 / 2.0)))
            dist_m = min(max((focal_px * 1.7) / bh, 0.5), 8.0)

        if track_lost:
            try:
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0))
            except Exception:
                pass
            state.write_velocity(0.0, 0.0)
            buf.clear()
            time.sleep(STEP_DT)
            continue

        cmd = buf.get_next(timeout=STEP_DT * 2)
        if cmd is not None:
            linear, angular = cmd
            try:
                ctrl = _vel_to_control(linear, angular, dist_m)
                vehicle.apply_control(ctrl)
            except Exception:
                pass
            state.write_velocity(linear, angular)
        else:
            try:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3))
            except Exception:
                pass

        fps_buf.append(time.perf_counter())
        if len(fps_buf) > 1:
            state.fps_ctrl = (len(fps_buf) - 1) / (fps_buf[-1] - fps_buf[0] + 1e-6)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, STEP_DT - elapsed))

    print("[Control] Thread exited.")


# ══════════════════════════════════════════════════════════════════════════════
# § 5  Pygame HUD panel
# ══════════════════════════════════════════════════════════════════════════════

class HUD:
    BG = (25, 25, 25); FG = (220, 220, 220)
    GREEN = (50, 210, 50); AMBER = (210, 160, 30); RED = (210, 50, 50)

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect  = rect
        self.font  = pygame.font.SysFont("monospace", 18)
        self.sfont = pygame.font.SysFont("monospace", 13)

    def render(self, surf: pygame.Surface, state: SharedState, step: int) -> None:
        pygame.draw.rect(surf, self.BG, self.rect)
        bbox, _, lost  = state.read_track()
        lin, ang       = state.read_velocity()
        status = "LOST" if lost else ("TRACKING" if bbox else "SEARCHING")
        sc = self.RED if lost else (self.GREEN if bbox else self.AMBER)

        rows = [
            ("PERSON FOLLOWER",          self.FG,    True),
            (f"Step  : {step}",          self.FG,    False),
            ("",                         self.FG,    False),
            (f"State : {status}",        sc,         False),
            (f"linear: {lin:+.3f} m/s",  self.GREEN, False),
            (f"angular:{ang:+.3f} r/s",  self.AMBER, False),
            ("",                         self.FG,    False),
            (f"Perc  : {state.fps_perc:4.1f} Hz", self.FG, False),
            (f"Nav   : {state.fps_nav:4.1f} Hz",  self.FG, False),
            (f"Ctrl  : {state.fps_ctrl:4.1f} Hz", self.FG, False),
            ("",                         self.FG,    False),
            ("Press Q to quit",          self.FG,    False),
        ]
        x0 = self.rect.x + 10
        y0 = self.rect.y + 12
        lh = 22
        for i, (txt, col, bold) in enumerate(rows):
            f = pygame.font.SysFont("monospace", 18, bold=bold)
            surf.blit(f.render(txt, True, col), (x0, y0 + i * lh))


# ══════════════════════════════════════════════════════════════════════════════
# § 6  CARLA world setup helpers
# ══════════════════════════════════════════════════════════════════════════════

def _spawn_ego(world, bp_lib, spawn_points):
    preferred = ["vehicle.micro.microlino", "vehicle.nissan.micra",
                 "vehicle.seat.leon", "vehicle.audi.tt"]
    all4 = sorted(
        [b for b in bp_lib.filter("vehicle.*")
         if b.has_attribute("number_of_wheels")
         and int(b.get_attribute("number_of_wheels")) == 4],
        key=lambda b: b.id,
    )
    blueprints = []
    for vid in preferred:
        try: blueprints.append(bp_lib.find(vid))
        except IndexError: pass
    blueprints += all4

    for sp in spawn_points:
        for bp in blueprints:
            v = world.try_spawn_actor(bp, sp)
            if v:
                return v
    raise RuntimeError("No free spawn point for ego vehicle.")


def _spawn_pedestrian(world, bp_lib, ego_tf, lead_dist=8.0):
    fwd = ego_tf.get_forward_vector()
    loc = ego_tf.location
    ped_bp = random.choice(list(bp_lib.filter("walker.pedestrian.*")))
    ped_sp = carla.Transform(
        carla.Location(x=loc.x + fwd.x * lead_dist,
                       y=loc.y + fwd.y * lead_dist,
                       z=loc.z + 0.5),
        ego_tf.rotation,
    )
    ped = world.try_spawn_actor(ped_bp, ped_sp)
    if ped is None:
        ped = world.spawn_actor(
            ped_bp,
            carla.Transform(world.get_random_location_from_navigation()),
        )
    ctrl_bp  = bp_lib.find("controller.ai.walker")
    ped_ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=ped)
    world.tick()
    ped_ctrl.start()
    walk_tgt = carla.Location(x=loc.x + fwd.x * 60.0,
                              y=loc.y + fwd.y * 60.0,
                              z=loc.z)
    ped_ctrl.go_to_location(walk_tgt)
    ped_ctrl.set_max_speed(0.8)
    return ped, ped_ctrl


# ══════════════════════════════════════════════════════════════════════════════
# § 7  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       type=int, default=2000)
    p.add_argument("--town",       default="Town01")
    p.add_argument("--tick_hz",    type=int, default=3)
    p.add_argument("--npcs",       type=int, default=10)
    p.add_argument("--max_steps",  type=int, default=2000)
    p.add_argument("--checkpoint", default=os.path.expanduser("~/OmniVLA/omnivla-edge"))
    p.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model_path", default=os.path.expanduser("~/OmniVLA/yolov8n-seg.pt"))
    p.add_argument("--cam_w",      type=int, default=640)
    p.add_argument("--cam_h",      type=int, default=480)
    return p.parse_args()


def main():
    args = parse_args()

    # ── connect to CARLA ──────────────────────────────────────────────────────
    print(f"[Main] Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)

    current_map = client.get_world().get_map().name
    if args.town in current_map:
        print(f"[Main] Already on {current_map}.")
        world = client.get_world()
    else:
        print(f"[Main] Loading {args.town} …")
        world = client.load_world(args.town)
    print(f"[Main] Map: {world.get_map().name}")

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / args.tick_hz
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    bp_lib       = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spectator    = world.get_spectator()
    actors       = []

    def cleanup(sig=None, _f=None):
        print("\n[Main] Cleaning up …")
        state.running = False
        settings.synchronous_mode = False
        tm.set_synchronous_mode(False)
        world.apply_settings(settings)
        for a in actors:
            try:
                if a.is_alive: a.destroy()
            except Exception: pass
        pygame.quit()
        print("[Main] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    # ── spawn actors ──────────────────────────────────────────────────────────
    vehicle = _spawn_ego(world, bp_lib, spawn_points)
    actors.append(vehicle)
    print(f"[Main] Ego: {vehicle.type_id}  at {vehicle.get_location()}")

    world.tick()
    ego_tf = vehicle.get_transform()

    ped, ped_ctrl = _spawn_pedestrian(world, bp_lib, ego_tf)
    actors += [ped, ped_ctrl]
    print(f"[Main] Pedestrian: {ped.type_id}  at {ped.get_location()}")

    # NPC vehicles
    ego_loc   = vehicle.get_location()
    npc_count = 0
    for sp in spawn_points:
        if npc_count >= args.npcs:
            break
        if sp.location.distance(ego_loc) < 5.0:
            continue
        candidates = [b for b in bp_lib.filter("vehicle.*")
                      if b.has_attribute("number_of_wheels")
                      and int(b.get_attribute("number_of_wheels")) == 4]
        npc = world.try_spawn_actor(random.choice(candidates), sp)
        if npc:
            npc.set_autopilot(True, 8000)
            actors.append(npc)
            npc_count += 1
    print(f"[Main] {npc_count} NPC vehicles spawned.")

    # ── attach RGB camera ─────────────────────────────────────────────────────
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.cam_w))
    cam_bp.set_attribute("image_size_y", str(args.cam_h))
    cam_bp.set_attribute("fov", "90")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.6, z=1.8)),
        attach_to=vehicle,
    )
    actors.append(camera)

    cap = CARLACameraCapture(camera, args.cam_w, args.cam_h)

    # warm up CARLA ticks so the camera queue fills
    for _ in range(10):
        world.tick()

    # ── shared state + buffer ─────────────────────────────────────────────────
    state = SharedState()
    buf   = ActionBuffer()

    # ── CARLA tick thread (keeps synchronous mode alive) ─────────────────────
    def _tick_loop():
        while state.running:
            try:
                world.tick()
                # re-target pedestrian every 90 ticks
                if hasattr(_tick_loop, "step"):
                    _tick_loop.step += 1
                else:
                    _tick_loop.step = 0
                if _tick_loop.step % 90 == 0:
                    try:
                        pf  = ped.get_transform().get_forward_vector()
                        pl  = ped.get_location()
                        ped_ctrl.go_to_location(
                            carla.Location(x=pl.x + pf.x * 20.0,
                                           y=pl.y + pf.y * 20.0,
                                           z=pl.z))
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(1.0 / args.tick_hz)

    tick_thread = threading.Thread(target=_tick_loop, name="carla_tick", daemon=True)

    # ── start pipeline threads ────────────────────────────────────────────────
    threads = [
        tick_thread,
        threading.Thread(
            target=carla_perception_thread,
            args=(state, cap, args.model_path, args.device,
                  vehicle, spectator, world),
            name="perception", daemon=True,
        ),
        threading.Thread(
            target=navigation_thread,
            args=(state, buf, args.checkpoint, args.device,
                  args.cam_w, args.cam_h),
            name="navigation", daemon=True,
        ),
        threading.Thread(
            target=carla_control_thread,
            args=(state, buf, vehicle),
            name="control", daemon=True,
        ),
        threading.Thread(
            target=_fps_logger, args=(state,),
            name="fps_logger", daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    # ── pygame window ─────────────────────────────────────────────────────────
    pygame.init()
    PANEL_W  = 260
    DISP_W   = args.cam_w + PANEL_W
    DISP_H   = max(args.cam_h, 380)
    screen   = pygame.display.set_mode((DISP_W, DISP_H))
    pygame.display.set_caption("Person Follower — CARLA")
    clock    = pygame.time.Clock()
    hud      = HUD(pygame.Rect(args.cam_w, 0, PANEL_W, DISP_H))

    print(f"\n[Main] Running — lang=\"follow the person ahead\"")
    print("[Main] Press Q or close window to quit.\n")

    step = 0
    try:
        while state.running and step < args.max_steps:
            # ── pygame events ─────────────────────────────────────────────────
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    cleanup()
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    cleanup()

            # ── draw camera feed ──────────────────────────────────────────────
            frame_bgr, _ = state.read_frame()
            if frame_bgr is not None:
                annotated = _draw_debug(frame_bgr, state)
                rgb = annotated[:, :, ::-1]          # BGR → RGB
                cam_surf = pygame.surfarray.make_surface(
                    np.transpose(rgb, (1, 0, 2)))
                screen.blit(cam_surf, (0, 0))

            # ── draw HUD panel ────────────────────────────────────────────────
            hud.render(screen, state, step)

            pygame.display.flip()
            clock.tick(30)
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
