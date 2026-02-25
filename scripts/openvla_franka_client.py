#!/usr/bin/env python3
"""
OpenVLA → Franka FR3 control client (Cartesian velocity server).

Reads RGB images from a ZED camera, runs OpenVLA inference to predict
end-effector deltas [dx, dy, dz, droll, dpitch, dyaw, gripper], applies
them as incremental pose updates, and sends the target EE pose to
franka_server_cartesian over gRPC at 10 Hz.

The Cartesian server runs a PD controller at 1 kHz converting the pose
error to Cartesian velocity commands — no client-side IK needed.

Prerequisites:
  1. franka_server_cartesian is running on the NUC/server:
       docker exec <container> bash /app/droid/franka_direct/launch_server_cartesian.sh
  2. ZED camera is connected via USB.
  3. GPU is available for OpenVLA inference.
  4. Python packages installed (see bottom of this file).

Usage:
  python scripts/openvla_franka_client.py \
      --task "pick up the red block" \
      --host 192.168.1.6 --port 50052 \
      --hz 10

Controls:
  Ctrl+C  : Stop and exit
  q+Enter : Stop and exit
"""

import argparse
import os
import select
import signal
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENVLA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DROID_ROOT = os.path.abspath(os.path.join(OPENVLA_ROOT, "..", "droid"))

# Make franka_direct importable
sys.path.insert(0, os.path.join(DROID_ROOT, "franka_direct", "python"))

from franka_direct_client import FrankaDirectClient


# ── Rotation helpers (same as simple_pose_direct.py) ─────────────────────────

def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose16_to_mat(pose16):
    """Column-major 16 floats -> 4x4 numpy array."""
    return np.array(pose16).reshape(4, 4, order="F")

def mat_to_pose16(T):
    """4x4 numpy array -> column-major 16 floats list."""
    return T.flatten(order="F").tolist()


# ── General helpers ──────────────────────────────────────────────────────────

def check_keyboard():
    """Non-blocking keyboard read. Returns stripped line or None."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


def load_openvla(model_path, device):
    """Load OpenVLA model and processor."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"[*] Loading OpenVLA model from: {model_path}")
    print("[*] This may take a few minutes on first run (downloading ~15GB)...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    print("[OK] OpenVLA model loaded")
    return vla, processor


@torch.inference_mode()
def predict_action(vla, processor, image_pil, task_label, device, unnorm_key=None):
    """
    Run OpenVLA inference on a single RGB image.

    Returns:
        np.ndarray of shape (7,): [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image_pil).to(device, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def init_zed_camera(resolution="HD720", fps=15):
    """Initialize the ZED camera and return (camera, runtime_parameters)."""
    import pyzed.sl as sl

    zed = sl.Camera()

    init_params = sl.InitParameters()
    res_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }
    init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.HD720)
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # We only need RGB

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {status}")

    runtime = sl.RuntimeParameters()
    print(f"[OK] ZED camera opened ({resolution}, {fps} fps)")
    return zed, runtime


def grab_rgb_image(zed, runtime):
    """Grab a single RGB frame from the ZED camera. Returns PIL Image or None."""
    import pyzed.sl as sl

    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None

    image_zed = sl.Mat()
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    # ZED returns BGRA numpy array
    bgra = image_zed.get_data()
    rgb = bgra[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return Image.fromarray(rgb.copy())


def apply_eef_delta(T_current, delta_6):
    """
    Apply a 6-DoF EEF delta to a 4x4 homogeneous transform.

    Args:
        T_current: 4x4 numpy array (current EE pose).
        delta_6:   [dx, dy, dz, droll, dpitch, dyaw] in base frame.
                   Translation in metres, rotation in radians.

    Returns:
        T_new: 4x4 numpy array (new EE pose).
    """
    dp = delta_6[:3]
    dr = delta_6[3:6]

    p_current = T_current[:3, 3]
    R_current = T_current[:3, :3]

    # Apply translation in base frame
    p_new = p_current + dp

    # Apply rotation as extrinsic XYZ in base frame (pre-multiply)
    R_delta = rot_z(dr[2]) @ rot_y(dr[1]) @ rot_x(dr[0])
    R_new = R_delta @ R_current

    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = p_new
    return T_new


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA → Franka FR3 control via ZED camera (Cartesian server)"
    )
    # Task
    parser.add_argument("--task", type=str, required=True,
                        help='Natural language task, e.g. "pick up the red block"')
    # Model
    parser.add_argument("--model", type=str, default="openvla/openvla-7b",
                        help="HF model name or local path (default: openvla/openvla-7b)")
    parser.add_argument("--unnorm_key", type=str, default=None,
                        help="Dataset key for action un-normalization (default: auto)")
    # Robot server
    parser.add_argument("--host", type=str, default="192.168.1.6",
                        help="franka_server_cartesian host (default: 192.168.1.6)")
    parser.add_argument("--port", type=int, default=50052,
                        help="franka_server_cartesian gRPC port (default: 50052)")
    # Control
    parser.add_argument("--hz", type=int, default=10,
                        help="Control loop frequency in Hz (default: 10)")
    parser.add_argument("--no_reset", action="store_true",
                        help="Skip robot reset to home position")
    parser.add_argument("--reset_speed", type=float, default=0.2,
                        help="Joint move speed factor [0..1] for reset (default: 0.2)")
    # Camera
    parser.add_argument("--camera_resolution", type=str, default="HD720",
                        choices=["HD2K", "HD1080", "HD720", "VGA"],
                        help="ZED camera resolution (default: HD720)")
    parser.add_argument("--crop", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"),
                        help="Crop region [x, y, width, height] in pixels. "
                             "If not set, center-crops to a square.")
    parser.add_argument("--save_video", type=str, default=None,
                        metavar="PATH",
                        help="Save camera feed to a video file (e.g. output.mp4). "
                             "If not set, no video is saved.")
    # Action scaling — tune these to match your setup
    parser.add_argument("--action_scale", type=float, default=1.0,
                        help="Multiplier for the 6-DoF EEF action (default: 1.0)")
    parser.add_argument("--gripper_threshold", type=float, default=0.5,
                        help="Gripper action threshold: >thresh=close, <-thresh=open (default: 0.5)")
    args = parser.parse_args()

    loop_period = 1.0 / args.hz

    # ── Ctrl+C handler ────────────────────────────────────────────────────────
    running = True

    def _sigint(sig, frame):
        nonlocal running
        print("\n\nCtrl+C detected. Stopping...")
        running = False

    signal.signal(signal.SIGINT, _sigint)

    print("=" * 60)
    print("OpenVLA → Franka FR3  (Cartesian server, ZED camera)")
    print("=" * 60)
    print(f"  Task:   {args.task}")
    print(f"  Model:  {args.model}")
    print(f"  Server: {args.host}:{args.port}")
    print(f"  Hz:     {args.hz}")
    print("=" * 60)

    # === Step 1: Load OpenVLA model ============================================
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("[WARN] No GPU detected — inference will be very slow!")
    vla, processor = load_openvla(args.model, device)

    # === Step 2: Initialize ZED camera =========================================
    zed, runtime = init_zed_camera(resolution=args.camera_resolution, fps=args.hz)

    # === Step 3: Connect to franka_server_cartesian ============================
    print(f"\nConnecting to franka_server_cartesian at {args.host}:{args.port} ...")
    client = FrankaDirectClient(host=args.host, port=args.port)
    try:
        state = client.wait_until_ready(timeout=20.0)
        print(f"[OK] Server ready (cmd_success_rate={state['cmd_success_rate']:.3f})")
    except (TimeoutError, RuntimeError) as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # === Step 4: Reset robot to home (blocking RPC) ============================

    # HOME_Q = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, 0.0]
    HOME_Q = [0.0, 0, 0.0, -4 * np.pi / 5, 0.0, 4 * np.pi / 5, 0.0]

    if not args.no_reset:
        print(f"Resetting robot to home position (speed={args.reset_speed}) ...")
        ok, msg = client.reset_to_joints(HOME_Q, speed=args.reset_speed)
        if ok:
            print("[OK] Robot at home position")
        else:
            print(f"[FAIL] Reset failed: {msg}")
            sys.exit(1)
    else:
        print("[SKIP] Robot reset skipped (--no_reset)")

    # Open gripper to known state
    print("Opening gripper ...")
    client.set_gripper_target(0.08, 0.1)
    time.sleep(1.0)
    print("[OK] Gripper open")

    # === Step 5: Warm-up inference =============================================
    print("Running warm-up inference ...")
    dummy_image = Image.new("RGB", (640, 480), color=(128, 128, 128))
    _ = predict_action(vla, processor, dummy_image, args.task, device, args.unnorm_key)
    print("[OK] Warm-up done")

    # === Step 6: Main control loop =============================================
    print()
    print("=" * 60)
    print("RUNNING — press Ctrl+C or type 'q'+Enter to stop")
    print("=" * 60)
    print()

    step_count = 0
    gripper_open = True
    video_writer = None  # initialized on first frame when --save_video is set
    GRIPPER_OPEN = 0.08   # Franka Hand max width [m]
    GRIPPER_CLOSE = 0.0
    GRIPPER_SPEED = 0.1   # finger speed [m/s]

    while running:
        loop_start = time.time()
        step_count += 1

        # ── Keyboard input ────────────────────────────────────────────────────
        key = check_keyboard()
        if key == "q":
            print("\n\n[DONE] 'q' pressed — quitting")
            break

        # ── Grab image from ZED camera ────────────────────────────────────────
        image_pil = grab_rgb_image(zed, runtime)
        if image_pil is None:
            print("\n[WARN] Failed to grab ZED frame, skipping step")
            continue

        # ── Crop image ────────────────────────────────────────────────────────
        if args.crop is not None:
            cx, cy, cw, ch = args.crop
            image_pil = image_pil.crop((cx, cy, cx + cw, cy + ch))
        else:
            # Center-crop to square (avoid squishing 16:9 → 1:1)
            w, h = image_pil.size
            if w != h:
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                image_pil = image_pil.crop((left, top, left + side, top + side))

        # Show camera feed in a window (press 'q' in the window to quit)
        bgr_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("ZED Camera", bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n\n[DONE] 'q' pressed in camera window — quitting")
            break

        # Save frame to video file
        if args.save_video is not None:
            if video_writer is None:
                h_f, w_f = bgr_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(args.save_video, fourcc, args.hz, (w_f, h_f))
                print(f"[OK] Recording video to {args.save_video} ({w_f}x{h_f} @ {args.hz} fps)")
            video_writer.write(bgr_frame)

        # ── Run OpenVLA inference ─────────────────────────────────────────────
        t_infer = time.time()
        action = predict_action(
            vla, processor, image_pil, args.task, device, args.unnorm_key
        )
        infer_ms = (time.time() - t_infer) * 1000

        # action shape (7,): [dx, dy, dz, droll, dpitch, dyaw, gripper]
        print(f"\n  raw action: {action}")
        eef_delta = action[:6] * args.action_scale
        gripper_action = action[6]

        # ── Get current robot state ───────────────────────────────────────────
        state = client.get_robot_state()
        if state["error"]:
            print(f"\n[ERROR] Robot error: {state['error']}")
            break

        # ── Apply EEF delta to current pose and send ──────────────────────────
        T_current = pose16_to_mat(state["pose"])
        T_target = apply_eef_delta(T_current, eef_delta)
        ok, msg = client.set_ee_target(mat_to_pose16(T_target))
        if not ok:
            print(f"\n[ERROR] SetEETarget failed: {msg}")
            break

        # ── Gripper control ───────────────────────────────────────────────────
        # OpenVLA gripper output convention:
        #   Positive values → close, negative values → open
        #   (exact convention depends on the dataset; adjust threshold as needed)
        want_closed = gripper_action > args.gripper_threshold
        want_open = gripper_action < -args.gripper_threshold
        if want_closed and gripper_open:
            client.set_gripper_target(GRIPPER_CLOSE, GRIPPER_SPEED)
            gripper_open = False
        elif want_open and not gripper_open:
            client.set_gripper_target(GRIPPER_OPEN, GRIPPER_SPEED)
            gripper_open = True

        # ── Regulate frequency ────────────────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_t = loop_period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

        actual_hz = 1.0 / max(time.time() - loop_start, 1e-6)
        p = T_current[:3, 3]
        sys.stdout.write(
            f"\r[Step {step_count:>6}] "
            f"Hz: {actual_hz:>5.1f} | "
            f"infer: {infer_ms:>5.0f}ms | "
            f"pos=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]m | "
            f"grip={'OPEN' if gripper_open else 'CLOSED'} | "
            f"delta=[{eef_delta[0]:+.4f} {eef_delta[1]:+.4f} {eef_delta[2]:+.4f}]    "
        )
        sys.stdout.flush()

    # === Cleanup ==============================================================
    print("\n\nOpenVLA control ended.")
    print(f"Total steps: {step_count}")
    try:
        client.stop()
        client.close()
    except Exception:
        pass
    try:
        zed.close()
    except Exception:
        pass
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {args.save_video}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
