#!/usr/bin/env python3
"""
OpenVLA-OFT → Franka FR3 control client (Cartesian velocity server).

Uses OpenVLA-OFT (Optimized Fine-Tuning) with L1 regression action head
and action chunking. Generates chunks of 8 actions per forward pass,
then executes them sequentially before re-querying the model.

Key differences from base OpenVLA (openvla_franka_client.py):
  - ~26x faster inference via parallel decoding + L1 regression head
  - Action chunking: 8 actions per forward pass (configurable open-loop)
  - Proprioceptive input (EEF pose + gripper state)
  - Continuous action prediction (not discrete tokenization)

Prerequisites:
  1. Install openvla-oft (replaces the base openvla prismatic package):
       git clone https://github.com/moojink/openvla-oft ~/cyqian/openvla-oft
       cd ~/cyqian/openvla-oft && pip install -e .
     If you need both base and OFT, use separate virtual environments.
  2. Extra packages:
       pip install huggingface_hub grpcio grpcio-tools "numpy<2"
  3. franka_server_cartesian running on the NUC/server.
  4. ZED camera connected via USB.
  5. GPU available (~16 GB VRAM in bf16).

Usage:
  python scripts/openvla_oft_franka_client.py \
      --task "pick up the red block" \
      --host 192.168.1.6 --port 50052 \
      --action_hz 20

Controls:
  Ctrl+C  : Stop and exit
  q+Enter : Stop and exit
"""

import argparse
import json
import math
import os
import select
import signal
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# ── OFT constants (LIBERO platform) ─────────────────────────────────────────
# Defined inline so the script works even if prismatic.vla.constants doesn't
# exist yet. The module is monkey-patched below before model loading triggers
# the OFT modeling code's imports.

NUM_ACTIONS_CHUNK = 8   # actions per forward pass
ACTION_DIM = 7          # [dx, dy, dz, droll, dpitch, dyaw, gripper]
PROPRIO_DIM = 8         # [eef_pos(3), eef_rot_axisangle(3), gripper_qpos(2)]

# Ensure prismatic.vla.constants exists for the OFT model's trust_remote_code
try:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: F811
except (ImportError, ModuleNotFoundError):
    import types as _types

    _constants_mod = _types.ModuleType("prismatic.vla.constants")
    for _k, _v in {
        "IGNORE_INDEX": -100,
        "ACTION_TOKEN_BEGIN_IDX": 31743,
        "STOP_INDEX": 2,
        "NUM_ACTIONS_CHUNK": NUM_ACTIONS_CHUNK,
        "ACTION_DIM": ACTION_DIM,
        "PROPRIO_DIM": PROPRIO_DIM,
        "ACTION_PROPRIO_NORMALIZATION_TYPE": "bounds_q99",
    }.items():
        setattr(_constants_mod, _k, _v)
    sys.modules["prismatic.vla.constants"] = _constants_mod

    # Also make sure prismatic.vla exists as a package in sys.modules
    if "prismatic.vla" not in sys.modules:
        _vla_mod = _types.ModuleType("prismatic.vla")
        _vla_mod.constants = _constants_mod
        sys.modules["prismatic.vla"] = _vla_mod
    else:
        sys.modules["prismatic.vla"].constants = _constants_mod


# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENVLA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DROID_ROOT = os.path.abspath(os.path.join(OPENVLA_ROOT, "..", "droid"))

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


def rotation_matrix_to_axis_angle(R):
    """Convert 3x3 rotation matrix to 3D axis-angle vector."""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta


def apply_eef_delta(T_current, delta_6):
    """
    Apply a 6-DoF EEF delta to a 4x4 homogeneous transform.

    Args:
        T_current: 4x4 numpy array (current EE pose).
        delta_6:   [dx, dy, dz, droll, dpitch, dyaw] in base frame.

    Returns:
        T_new: 4x4 numpy array (new EE pose).
    """
    dp = delta_6[:3]
    dr = delta_6[3:6]

    p_new = T_current[:3, 3] + dp
    R_delta = rot_z(dr[2]) @ rot_y(dr[1]) @ rot_x(dr[0])
    R_new = R_delta @ T_current[:3, :3]

    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = p_new
    return T_new


# ── Inline MLP classes (from openvla-oft action_heads.py / projectors.py) ────

class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    """MLP action head that maps LLM hidden states to continuous actions."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)


class ProprioProjector(nn.Module):
    """Projects proprioceptive state into the LLM's embedding space."""

    def __init__(self, llm_dim, proprio_dim):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)

    def forward(self, proprio):
        return self.fc2(self.act_fn1(self.fc1(proprio)))


# ── General helpers ──────────────────────────────────────────────────────────

def check_keyboard():
    """Non-blocking keyboard read. Returns stripped line or None."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


# ── Checkpoint name mapping (HuggingFace Hub) ────────────────────────────────

ACTION_HEAD_CKPT = {
    "moojink/openvla-7b-oft-finetuned-libero-spatial": "action_head--150000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-object": "action_head--150000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-goal": "action_head--50000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-10": "action_head--150000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "action_head--300000_checkpoint.pt",
}

PROPRIO_PROJ_CKPT = {
    "moojink/openvla-7b-oft-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-object": "proprio_projector--150000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-goal": "proprio_projector--50000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-10": "proprio_projector--150000_checkpoint.pt",
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "proprio_projector--300000_checkpoint.pt",
}

UNNORM_KEY_MAP = {
    "libero-spatial": "libero_spatial_no_noops",
    "libero-object": "libero_object_no_noops",
    "libero-goal": "libero_goal_no_noops",
    "libero-10": "libero_10_no_noops",
    "libero-spatial-object-goal-10": "libero_spatial_no_noops",
}


def _load_component_state_dict(path):
    """Load state dict and strip DDP 'module.' prefix if present."""
    sd = torch.load(path, weights_only=True)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


# ── Model loading ────────────────────────────────────────────────────────────

def load_oft_model(model_path, device):
    """
    Load OpenVLA-OFT model, action head, and proprio projector from HuggingFace.

    Returns:
        (vla, processor, action_head, proprio_projector)
    """
    print(f"[*] Loading OpenVLA-OFT model from: {model_path}")
    print("[*] This may take a few minutes on first run (downloading ~15 GB)...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    # num_images_in_input is set later in main() based on whether a wrist camera is used

    # Get LLM hidden dimension
    llm_dim = getattr(vla, "llm_dim", None) or vla.config.text_config.hidden_size
    print(f"[OK] Model loaded (llm_dim={llm_dim})")

    # Load dataset statistics for action/proprio normalization
    stats_path = hf_hub_download(repo_id=model_path, filename="dataset_statistics.json")
    with open(stats_path, "r") as f:
        vla.norm_stats = json.load(f)

    # ── Load L1 regression action head ────────────────────────────────────────
    print("[*] Loading action head ...")
    action_head = L1RegressionActionHead(
        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM,
    )
    ah_filename = ACTION_HEAD_CKPT.get(model_path)
    if ah_filename is None:
        raise ValueError(
            f"No action head checkpoint mapped for '{model_path}'.\n"
            f"Known models: {list(ACTION_HEAD_CKPT.keys())}"
        )
    ah_path = hf_hub_download(repo_id=model_path, filename=ah_filename)
    action_head.load_state_dict(_load_component_state_dict(ah_path))
    action_head = action_head.to(torch.bfloat16).to(device)
    action_head.eval()
    print("[OK] Action head loaded")

    # ── Load proprio projector ────────────────────────────────────────────────
    print("[*] Loading proprio projector ...")
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
    pp_filename = PROPRIO_PROJ_CKPT.get(model_path)
    if pp_filename is None:
        raise ValueError(
            f"No proprio projector checkpoint mapped for '{model_path}'.\n"
            f"Known models: {list(PROPRIO_PROJ_CKPT.keys())}"
        )
    pp_path = hf_hub_download(repo_id=model_path, filename=pp_filename)
    proprio_projector.load_state_dict(_load_component_state_dict(pp_path))
    proprio_projector = proprio_projector.to(torch.bfloat16).to(device)
    proprio_projector.eval()
    print("[OK] Proprio projector loaded")

    return vla, processor, action_head, proprio_projector


# ── Proprio normalization ────────────────────────────────────────────────────

def normalize_proprio(proprio, norm_stats):
    """Normalize proprio state to [-1, 1] using q01/q99 bounds (LIBERO convention)."""
    mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)))
    q99 = np.array(norm_stats["q99"], dtype=np.float64)
    q01 = np.array(norm_stats["q01"], dtype=np.float64)
    return np.clip(
        np.where(mask, 2.0 * (proprio - q01) / (q99 - q01 + 1e-8) - 1.0, proprio),
        -1.0, 1.0,
    )


# ── Inference ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict_action_chunk(vla, processor, action_head, proprio_projector,
                         image_pil, proprio, task_label, unnorm_key, device,
                         wrist_image_pil=None):
    """
    Run OpenVLA-OFT inference on RGB image(s) + proprio state.

    Args:
        image_pil:       PIL Image from the third-person camera.
        wrist_image_pil: PIL Image from the wrist camera (optional).
        proprio:         Raw (unnormalized) proprio state array.

    Returns:
        np.ndarray of shape (NUM_ACTIONS_CHUNK, 7):
            each row is [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process primary (third-person) image
    inputs = processor(prompt, image_pil).to(device, dtype=torch.bfloat16)

    # If wrist camera is available, process it and concatenate pixel values
    if wrist_image_pil is not None:
        wrist_inputs = processor(prompt, wrist_image_pil).to(device, dtype=torch.bfloat16)
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
        )

    # Normalize proprio using dataset statistics
    proprio_norm_stats = vla.norm_stats[unnorm_key]["proprio"]
    normed_proprio = normalize_proprio(proprio, proprio_norm_stats)

    # Call predict_action with OFT-specific parameters
    # Returns (actions, hidden_states); actions are already unnormalized
    actions, _ = vla.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        do_sample=False,
        proprio=normed_proprio,
        proprio_projector=proprio_projector,
        action_head=action_head,
    )

    # actions shape: (NUM_ACTIONS_CHUNK, ACTION_DIM) = (8, 7)
    return actions


# ── Camera ───────────────────────────────────────────────────────────────────

def list_zed_cameras():
    """List all connected ZED cameras with their serial numbers."""
    import pyzed.sl as sl
    devices = sl.Camera.get_device_list()
    print(f"[*] Found {len(devices)} ZED camera(s):")
    for d in devices:
        print(f"    Serial: {d.serial_number}  Model: {d.camera_model}  Path: {d.path}")
    return devices


def init_zed_camera(resolution="HD720", fps=15, serial_number=None, label=""):
    """
    Initialize a ZED camera and return (camera, runtime_parameters).

    Args:
        serial_number: If set, open the specific camera by serial number.
                       Required when multiple ZED cameras are connected.
        label: Label for log messages (e.g. "third-person", "wrist").
    """
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
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if serial_number is not None:
        init_params.set_from_serial_number(serial_number)

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        sn_str = f" serial={serial_number}" if serial_number else ""
        raise RuntimeError(f"Failed to open ZED camera{sn_str}: {status}")

    sn = zed.get_camera_information().serial_number
    tag = f" [{label}]" if label else ""
    print(f"[OK] ZED camera opened{tag} (serial={sn}, {resolution}, {fps} fps)")
    return zed, sl.RuntimeParameters()


def grab_rgb_image(zed, runtime):
    """Grab a single RGB frame from the ZED camera. Returns PIL Image or None."""
    import pyzed.sl as sl

    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None

    image_zed = sl.Mat()
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    bgra = image_zed.get_data()
    rgb = bgra[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    return Image.fromarray(rgb.copy())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA-OFT → Franka FR3 control via ZED camera (Cartesian server)"
    )
    # Task
    parser.add_argument("--task", type=str, required=True,
                        help='Natural language task, e.g. "pick up the red block"')
    # Model
    parser.add_argument("--model", type=str,
                        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
                        help="HuggingFace OFT checkpoint (default: libero-spatial)")
    parser.add_argument("--unnorm_key", type=str, default=None,
                        help="Dataset key for un-normalization (auto-detected from model name)")
    # Robot server
    parser.add_argument("--host", type=str, default="192.168.1.6",
                        help="franka_server_cartesian host (default: 192.168.1.6)")
    parser.add_argument("--port", type=int, default=50052,
                        help="franka_server_cartesian gRPC port (default: 50052)")
    # Control
    parser.add_argument("--action_hz", type=int, default=20,
                        help="Rate at which individual actions are sent to the robot (default: 20, "
                             "matching LIBERO's 20 Hz training data)")
    parser.add_argument("--open_loop_steps", type=int, default=None,
                        help=f"Actions to execute per chunk before re-querying "
                             f"(default: {NUM_ACTIONS_CHUNK}, the full chunk)")
    parser.add_argument("--no_reset", action="store_true",
                        help="Skip robot reset to home position")
    parser.add_argument("--reset_speed", type=float, default=0.2,
                        help="Joint move speed factor [0..1] for reset (default: 0.2)")
    # Camera
    parser.add_argument("--camera_resolution", type=str, default="HD720",
                        choices=["HD2K", "HD1080", "HD720", "VGA"],
                        help="ZED camera resolution (default: HD720)")
    parser.add_argument("--third_person_serial", type=int, default=None,
                        help="Serial number of the third-person ZED camera (e.g. ZED 2i). "
                             "Run with --list_cameras to see available serials.")
    parser.add_argument("--wrist_serial", type=int, default=None,
                        help="Serial number of the wrist-mounted ZED camera (e.g. ZED Mini). "
                             "If not set, only the third-person camera is used (1 image input).")
    parser.add_argument("--list_cameras", action="store_true",
                        help="List connected ZED cameras and exit")
    parser.add_argument("--crop", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"),
                        help="Crop region for third-person camera [x, y, w, h] in pixels. "
                             "If not set, center-crops to a square.")
    parser.add_argument("--wrist_crop", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"),
                        help="Crop region for wrist camera [x, y, w, h] in pixels. "
                             "If not set, center-crops to a square.")
    parser.add_argument("--save_video", type=str, default=None, metavar="PATH",
                        help="Save camera feed to a video file (e.g. output.mp4)")
    # Action tuning
    parser.add_argument("--action_scale", type=float, default=1.0,
                        help="Multiplier for the 6-DoF EEF delta (default: 1.0)")
    parser.add_argument("--gripper_threshold", type=float, default=0.5,
                        help="Gripper action threshold (default: 0.5)")
    # OFT-specific
    parser.add_argument("--no_proprio", action="store_true",
                        help="Disable proprioceptive input (not recommended)")
    parser.add_argument("--center_crop", action="store_true", default=True,
                        help="Apply sqrt(0.9) center crop to match OFT training (default: True)")
    parser.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    args = parser.parse_args()

    # Handle --list_cameras
    if args.list_cameras:
        list_zed_cameras()
        sys.exit(0)

    if args.open_loop_steps is None:
        args.open_loop_steps = NUM_ACTIONS_CHUNK
    action_period = 1.0 / args.action_hz
    inference_hz = args.action_hz / args.open_loop_steps
    use_wrist = args.wrist_serial is not None
    num_images = 2 if use_wrist else 1

    # Auto-detect unnorm_key from model name
    if args.unnorm_key is None:
        for suffix, key in UNNORM_KEY_MAP.items():
            if suffix in args.model:
                args.unnorm_key = key
                break
        if args.unnorm_key is None:
            args.unnorm_key = "libero_spatial_no_noops"
        print(f"[*] Auto-detected unnorm_key: {args.unnorm_key}")

    # ── Ctrl+C handler ────────────────────────────────────────────────────────
    running = True

    def _sigint(sig, frame):
        nonlocal running
        print("\n\nCtrl+C detected. Stopping...")
        running = False

    signal.signal(signal.SIGINT, _sigint)

    print("=" * 60)
    print("OpenVLA-OFT → Franka FR3  (action chunking, L1 regression)")
    print("=" * 60)
    print(f"  Task:       {args.task}")
    print(f"  Model:      {args.model}")
    print(f"  Server:     {args.host}:{args.port}")
    print(f"  Action Hz:  {args.action_hz} (rate actions are sent to robot)")
    print(f"  Infer Hz:   {inference_hz:.1f} (rate model is queried)")
    print(f"  Chunk:      {args.open_loop_steps}/{NUM_ACTIONS_CHUNK} actions open-loop")
    print(f"  Cameras:    {num_images} ({'third-person + wrist' if use_wrist else 'third-person only'})")
    print(f"  Proprio:    {not args.no_proprio}")
    print(f"  CenterCrop: {args.center_crop}")
    print("=" * 60)

    # === Step 1: Load OpenVLA-OFT model ========================================
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("[WARN] No GPU detected — inference will be very slow!")
    vla, processor, action_head, proprio_projector = load_oft_model(args.model, device)

    # Set number of input images on the vision backbone
    if hasattr(vla, "vision_backbone") and hasattr(vla.vision_backbone, "set_num_images_in_input"):
        vla.vision_backbone.set_num_images_in_input(num_images)

    # Verify unnorm_key exists in model's norm_stats
    if args.unnorm_key not in vla.norm_stats:
        alt_key = f"{args.unnorm_key}_no_noops"
        if alt_key in vla.norm_stats:
            args.unnorm_key = alt_key
        else:
            avail = list(vla.norm_stats.keys())
            print(f"[WARN] unnorm_key '{args.unnorm_key}' not found. Available: {avail}")
            args.unnorm_key = avail[0]
            print(f"[WARN] Using: {args.unnorm_key}")

    # === Step 2: Initialize ZED camera(s) ======================================
    cam_fps = max(args.action_hz, 15)
    zed, runtime = init_zed_camera(
        resolution=args.camera_resolution, fps=cam_fps,
        serial_number=args.third_person_serial, label="third-person",
    )
    zed_wrist, runtime_wrist = None, None
    if use_wrist:
        zed_wrist, runtime_wrist = init_zed_camera(
            resolution=args.camera_resolution, fps=cam_fps,
            serial_number=args.wrist_serial, label="wrist",
        )

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
    dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    dummy_wrist = Image.new("RGB", (224, 224), color=(128, 128, 128)) if use_wrist else None
    dummy_proprio = np.zeros(PROPRIO_DIM, dtype=np.float64)
    _ = predict_action_chunk(
        vla, processor, action_head,
        proprio_projector if not args.no_proprio else None,
        dummy_img, dummy_proprio, args.task, args.unnorm_key, device,
        wrist_image_pil=dummy_wrist,
    )
    print("[OK] Warm-up done")

    # === Step 6: Main control loop =============================================
    print()
    print("=" * 60)
    print("RUNNING — press Ctrl+C or type 'q'+Enter to stop")
    print("=" * 60)
    print()

    step_count = 0
    chunk_count = 0
    gripper_open = True
    video_writer = None
    GRIPPER_OPEN = 0.08   # Franka Hand max width [m]
    GRIPPER_CLOSE = 0.0
    GRIPPER_SPEED = 0.1   # finger speed [m/s]

    while running:
        # ── Check keyboard ────────────────────────────────────────────────
        key = check_keyboard()
        if key == "q":
            print("\n\n[DONE] 'q' pressed — quitting")
            break

        # ── Grab image from third-person ZED camera ─────────────────────
        image_pil = grab_rgb_image(zed, runtime)
        if image_pil is None:
            print("\n[WARN] Failed to grab third-person frame, retrying ...")
            continue

        # ── Grab image from wrist ZED camera (if available) ───────────
        wrist_pil = None
        if use_wrist:
            wrist_pil = grab_rgb_image(zed_wrist, runtime_wrist)
            if wrist_pil is None:
                print("\n[WARN] Failed to grab wrist frame, retrying ...")
                continue

        # ── Crop third-person image ───────────────────────────────────
        if args.crop is not None:
            cx, cy, cw, ch = args.crop
            image_pil = image_pil.crop((cx, cy, cx + cw, cy + ch))
        else:
            w, h = image_pil.size
            if w != h:
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                image_pil = image_pil.crop((left, top, left + side, top + side))

        # ── Crop wrist image ─────────────────────────────────────────
        if wrist_pil is not None:
            if args.wrist_crop is not None:
                cx, cy, cw, ch = args.wrist_crop
                wrist_pil = wrist_pil.crop((cx, cy, cx + cw, cy + ch))
            else:
                w, h = wrist_pil.size
                if w != h:
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    wrist_pil = wrist_pil.crop((left, top, left + side, top + side))

        # Apply sqrt(0.9) center crop to match OFT training augmentation
        if args.center_crop:
            crop_factor = math.sqrt(0.9)
            w, h = image_pil.size
            new_w, new_h = int(w * crop_factor), int(h * crop_factor)
            left, top = (w - new_w) // 2, (h - new_h) // 2
            image_pil = image_pil.crop((left, top, left + new_w, top + new_h))
            if wrist_pil is not None:
                w, h = wrist_pil.size
                new_w, new_h = int(w * crop_factor), int(h * crop_factor)
                left, top = (w - new_w) // 2, (h - new_h) // 2
                wrist_pil = wrist_pil.crop((left, top, left + new_w, top + new_h))

        # Show camera feed(s)
        bgr_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("Third-Person Camera", bgr_frame)
        if wrist_pil is not None:
            bgr_wrist = cv2.cvtColor(np.array(wrist_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow("Wrist Camera", bgr_wrist)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n\n[DONE] 'q' in camera window — quitting")
            break

        # Save frame to video (third-person only)
        if args.save_video is not None:
            if video_writer is None:
                h_f, w_f = bgr_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(args.save_video, fourcc, int(inference_hz), (w_f, h_f))
                print(f"[OK] Recording video to {args.save_video} ({w_f}x{h_f} @ {inference_hz:.0f} fps)")
            video_writer.write(bgr_frame)

        # ── Build proprioceptive state ────────────────────────────────────
        state = client.get_robot_state()
        if state["error"]:
            print(f"\n[ERROR] Robot error: {state['error']}")
            break

        T_current = pose16_to_mat(state["pose"])
        eef_pos = T_current[:3, 3]
        eef_rot = rotation_matrix_to_axis_angle(T_current[:3, :3])
        # LIBERO proprio format: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
        # Approximate Franka gripper as two symmetric fingers
        gripper_width = 0.08 if gripper_open else 0.0
        proprio = np.concatenate([eef_pos, eef_rot, [gripper_width / 2, gripper_width / 2]])

        # ── Run OFT inference → action chunk ──────────────────────────────
        t_infer = time.time()
        action_chunk = predict_action_chunk(
            vla, processor, action_head,
            proprio_projector if not args.no_proprio else None,
            image_pil, proprio, args.task, args.unnorm_key, device,
            wrist_image_pil=wrist_pil,
        )
        infer_ms = (time.time() - t_infer) * 1000
        chunk_count += 1

        print(f"\n  [Chunk {chunk_count}] inference: {infer_ms:.0f}ms  "
              f"shape: {action_chunk.shape}")

        # ── Execute action chunk open-loop ────────────────────────────────
        n_exec = min(args.open_loop_steps, len(action_chunk))
        for i in range(n_exec):
            if not running:
                break

            action_start = time.time()
            step_count += 1

            action = action_chunk[i]
            print(f"    [{i+1}/{n_exec}] raw: {action}")

            eef_delta = action[:6] * args.action_scale
            gripper_action = action[6]

            # Read current pose (incremental update from actual state)
            state = client.get_robot_state()
            if state["error"]:
                print(f"\n[ERROR] Robot error: {state['error']}")
                running = False
                break

            T_current = pose16_to_mat(state["pose"])
            T_target = apply_eef_delta(T_current, eef_delta)
            ok, msg = client.set_ee_target(mat_to_pose16(T_target))
            if not ok:
                print(f"\n[ERROR] SetEETarget failed: {msg}")
                running = False
                break

            # Gripper control
            want_closed = gripper_action > args.gripper_threshold
            want_open = gripper_action < -args.gripper_threshold
            if want_closed and gripper_open:
                client.set_gripper_target(GRIPPER_CLOSE, GRIPPER_SPEED)
                gripper_open = False
            elif want_open and not gripper_open:
                client.set_gripper_target(GRIPPER_OPEN, GRIPPER_SPEED)
                gripper_open = True

            # Regulate frequency
            elapsed = time.time() - action_start
            sleep_t = action_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            actual_hz = 1.0 / max(time.time() - action_start, 1e-6)
            p = T_current[:3, 3]
            sys.stdout.write(
                f"\r[Step {step_count:>6} | Chunk {chunk_count}:{i+1}/{n_exec}] "
                f"Hz:{actual_hz:>5.1f} | "
                f"pos=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}] | "
                f"grip={'OPEN' if gripper_open else 'CLOSED'} | "
                f"delta=[{eef_delta[0]:+.4f} {eef_delta[1]:+.4f} {eef_delta[2]:+.4f}]    "
            )
            sys.stdout.flush()

            # Check keyboard between chunk actions
            key = check_keyboard()
            if key == "q":
                print("\n\n[DONE] 'q' pressed")
                running = False
                break

    # === Cleanup ==============================================================
    print(f"\n\nOpenVLA-OFT control ended.")
    print(f"Total steps: {step_count}, Chunks: {chunk_count}")
    try:
        client.stop()
        client.close()
    except Exception:
        pass
    try:
        zed.close()
    except Exception:
        pass
    if zed_wrist is not None:
        try:
            zed_wrist.close()
        except Exception:
            pass
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {args.save_video}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
