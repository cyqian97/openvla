# OpenVLA Franka Client

Use a pretrained OpenVLA model to control a Franka FR3 robot arm in real time
via a ZED stereo camera.

## Architecture

```
┌──────────────────── Client (this script, GPU workstation) ────────────────────┐
│                                                                               │
│  ZED Camera ──RGB frame──► OpenVLA (7B) ──action[7]──► apply_eef_delta()      │
│                              on GPU                         │                 │
│                                                     4x4 target pose           │
│                                                             │                 │
│                                            FrankaDirectClient.set_ee_target() │
│                                                             │                 │
└─────────────────────────────────────────────────────────────┼─────────────────┘
                                                              │ gRPC
┌─────────────────── Server (NUC, Docker) ────────────────────┼─────────────────┐
│                                                             ▼                 │
│  franka_server_cartesian    PD pose error → Cartesian velocity → libfranka    │
│  (C++, 1 kHz RT loop)      joint impedance controller handles IK & gravity    │
│                                                             │                 │
└─────────────────────────────────────────────────────────────┼─────────────────┘
                                                              │ 1 kHz torques
                                                              ▼
                                                        Franka FR3
```

The client runs at 10 Hz (configurable). The server internally runs at 1 kHz,
interpolating toward whatever EE pose target was last received.

## Prerequisites

1. **franka_server_cartesian** running on the NUC (inside Docker):

   ```bash
   docker exec <container> bash /app/droid/franka_direct/build.sh
   docker exec <container> bash /app/droid/franka_direct/launch_server_cartesian.sh
   ```

2. **ZED camera** connected via USB to the GPU workstation.

3. **GPU** available (CUDA). The 7B model needs ~15 GB VRAM in bf16.

4. **Python packages** installed on the workstation:

   ```bash
   # From the openvla repo root:
   pip install -e .

   # Flash Attention (optional but recommended — speeds up inference):
   pip install flash-attn --no-build-isolation
   # If flash-attn fails to install, edit load_openvla() in the script:
   #   change attn_implementation="flash_attention_2" to "eager"

   # ZED SDK Python wrapper:
   # First install the ZED SDK: https://www.stereolabs.com/developers/release
   # Then install the Python API using the bundled script (NOT pip):
   python /usr/local/zed/get_python_api.py
   # Note: "pip install pyzed-sl" does NOT work — the package is not on PyPI.
   # The get_python_api.py script builds a wheel for your specific Python
   # version and CUDA setup. If the path above doesn't exist, find it with:
   #   find /usr/local/zed -name "get_python_api.py"

   # gRPC client and codegen tools:
   pip install grpcio grpcio-tools

   # NumPy 1.x (torch 2.2.0 is incompatible with NumPy 2.x):
   pip install "numpy<2"
   ```

5. **gRPC Python stubs** generated in `~/cyqian/droid/franka_direct/python/`:

   ```bash
   bash ~/cyqian/droid/franka_direct/python/generate_stubs.sh
   ```

## Usage

```bash
python scripts/openvla_franka_client.py \
    --task "pick up the red ball" \
    --host 192.168.1.6 --port 50052 \
    --hz 10
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | *(required)* | Natural language task instruction |
| `--model` | `openvla/openvla-7b` | HuggingFace model name or local checkpoint path |
| `--unnorm_key` | `None` (auto) | Dataset key for action un-normalization. Leave as default for base model; set to your dataset name for fine-tuned checkpoints |
| `--host` | `192.168.1.6` | franka_server_cartesian IP |
| `--port` | `50052` | franka_server_cartesian gRPC port |
| `--hz` | `10` | Control loop frequency (Hz) |
| `--no_reset` | `False` | Skip homing the robot on startup |
| `--reset_speed` | `0.2` | Joint move speed factor [0..1] for the homing move |
| `--camera_resolution` | `HD720` | ZED resolution: `HD2K`, `HD1080`, `HD720`, `VGA` |
| `--crop` | `None` (center-square) | Crop region `X Y W H` in pixels. If not set, center-crops to a square to avoid 16:9→1:1 distortion |
| `--save_video` | `None` | Save camera feed to a video file (e.g. `output.mp4`). If not set, no video is saved |
| `--action_scale` | `1.0` | Multiplier applied to the 6-DoF EEF delta |
| `--gripper_threshold` | `0.5` | Gripper action value above which the gripper closes (below negative threshold it opens) |

### Stopping

- **Ctrl+C** — graceful stop (SIGINT handler sets `running=False`)
- **q + Enter** — quit via keyboard polling

## Code Structure

The script is organized into helpers at the top and a sequential `main()` at
the bottom.

### Helper functions (lines 53–190)

**Rotation & pose math** (lines 55–74):
`rot_x`, `rot_y`, `rot_z` build 3x3 rotation matrices. `pose16_to_mat` and
`mat_to_pose16` convert between libfranka's 16-element column-major format and
4x4 numpy arrays. These match the conventions in `simple_pose_direct.py`.

**`load_openvla()`** (lines 86–104):
Downloads (on first run, ~15 GB) and loads the OpenVLA model and its HF
processor onto the GPU in bf16 precision. Uses flash attention if available.

**`predict_action()`** (lines 107–118):
Wraps a single forward pass: builds the text prompt from the task label,
tokenizes image + text, runs `vla.predict_action()`, returns a 7-element numpy
array `[dx, dy, dz, droll, dpitch, dyaw, gripper]`. Decorated with
`@torch.inference_mode()` to disable gradient tracking.

**`init_zed_camera()` / `grab_rgb_image()`** (lines 121–159):
Thin wrappers around the ZED SDK. The camera is opened with depth disabled
(RGB only). `grab_rgb_image` retrieves the left view, converts BGRA→RGB, and
returns a PIL Image.

**`apply_eef_delta()`** (lines 162–190):
Core pose update. Given the current 4x4 EE pose and a 6-DoF delta:
- Translation `[dx, dy, dz]` is added in the **base frame**.
- Rotation `[droll, dpitch, dyaw]` is applied as extrinsic XYZ (pre-multiply
  `Rz @ Ry @ Rx @ R_current`), also in the base frame.

Returns the new 4x4 target pose.

### main() control flow (lines 195–390)

```
Step 1: Load OpenVLA model onto GPU
Step 2: Open ZED camera
Step 3: Connect to franka_server_cartesian via gRPC, wait for ready
Step 4: Reset robot to home joint configuration (blocking RPC)
Step 5: Warm-up inference (dummy image, primes CUDA kernels)
Step 6: Main control loop
        ┌─────────────────────────────────────────────────┐
        │  check keyboard (non-blocking)                  │
        │  grab RGB image from ZED                        │
        │  run OpenVLA inference → action[7]              │
        │  scale action[:6] by --action_scale             │
        │  get current robot state (pose, gripper, etc.)  │
        │  apply_eef_delta(current_pose, action[:6])      │
        │  client.set_ee_target(new_pose)                 │
        │  gripper: open/close based on action[6]         │
        │  sleep to regulate to --hz                      │
        │  print status line                              │
        └─────────────────────────────────────────────────┘
Cleanup: stop server control loop, close camera
```

### Key design decisions

- **No IK solver on the client.** The Cartesian server handles IK internally
  via libfranka's joint impedance controller. The client only sends target EE
  poses. This removes the dependency on `dm-control`/`dm-robotics`/`mujoco`.

- **Incremental pose updates.** Each step reads the *actual* measured pose
  from the server and adds the OpenVLA delta on top of it. This avoids drift
  from accumulating deltas on a stale reference.

- **Frequency regulation.** The loop sleeps for the remainder of each period
  to maintain the target Hz. If inference takes longer than the budget, the
  loop simply runs slower (no frame skipping).

## Tuning Tips

- **`--action_scale`**: The base `openvla-7b` was trained on BridgeData V2
  (WidowX robot). Action magnitudes will not match the Franka workspace.
  Start with a small value (e.g., `0.1`) and increase.

- **Gripper convention**: The sign of `action[6]` (positive = close vs open)
  depends on the training dataset. If the gripper behaves backwards, negate
  the threshold logic or flip the sign.

- **Fine-tuning**: For reliable performance on your specific tasks, fine-tune
  OpenVLA on demonstrations collected with your Franka + ZED setup. Use
  `--model /path/to/finetuned/checkpoint` and set `--unnorm_key` to match
  the dataset name used during training.
