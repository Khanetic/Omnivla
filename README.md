# OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![ICRA 2026](https://img.shields.io/badge/ICRA-2026-blue)](https://omnivla-nav.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2509.19480-b31b1b)](https://arxiv.org/abs/2509.19480)
[![Project Page](https://img.shields.io/badge/Project-Page-purple)](https://omnivla-nav.github.io)

**Noriaki Hirose**¹², **Catherine Glossop**¹, **Dhruv Shah**³, **Sergey Levine**¹

¹ UC Berkeley (BAIR) &nbsp;|&nbsp; ² Toyota Motor North America &nbsp;|&nbsp; ³ Princeton University

*IEEE International Conference on Robotics and Automation (ICRA) 2026*

---

## What is OmniVLA?

OmniVLA is a navigation foundation model that accepts goals in **any combination of three modalities** and outputs velocity commands to drive a robot:

| Input Modality | Example |
|---|---|
| 🗣️ **Language** | `"move toward the blue trash bin"` |
| 📍 **GPS pose** | Latitude / longitude target |
| 🖼️ **Goal image** | Photo of the destination |

It is built on top of [OpenVLA-OFT](https://openvla-oft.github.io/) (a 7B-parameter VLA backbone) and trained on **9,500 hours** of navigation data across 10 robot platforms.

### Architecture

```
Camera image(s) ──► EfficientNet / ViT encoder ──┐
GPS goal pose   ──► Pose projector               ├──► Transformer ──► Waypoints
Goal image      ──► CLIP encoder                 │     (VLA)          (linear vel,
Language prompt ──► CLIP text encoder ───────────┘                    angular vel)
```

OmniVLA uses **randomised modality dropout** during training so it gracefully handles missing inputs at inference time.

---

## Models

| Model | Params | VRAM | Best for |
|---|---|---|---|
| **OmniVLA** (full) | 7.5 B | ~15 GB | Max accuracy, multi-GPU server |
| **OmniVLA-edge** | ~50 M | < 2 GB | Real-time on edge devices / consumer GPUs |

---

## Project Structure

```
OmniVLA/
├── inference/
│   ├── run_omnivla.py          # Full model inference (language/pose/image goals)
│   ├── run_omnivla_edge.py     # Edge model inference
│   ├── carla_omnivla.py        # CARLA bridge — bike following
│   ├── carla_omnivla_person.py # CARLA bridge — person/pedestrian following
│   ├── utils_policy.py         # Model loading & image transforms
│   ├── current_img.jpg         # Sample current camera image
│   └── goal_img.jpg            # Sample goal image
├── vla-scripts/
│   ├── train_omnivla.py        # Training from OpenVLA checkpoints
│   └── train_omnivla_dataset.py# Training on full multi-dataset corpus
├── vint_train/                 # ViNT backbone (used by OmniVLA-edge)
├── prismatic/                  # Vision-language backbone components
├── config_nav/                 # Dataset & training config YAMLs
├── omnivla-edge/               # Downloaded edge checkpoint (omnivla-edge.pth)
├── SETUP.md                    # Conda environment setup
└── README_CARLA.md             # CARLA integration guide
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/NHirose/OmniVLA.git
cd OmniVLA
```

### 2. Create conda environment
```bash
conda create -n omnivla python=3.10 -y
conda activate omnivla
```

### 3. Install PyTorch
```bash
# PyTorch bundles its own CUDA runtime — no separate CUDA toolkit required
pip install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install the OmniVLA package
```bash
pip install -e .
```

### 5. Fix CLIP dependency
```bash
# openai-clip 1.0.1 has a broken pkg_resources import — replace with official build
pip uninstall openai-clip -y
pip install git+https://github.com/openai/CLIP.git
```

### 6. (Optional) Flash Attention — required for full model training only
```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
# Note: compiles from source, takes ~15 minutes
```

---

## Inference

### OmniVLA-edge (recommended — runs on any GPU ≥ 4 GB)

**Download checkpoint (~414 MB):**
```bash
git clone https://huggingface.co/NHirose/omnivla-edge
# If git-lfs is not installed, download the weights directly:
wget -O omnivla-edge/omnivla-edge.pth \
    "https://huggingface.co/NHirose/omnivla-edge/resolve/main/omnivla-edge.pth"
```

**Run inference on sample images:**
```bash
conda activate omnivla
python inference/run_omnivla_edge.py
# Output: inference/1_ex_omnivla_edge.jpg
xdg-open inference/1_ex_omnivla_edge.jpg
```

**Change goal modality** — edit `inference/run_omnivla_edge.py` around line 425:
```python
pose_goal  = False   # use GPS pose as goal
image_goal = False   # use goal image
lan_prompt = True    # use language prompt  ← default
```

---

### OmniVLA full model (requires ~15 GB VRAM)

**Download checkpoints:**
```bash
git clone https://huggingface.co/NHirose/omnivla-original
git clone https://huggingface.co/NHirose/omnivla-original-balance   # data-balanced variant
git clone https://huggingface.co/NHirose/omnivla-finetuned-cast      # CAST fine-tune
```

**Run inference:**
```bash
python inference/run_omnivla.py
# Output: inference/1_ex.jpg
```

**Use CAST fine-tuned weights** — edit `InferenceConfig` in `run_omnivla.py`:
```python
vla_path = "./omnivla-finetuned-cast"
step     = <checkpoint_step>
```

---

## CARLA Simulator Integration

Runs OmniVLA-edge as the driving policy inside the CARLA autonomous driving simulator.

### Step 1 — Start CARLA (with GUI)
```bash
DISPLAY=:1 ~/CARLA_0.9.16/CarlaUE4.sh -windowed -ResX=1280 -ResY=720 &
# Wait ~20 seconds for CARLA to fully load
```

Headless (no window, faster):
```bash
DISPLAY=:1 ~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen &
```

### Step 2a — Bike following
```bash
conda activate omnivla
cd ~/OmniVLA
python inference/carla_omnivla.py
```

Spawns a **Vespa scooter** 15 m ahead. OmniVLA steers the Tesla toward it using live pose + language goal.

### Step 2b — Person following (recommended — better speed match)
```bash
python inference/carla_omnivla_person.py
```

Spawns a **walking pedestrian** 8 m ahead using CARLA's WalkerAIController at 1.2 m/s. This is a significantly better match for OmniVLA's training distribution (walking-speed robots) — the Tesla can keep pace without throttle scaling hacks.

### Step 2c — Car driving without following a person or bike
```bash
python inference/carla_omnivla_lane.py
```

Spawns only the **ego car + background traffic**. OmniVLA uses a **lane waypoint ahead** as the goal pose, so it drives forward in-lane without spawning or following any pedestrian or bicycle.

Press **Ctrl+C** to stop and clean up all spawned actors.

### What happens
- Spawns a **Tesla Model 3** ego vehicle in Town01 (ClearNoon weather)
- Spawns **19 NPC vehicles** driving on autopilot for a realistic scene
- OmniVLA reads the front camera at **3 Hz** and steers the car
- Spectator camera follows the ego vehicle in **3rd-person chase view**
- Each step is saved to `inference/carla_run/`, `inference/carla_run_person/`, or `inference/carla_run_lane/`

### Make a video
```bash
# Bike run
ffmpeg -framerate 3 -i inference/carla_run/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_carla_bike.mp4

# Person run
ffmpeg -framerate 3 -i inference/carla_run_person/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_carla_person.mp4

# Lane-driving run
ffmpeg -framerate 3 -i inference/carla_run_lane/step_%04d.jpg \
    -c:v libx264 -pix_fmt yuv420p omnivla_carla_lane.mp4
```

### CARLA config options
```python
TOWN          = "Town01"                  # Town01–Town07
TICK_HZ       = 3                         # inference frequency (Hz)
LANGUAGE_GOAL = "follow the person ahead" # language instruction
MAX_STEPS     = 9999                      # steps before auto-stop
```

---

## Training

### Sample training (understand data format first)

```bash
# Download MBRA codebase
cd ..
git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
cd OmniVLA

# Download MBRA model
git clone https://huggingface.co/NHirose/MBRA/

# Train from OpenVLA checkpoints (fill in N_GPUS and BATCH_SIZE)
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
    vla-scripts/train_omnivla.py \
    --vla_path openvla/openvla-7b \
    --dataset_name omnivla \
    --num_images_in_input 2 \
    --batch_size <BATCH_SIZE> \
    --wandb_entity "<your_entity>" \
    --wandb_project "omnivla"
```

> Minimum: 20 GB GPU VRAM even in debug mode. Full training used 8× H100 80 GB.

### Fine-tune from OmniVLA checkpoints
```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
    vla-scripts/train_omnivla.py \
    --vla_path ./omnivla-original \
    --dataset_name omnivla \
    --num_images_in_input 2 \
    --batch_size <BATCH_SIZE> \
    --wandb_entity "<your_entity>" \
    --wandb_project "omnivla"
```

### Full multi-dataset training (GNM + LeLaN + FrodoBots + BDD + CAST)

1. Download datasets: [GNM](https://github.com/robodhruv/visualnav-transformer), [LeLaN](https://huggingface.co/datasets/NHirose/LeLaN_dataset_NoMaD_traj/tree/main), [FrodoBots](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA), [CAST](https://openvla-oft.github.io/), [BDD](https://huggingface.co/datasets/NHirose/BDD_OmniVLA)
2. Clone lerobot for FrodoBots dataloader: `git clone https://github.com/huggingface/lerobot.git`
3. Set data paths in `config_nav/mbra_and_dataset_config.yaml`
4. Run:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
    vla-scripts/train_omnivla_dataset.py \
    --vla_path ./omnivla-original \
    --dataset_name omnivla \
    --wandb_entity "<your_entity>" \
    --wandb_project "omnivla"
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'pkg_resources'` | `pip uninstall openai-clip -y && pip install git+https://github.com/openai/CLIP.git` |
| `TOMLDecodeError: Unclosed array` | Missing comma in `pyproject.toml` line 66 — already fixed in this repo |
| `Model weights not found` | Re-run the `wget` download command |
| CARLA timeout | `pkill -f CarlaUE4` then restart |
| No car in CARLA window | Wait for `Spawned ego vehicle` in the terminal — spectator auto-follows |
| OOM on full model | Use OmniVLA-edge instead (only ~50 M params) |

---

## Citation

```bibtex
@misc{hirose2025omnivla,
    title   = {OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation},
    author  = {Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
    year    = {2025},
    eprint  = {2509.19480},
    archivePrefix = {arXiv},
    primaryClass  = {cs.RO},
    url     = {https://arxiv.org/abs/2509.19480},
}
```

---

## Acknowledgements

Built on top of [OpenVLA-OFT](https://openvla-oft.github.io/). CARLA integration by local setup.
