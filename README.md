[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kOYURy8TVGIryGueFkyb3NKfmZQbl2bw#scrollTo=X0bdVfpHhSpK)

# EyeOfAI: Real-Time Multi-Modal Visual Intelligence and Autonomous Perception System

## System Status and Release Notes
This repository represents **Version 1.0 (v1.0)** of the EyeOfAI System. The system is under active development, architectural refinement, and continuous optimization. 

The initial release of EyeOfAI is the result of 6 months of continuous engineering, rigorous testing, structural refactoring, model backbone evaluation, and edge-case resolution. Over this development period, multiple deep learning architectures were evaluated, system bottlenecks were systematically eliminated, multi-threaded pipelines were re-architected, and adaptive self-healing mechanisms were integrated to achieve maximum runtime stability under intensive visual loads.

---

## Executive Summary and Purpose
EyeOfAI is an enterprise-grade, multi-modal perception framework designed for real-time video analysis across unconstrained environments. Standard computer vision pipelines frequently suffer from rigid vocabulary limits, high latency when combining visual sub-tasks, and frame drops under varying environmental lighting or hardware thermal throttling.

EyeOfAI solves these fundamental challenges through a unified multi-threaded framework that integrates:
1. Open-Vocabulary Object Detection (Detic with Swin-B backbone).
2. High-Precision Multi-Object Tracking (ByteTrack).
3. Spatio-Temporal Human Action Recognition (PyTorchVideo MViT-B 16x4 on Kinetics-700).
4. Asynchronous Facial Intelligence (DeepFace: Emotion, Age, Gender).
5. Dynamic Environmental Adaptation (CLAHE-based Scene Calibrator).
6. Adaptive Resource Allocation (Dynamic Performance Governor).
7. Autonomous Fault Management (Self-Healing Runtime Engine).

---

## Technical Component Rationale

### Open-Vocabulary Object Detection: Detic + Swin-B
- Selection Rationale: Standard object detectors are limited to small fixed category sets (e.g., COCO's 80 classes). Detic (Detection with Image-level Labels) leverages CLIP embeddings mapped to LVIS categories (1200+ classes), enabling open-vocabulary detection without requiring fine-tuning for new concepts.
- Backbone Choice: Swin-B (Swin Transformer Base) provides feature extraction capabilities across varied object scales, crucial for dense multi-object scenes.

### Multi-Object Tracking: ByteTrack
- Selection Rationale: ByteTrack preserves object identities across frame sequences by utilizing high-confidence and low-confidence detection boxes during association. This prevents identity switches caused by transient occlusions or fast motion.

### Action Recognition: PyTorchVideo MViT-B 16x4 on Kinetics-700
- Selection Rationale: Spatio-temporal motion analysis requires understanding frame sequences over time. MViT-B (Multiscale Vision Transformer Base) trained on Kinetics-700 evaluates 700 fine-grained action categories.
- Operational Offloading: To prevent VRAM contention with Detic on single-GPU hardware (such as NVIDIA T4), MViT inference is offloaded to optimized CPU worker threads, keeping GPU memory dedicated to object detection.

### Facial Intelligence: DeepFace Asynchronous Pool
- Selection Rationale: DeepFace provides comprehensive facial trait estimation. Using RetinaFace as the primary alignment backend ensures robust detection even under off-angle head poses.
- Multi-Threaded Execution: Facial analysis runs asynchronously in background worker pools with configurable frame-skip intervals, preventing main loop blocking.

---

## System Architecture and Behind-The-Scenes Pipeline

EyeOfAI operates on a multi-threaded, asynchronous processing graph designed for minimal end-to-end latency.

```
[ Video Input Stream ]
          │
          ▼
[ Scene Calibrator ] ─── (Real-Time CLAHE & HSV Brightness Normalization)
          │
          ▼
[ Performance Governor ] ─── (Monitors Target FPS & Adapts Resolution/Intervals)
          │
          ▼
[ Detic Object Detector ] ─── (Swin-B Open-Vocabulary Feature Extraction)
          │
          ▼
[ ByteTrack Association ] ─── (Kalman Filter & IoU Assignment Matrix)
          │
      ┌───┴──────────────────────────────┐
      ▼                                  ▼
[ Person Attribute Store ]    [ Action Recognizer Pool ]
  - Asynchronous DeepFace       - 16-Frame Temporal Queue
  - Emotion Smoothing Window    - MViT-B 16x4 CPU Inference
  - Age / Gender Timers         - Kinetics-700 Action Mapping
      │                                  │
      └───────────────┬──────────────────┘
                      ▼
          [ Visual Output & HUD ]
```

### Execution Flow Step-by-Step
1. Frame Capture and Preprocessing: Frames are ingested from standard video streams or files and resized according to the active resolution settings.
2. Scene Calibration: The `SceneCalibrator` calculates frame brightness in HSV space using an Exponential Moving Average (EMA). It applies CLAHE on the L channel in LAB color space and modifies object detection confidence thresholds dynamically.
3. Open-Vocabulary Detection: Preprocessed frames pass to Detic (Swin-B). Detic extracts bounding boxes, confidence scores, and LVIS class labels.
4. Tracking and Identity Assignment: Detic bounding boxes pass to ByteTrack. Tracks are updated, assigned unique IDs, and persistent trajectories are maintained.
5. Asynchronous Facial Intelligence: Tracked bounding boxes corresponding to human detections are cropped with context padding. The cropped region is passed to the `AsyncDeepFacePool`. Face results (emotion, age, gender) are stored in the `PersonAttributeStore`. Emotion predictions undergo majority-voting smoothing across a rolling window.
6. Spatio-Temporal Action Recognition: Cropped person trajectories accumulate in per-track frame queues. Once a queue reaches 16 frames (with stride 4), a preprocessed tensor of shape (1, 3, 16, 224, 224) is passed to `ActionRecognizer` on CPU. The top predicted action is written back to the store.
7. Performance Governor Feedback Loop: On every processing cycle, `PerformanceGovernor` computes rolling FPS. If FPS drops below the target (e.g., 25 FPS), the governor transitions between three operating modes:
   - Level 0 (FULL): Native resolution, standard analysis intervals.
   - Level 1 (REDUCED_FACE): 75% resolution scaling, 2x analysis intervals.
   - Level 2 (LITE): 50% resolution scaling, 4x analysis intervals.
8. Self-Healing Exception Handling: All component operations are wrapped in `self_heal` contexts. If an exception or memory pressure occurs, the framework clears GPU cache (`torch.cuda.empty_cache()`), forces garbage collection, and executes recovery procedures without terminating the pipeline.

---

## Complete Feature Matrix

- Open-Vocabulary Detection: Detection across 1200+ LVIS categories without domain re-training.
- Persistent Track Maintenance: ByteTrack tracking with configurable track buffers and matching thresholds.
- Fine-Grained Action Recognition: MViT-B 16x4 evaluation on Kinetics-700 (700 categories).
- Temporal Frame Queue: Automatic sampling and center-cropping (224x224) for action tensor preparation.
- Facial Attribute Estimation: Multi-attribute face analysis covering dominant emotion, age bracket, and gender.
- Temporal Emotion Smoothing: Majority-vote filtering across rolling prediction windows to eliminate flickering predictions.
- Dynamic Scene Normalization: Adaptive CLAHE combined with real-time HSV brightness tracking and confidence score offsets.
- Adaptive Performance Governor: Dynamic three-level degradation and recovery engine based on real-time FPS feedback.
- Asynchronous Worker Pools: Non-blocking ThreadPoolExecutors for DeepFace and MViT processing tasks.
- Self-Healing Runtime: Automated exception interception, CUDA cache clearing, and soft-restart capability.
- Real-Time Telemetry and Monitoring: System stats including GPU VRAM usage, CPU usage, RAM allocation, FPS history, and event logs.
- Video Demonstration Support: Dedicated directory structure for previewing test execution on sample videos.

---

## Directory Structure

The repository contains real files and module structures structured as follows:

```
EyeOfAI-System/
├── EyeOfAI_System.py
├── EyeOfAI_System.ipynb
├── requirements.txt
├── README.md
├── Videos/
│   ├── demo_sample_01.mp4
│   └── demo_sample_02.mp4
├── detectron2/
│   ├── setup.py
│   └── detectron2/
├── Detic/
│   ├── configs/
│   │   └── Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml
│   ├── datasets/
│   │   └── metadata/
│   │       ├── lvis_v1_clip_a+cname.npy
│   │       ├── o365_clip_a+cnamefix.npy
│   │       ├── oid_clip_a+cname.npy
│   │       └── coco_clip_a+cname.npy
│   └── models/
│       └── Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
├── ByteTrack/
│   ├── setup.py
│   └── yolox/
│       └── tracker/
│           └── byte_tracker.py
└── mvit_k700/
    └── MVIT_B_16x4_K700.pyth
```

---

## Demonstration Videos
Sample test videos used during system verification and benchmark evaluation are stored in the `Videos/` directory. Users can review these demo videos to observe real-time detection, tracking, action recognition, and facial intelligence prior to running the system on local hardware.

---

## Prerequisites and Installation

### Hardware Requirements
- GPU: NVIDIA GPU with CUDA support (8 GB VRAM minimum, 15+ GB recommended for Swin-B model).
- CPU: 4+ physical cores.
- RAM: 16 GB minimum.

### Software Requirements
- Linux (Ubuntu 20.04/22.04 recommended) or Google Colab environment.
- Python 3.10+
- PyTorch 2.3.0 with CUDA 12.1 support.

### System Dependencies (`requirements.txt`)
Create a `requirements.txt` file with the following contents:

```
torch==2.3.0
torchvision==0.18.0
opencv-python-headless
Pillow
requests
pyyaml
cython
matplotlib
pycocotools
ninja
rich
psutil
deepface
tf-keras
python-jose
pytorchvideo
av>=9.0.0
loguru
lap
cython_bbox
thop
lvis
```

### Quick Start Guide

1. Clone the repository:
```bash
git clone https://github.com/Don-Youssef/EyeOfAI-System.git
cd EyeOfAI-System
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Execute the Python system pipeline:
```bash
python EyeOfAI_System.py
```

4. Alternatively, launch the Jupyter / Colab Notebook:
Open `EyeOfAI_System.ipynb` in Google Colab or your local Jupyter Lab instance.

---

## Configuration Reference (`EyeConfig`)

The system behavior is controlled via the `EyeConfig` dataclass in `EyeOfAI_System.py`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `detic_threshold` | float | `0.0` | Minimum confidence score for object detections |
| `vocabulary` | str | `"lvis"` | Open-vocabulary target dataset (`lvis`, `objects365`, `coco`, `openimages`) |
| `track_thresh` | float | `0.5` | Minimum confidence score to initiate or retain a ByteTrack track |
| `track_buffer` | int | `30` | Number of lost frames before a track ID is purged |
| `match_thresh` | float | `0.8` | IoU threshold for matching detections to existing tracks |
| `deepface_backend` | str | `"retinaface"` | Primary face detector backend for DeepFace analysis |
| `emotion_interval_sec` | float | `2.0` | Update interval in seconds for face emotion analysis |
| `age_gender_interval_sec` | float | `4.0` | Update interval in seconds for age and gender analysis |
| `deepface_emotion_smooth_window` | int | `5` | Window size for majority-vote emotion prediction smoothing |
| `action_clip_len` | int | `16` | Number of temporal frames per MViT action clip |
| `action_frame_stride` | int | `4` | Temporal sampling stride for MViT input tensors |
| `gov_target_fps` | float | `25.0` | Target FPS maintained by the Performance Governor |
| `async_workers` | int | `2` | Number of worker threads allocated for DeepFace execution |

---

## License and Acknowledgments
EyeOfAI integrates and builds upon key open-source research frameworks:
- Detectron2 & Detic: FAIR (Meta AI Research)
- PyTorchVideo & MViT: FAIR (Meta AI Research)
- ByteTrack: YOLOX / ByteTrack Authors
- DeepFace: Sefik Ilkin Serengil
