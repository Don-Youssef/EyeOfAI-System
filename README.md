# EyeOfAI System: Real-Time Open-Vocabulary Visual Intelligence Core (v1.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kOYURy8TVGIryGueFkyb3NKfmZQbl2bw#scrollTo=X0bdVfpHhSpK)
[![GitHub Repository](https://img.shields.io/badge/GitHub-EyeOfAI--System-181717?logo=github)](https://github.com/Don-Youssef/EyeOfAI-System?tab=readme-ov-file)
[![System Version](https://img.shields.io/badge/Version-1.0.0--beta-blue.svg)]()
[![Development Status](https://img.shields.io/badge/Status-Under_Active_Development-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0+cu121-red.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)]()

---

## Executive Overview & Version Release Notes

Welcome to the official repository of the **EyeOfAI System (Version 1.0)**. 

EyeOfAI is an enterprise-grade, multi-modal, real-time computer vision framework designed to perform open-vocabulary object detection across 21,000+ semantic categories, multi-object motion tracking, real-time facial telemetry (emotion classification, age estimation, gender identification), and temporal human action recognition (700 Kinetics classes) within a single unified processing pipeline.

### Engineering Development History
Version 1.0 represents the culmination of **6 months of continuous intensive engineering, architectural refactoring, and benchmark optimization**. Building a multi-model pipeline that concurrently executes large vision-language backbones, face analytics networks, and 3D spatio-temporal video transformers without introducing pipeline starvation or out-of-memory (OOM) faults presented significant technical hurdles. 

Over this 6-month development lifecycle:
* The core architecture underwent multiple complete structural overhauls to transition from synchronous execution to an asynchronous, multi-threaded producer-consumer pipeline.
* Deep neural models were iteratively swapped, benchmarked, and re-engineered to achieve an optimal trade-off between floating-point operations (FLOPs), VRAM memory footprint, and inference latency.
* Fallback control strategies and self-healing memory managers were developed to guarantee 24/7 continuous operational stability on mid-tier hardware (e.g., single NVIDIA T4 GPUs).

*Note: This repository contains Version 1.0 of the EyeOfAI System. It is under active development and optimization, with ongoing research focused on edge optimization, TensorRT quantization, and multi-camera stream synchronization.*

---

## System Purpose & Problem Statement

Standard computer vision systems suffer from four fundamental architectural limitations:
1. **Closed-Set Vocabulary Bottleneck**: Traditional object detectors (e.g., YOLO, standard Faster R-CNN) are restricted to fixed dataset classes (e.g., COCO 80 categories). They fail entirely when encountering novel or domain-specific objects in real-world environments.
2. **Environmental Illumination Sensitivity**: Standard models experience catastrophic drop-offs in detection recall when exposed to under-exposed (nighttime/shadows) or over-exposed (glare/intense backlighting) video feeds.
3. **Pipeline Latency & Hardware Contention**: Stacking multiple independent deep learning models (Detector + Tracker + Face Analyser + Action Recognizer) on a single GPU leads to compute starvation, thread blocking, and VRAM exhaustion.
4. **Brittle Fault Handling**: A single runtime error or memory spike in one sub-model typically crashes the entire multi-stage computer vision process.

### How EyeOfAI Solves These Challenges
EyeOfAI addresses these issues through a holistic engineering design:
* **Zero-Shot Open-Vocabulary Detection**: Integrates Detic (Detecting Twenty-Thousand Classes) backed by a Swin-Transformer (Swin-B) backbone and CLIP text-image embeddings, enabling real-time detection of over 21,000 object categories without model retraining.
* **Adaptive Scene Calibration**: Implements an automated luminance sensing engine that adjusts Contrast-Limited Adaptive Histogram Equalization (CLAHE) in the LAB color space dynamically based on frame brightness, while shifting detection confidence thresholds in real time.
* **CPU/GPU Compute Partitioning**: Offloads deep multi-modal feature extraction by running heavy VRAM-bound object detection (Detic Swin-B) on CUDA GPU, while executing facial analytics (DeepFace/RetinaFace) and 3D spatio-temporal action recognition (PyTorchVideo MViT-B 16x4) asynchronously on multi-threaded CPU workers.
* **Self-Healing Infrastructure & Dynamic Governor**: Incorporates a closed-loop system telemetry monitor, automated CUDA memory flushes, and a 3-tier performance governor that dynamically scales resolution, detection thresholds, and worker sampling rates to maintain target frame rates under heavy thermal load.

---

## Architectural Schema & Backend Operations

The EyeOfAI pipeline operates on a decoupled multi-threaded architecture. The diagram below details the data flow from raw video frame ingestion to final composite frame rendering:

```
+-----------------------------------------------------------------------------------+
|                                VIDEO STREAM INGESTION                             |
|                           (File / Camera / RTSP / Colab)                          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              ADAPTIVE SCENE CALIBRATOR                            |
|  - HSV Luminance Sensing (EMA Brightness Calculation)                             |
|  - Dynamic CLAHE Enhancement on LAB L-Channel                                     |
|  - Detection Confidence Offset Calculation                                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             PERFORMANCE GOVERNOR GATE                             |
|  - Evaluates Rolling Average FPS against Target (e.g., 25 FPS)                    |
|  - Selects Quality Level: FULL (L0) | REDUCED_FACE (L1) | LITE (L2)                 |
|  - Applies Dynamic Frame Downscaling & Skip Ratios                                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        DETIC SWIN-B OPEN-VOCABULARY DETECTOR                      |
|                                    (CUDA GPU)                                     |
|  - LVIS / Objects365 / OpenImages / COCO / ImageNet-21k Open-Vocabulary           |
|  - Extracts Bounding Boxes, Class Identifiers, and Confidence Scores              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            BYTETRACK MULTI-OBJECT TRACKER                         |
|  - Kalman Filtering & Hungarian Algorithm Data Association                        |
|  - Assigns Persistent Track IDs across Temporal Frames                            |
+-----------------------------------------------------------------------------------+
                                          |
                                          +-----------------------+
                                          |                       |
                                          v                       v
+---------------------------------------------------+ +-----------------------------+
|          ASYNC FACIAL ANALYTICS WORKERS           | |  ASYNC ACTION RECOGNIZER    |
|                     (CPU Pool)                    | |         (CPU Pool)          |
|  - Face Extraction (RetinaFace Backend)           | | - PyTorchVideo MViT-B 16x4  |
|  - Emotion Recognition + Majority Voting          | | - Kinetics-700 Actions      |
|  - Age Estimation & Gender Identification         | | - BCTHW Tensor Formatting   |
+---------------------------------------------------+ +-----------------------------+
                                          |                       |
                                          +-----------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            PERSON ATTRIBUTE STORE (CACHE)                         |
|  - Non-blocking Thread-Safe Attribute Retrieval (Emotion, Age, Gender, Action)   |
|  - Automatic Garbage Collection for Terminated Track IDs                          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             HUD & VISUAL RENDER ENGINE                            |
|  - L-Shaped Bounding Box Corner Rendering & Custom Typography                     |
|  - Real-Time Telemetry HUD Overlay (FPS, VRAM, CPU, RAM, Brightness, Governor)    |
+-----------------------------------------------------------------------------------+
```

---

## Detailed System Component Breakdown

### 1. Bootstrap & Auto-Provisioning Engine
When `EyeOfAI_System.py` or `EyeOfAI_System.ipynb` is executed, the bootstrapper initializes the execution environment automatically:
* **PyTorch Serialization Shim Creation**: Constructs a virtual runtime shim inside `torch.utils.serialization` if missing, preventing import crashes across varied PyTorch distribution releases.
* **NumPy 1.24+ Type Alias Patching**: Restores legacy type aliases (`np.bool`, `np.float`, `np.int`, `np.object`) dynamically at runtime to prevent downstream deprecation errors in legacy sub-libraries (such as ByteTrack).
* **Automated Dependency Injection & Build System**: Checks for and auto-installs system C compilers (`build-essential`), `detectron2`, `Detic`, `CenterNet2`, `ByteTrack`, `pytorchvideo`, `deepface`, `pycocotools`, and `ninja`.
* **Dynamic Classifier Metadata Provisioning**: Downloads pre-computed CLIP text embedding feature classifiers (`.npy`) for LVIS, Objects365, OpenImages, and COCO vocabularies into `/content/Detic/datasets/metadata`. Automatically generates `lvis_v1_categories.py` if missing.
* **Automatic Weight Fetching**: Validates and downloads large neural network weight checkpoints:
  - Detic Swin-B LCOCOI21k Checkpoint (~2.6 GB)
  - PyTorchVideo MViT-B 16x4 Kinetics-700 Checkpoint (~140 MB)

#### Architectural Rationale: Why Weights & Models are Downloaded Dynamically at Runtime
A common question from engineers is why model weights are downloaded at runtime rather than being committed to the GitHub repository or saved offline locally:
1. **Git Repository Optimization**: Storing multi-gigabyte binary weights in Git bloats repository clone times, violates standard GitHub file size limits (100 MB per file), and consumes Git LFS bandwidth quotas rapidly.
2. **Environment Agnosticism**: Dynamic fetching enables seamless single-click deployment on cloud platforms (Google Colab, AWS EC2, Kaggle Notebooks, Docker containers) without requiring users to manually manage multi-gigabyte download paths.
3. **Weight Hash & Integrity Verification**: Downloading from official release channels ensures that model weights remain uncorrupted, exact, and up-to-date with upstream checkpoints.

---

### 2. Configuration Management Core (`EyeConfig`)
All operational hyperparameters are centralized inside a dataclass (`EyeConfig`), facilitating runtime adjustments without modifying system code:

```python
@dataclass
class EyeConfig:
    # --- Object Detection (Detic Swin-B) ---
    detic_threshold: float = 0.0          # Set to 0.0 to pass all detections to the Scene Calibrator offset
    detic_model_path: str  = "/content/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    detic_config: str      = "/content/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    vocabulary: str        = "lvis"       # Options: "lvis" (1,200+ classes), "objects365", "openimages", "coco"

    # --- Multi-Object Tracking (ByteTrack) ---
    track_thresh: float = 0.5             # High confidence tracking threshold
    track_buffer: int   = 30              # Frames to retain lost track identity before purging
    match_thresh: float = 0.8             # IoU threshold for matching detections to existing tracks
    min_box_area: float = 100.0           # Minimum bounding box area (in pixels) to track

    # --- Facial Telemetry (DeepFace) ---
    deepface_backend: str          = "retinaface" # Face detection backend (retinaface chosen for accuracy)
    emotion_interval_sec: float    = 2.0          # Re-evaluation interval for facial emotion
    age_gender_interval_sec: float = 4.0          # Re-evaluation interval for age and gender
    deepface_frame_skip: int       = 10           # Frame skip interval for face checks
    deepface_emotion_smooth_window: int = 5       # Rolling window size for majority-vote emotion smoothing
    deepface_emotion_confidence_threshold: float = 0.6 # Confidence cut-off for accepting emotion labels

    # --- Visual Overlay & UI ---
    box_color: Tuple[int, int, int] = (0, 0, 220) # BGR accent color for bounding box corners
    box_thickness: int              = 1
    font_scale: float               = 0.42
    l_size: int                     = 14           # Pixel length of L-shaped corner markers
    show_hud: bool                  = False        # Toggle for real-time telemetry HUD overlay

    # --- Video Ingestion & Resolution ---
    input_width: int      = 960
    input_height: int     = 540
    max_fps: int          = 30
    frame_queue_size: int = 8

    # --- Performance Governor ---
    gov_target_fps: float   = 25.0        # Target FPS threshold
    gov_window: int         = 30          # Rolling sample window size
    gov_check_interval: int = 10          # Frames between governor level re-evaluations
    gov_levels: int         = 3           # Quality levels: 0=FULL, 1=REDUCED_FACE, 2=LITE

    # --- Adaptive Scene Calibrator (CLAHE) ---
    clahe_clip_init: float      = 2.0
    clahe_tile: int             = 8
    brightness_ema_alpha: float = 0.1     # Exponential Moving Average smoothing factor
    bright_lo: float            = 60.0    # Dark threshold triggering CLAHE boost
    bright_hi: float            = 190.0   # Over-exposure threshold triggering CLAHE dampening

    # --- Async Concurrency & Action Recognition ---
    async_workers: int          = 2       # Worker threads for background analytics
    action_clip_len: int        = 16      # Temporal frames fed to MViT (must be 16)
    action_frame_stride: int    = 4       # Temporal frame stride (must be 4)
    action_interval_frames: int = 16      # New frames required before re-running action inference
    action_top_k: int           = 1
    mvit_k700_path: str         = "/content/mvit_k700/MVIT_B_16x4_K700.pyth"

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str  = "/content/eyeofai_debug.log"
```

---

### 3. System Telemetry & Self-Healing Core (`SystemMonitor` & `self_heal`)
Real-time monitoring and fault tolerance are handled by `SystemMonitor` and the `self_heal` supervisor:

* **System Telemetry**:
  - Tracks rolling average FPS, per-frame detection counts, total system uptime, CPU usage percentage, virtual memory (RAM) usage, and dedicated GPU VRAM allocations.
  - Maintains a circular thread-safe log buffer (`events_log`) holding the 200 most recent operational events with millisecond-precision timestamps (`HH:MM:SS.mmm`).
* **Self-Healing Supervisor (`self_heal`)**:
  - Intercepts uncaught exceptions in critical execution blocks.
  - Automatically attempts exponential backoff retries (up to 3 attempts).
  - Flushes CUDA caches (`torch.cuda.empty_cache()`) and forces Python garbage collection (`gc.collect()`) upon fault detection to recover from transient OOM conditions.

---

### 4. Dynamic Performance Governor (`PerformanceGovernor`)
To prevent frame drops during computational spikes, the `PerformanceGovernor` dynamically adjusts workload complexity based on real-time performance metrics:

#### Governor Operational Levels
* **Level 0 (FULL)**:
  - Resolution: Native (960x540)
  - Face Analytics Interval: Normal (1x base interval)
  - Detection Confidence Cutoff: Base setting (+0.00 offset)
* **Level 1 (REDUCED_FACE)**:
  - Resolution: 75% Scale (720x405)
  - Face Analytics Interval: Half-rate (2x base interval)
  - Detection Confidence Cutoff: Slight elevation (+0.05 offset)
* **Level 2 (LITE)**:
  - Resolution: 50% Scale (480x270)
  - Face Analytics Interval: Quarter-rate (4x base interval)
  - Detection Confidence Cutoff: Aggressive suppression (+0.10 offset)

#### Load Factor Evaluation & Recovery Hysteresis
The governor monitors a rolling average of frame rates. When the load factor (`target_fps / actual_fps`) exceeds `1.35` (indicating a >35% drop below target FPS), the governor immediately steps down one level to restore throughput. 

To avoid rapid level thrashing, recovery to higher quality levels requires sustained performance: the load factor must remain below `0.80` for 20 consecutive evaluation checks (`_RECOVER_NEEDED`) before stepping back up.

---

### 5. Adaptive Scene Calibrator (`SceneCalibrator`)
Video feeds captured under varying outdoor or low-light conditions often degrade standard neural network features. The `SceneCalibrator` mitigates this through dynamic enhancement:

```
Raw Frame -> Downscale (160x90) -> HSV V-Channel Extraction -> Compute Mean Brightness (b)
                                                                             |
                                                                             v
CLAHE Enhancement <- Recombine LAB Channels <- Apply CLAHE to L-Channel <- Compute EMA Brightness & Clip Limit
                                                                             |
                                                                             v
                                                                Shift Confidence Offset
```

#### Architectural Rationale: Why Applying CLAHE in LAB Color Space is Essential
Applying standard histogram equalization or CLAHE directly to RGB/BGR channels causes significant color distortion, unnatural saturation shifts, and chrominance artifacts that confuse deep object detectors. 

EyeOfAI converts images to the **LAB Color Space**, where channel **L** represents Pure Luminance (lightness), while channels **A** and **B** represent color chrominance. Applying CLAHE strictly to the **L-channel** enhances local contrast and uncovers hidden detail in dark shadows without altering the underlying color spectrum.

Additionally, the calibrator adjusts object detection confidence cut-offs in real time:
* **Dark Scenes (Brightness < 60)**: Increases CLAHE clip limit to 3.0 and sets confidence offset to `-0.07`, boosting detector sensitivity to capture low-contrast objects.
* **Over-Exposed Scenes (Brightness > 190)**: Dampens CLAHE clip limit to 1.2 and sets confidence offset to `+0.06`, reducing false positives caused by specular reflections and bright glare.

---

### 6. Open-Vocabulary Object Detection Engine (Detic Swin-B)
Object detection is powered by **Detic (Detecting Twenty-Thousand Classes)** using a **Swin-B (Swin Transformer Base)** backbone paired with a CenterNet2 region proposal network:
* **21,000+ Class Vocabulary Support**: Leverages joint training on LVIS, COCO, Objects365, and ImageNet-21k datasets.
* **CLIP Embedding Integration**: Maps region visual proposals directly into CLIP's shared text-image embedding space, enabling open-set detection of arbitrary objects based on natural language text queries without model re-training.
* **High-Throughput VRAM Management**: Utilizes `detectron2`'s `DefaultPredictor` running on CUDA GPU, benefiting from cuDNN auto-tuning and TF32 matrix math operations.

---

### 7. Multi-Object Tracking Pipeline (ByteTrack)
Bounding box predictions are passed to **ByteTrack** for temporal association across video frames:
* **Dual Association Strategy**: Association is performed using high-confidence and low-confidence detection boxes separately via Kalman filtering and the Hungarian algorithm. Low-score detections (often caused by occlusion or motion blur) are retained for track continuity rather than discarded.
* **Identity Retention**: Maintains track state for up to 30 frames (`track_buffer`) during total occlusions before deleting the track ID.
* **Zero GPU Overhead**: ByteTrack relies on motion modeling and bounding-box geometry (IoU), running entirely on CPU and leaving GPU memory free for Detic.

---

### 8. Async Facial Telemetry Engine (DeepFace & RetinaFace)
When tracked object bounding boxes are classified as a `person`, crop regions are extracted and routed to the facial analytics system:
* **State-of-the-Art Face Detection**: Employs `RetinaFace` as the primary backend detector to localize facial landmarks even under partial face tilt or rotation.
* **Multi-Attribute Estimation**: Concurrently predicts facial emotion, age group, and gender.
* **Temporal Emotion Smoothing**: Raw emotion predictions fluctuate across consecutive frames due to lighting or expression shifts. EyeOfAI buffers predictions per track ID over a rolling window (`deepface_emotion_smooth_window = 5`) and applies majority-voting, suppressing transient noise.
* **In-Flight Deduplication**: Tracks active analysis requests using thread-safe sets (`_analyzing_emotion`, `_analyzing_age`, `_analyzing_gender`). If an analysis task is already running for a track ID, redundant submissions are skipped.

---

### 9. Spatio-Temporal Action Recognition Core (`ActionRecognizer`)
Action recognition is executed using **PyTorchVideo's MViT-B 16x4 (Multiscale Vision Transformer)** pre-trained on the **Kinetics-700 dataset**:

#### Input Tensor Pipeline & Spatial Preprocessing
MViT requires a 5D input tensor formatted as `(Batch, Channels, Time, Height, Width)` or `(1, 3, 16, 224, 224)`:
1. **Temporal Sampling**: A rolling frame buffer retains raw person crop images per track ID. Once the buffer reaches 64 frames (`16 clip_len * 4 stride`), 16 frames are sampled at regular 4-frame intervals.
2. **ShortSideScale**: Person crops are scaled so their shortest dimension equals `256` pixels while preserving aspect ratio.
3. **CenterCrop**: A square `224x224` region is extracted from the center of the frame.
4. **Kinetics Normalization**: Pixel values are normalized using Kinetics-700 dataset statistics: `Mean = [0.45, 0.45, 0.45]` and `Std = [0.225, 0.225, 0.225]`.

#### Architectural Rationale: Why Action Recognition Runs on CPU
MViT-B 16x4 performs 3D spatio-temporal convolutions across temporal frame stacks. Executing MViT on GPU alongside Detic's Swin-B transformer (~2.6 GB VRAM) caused periodic VRAM spikes that led to CUDA Out-Of-Memory (OOM) faults on 16 GB GPUs (e.g., NVIDIA T4). 

By hosting MViT-B on CPU and isolating inference inside a dedicated background worker (`ThreadPoolExecutor`), GPU VRAM usage remains constant, while action predictions update seamlessly without dropping video frames.

---

### 10. Non-Blocking Person Attribute Store (`PersonAttributeStore`)
The `PersonAttributeStore` acts as a thread-safe, lock-protected cache bridging background CPU workers and the main visualization thread:
* **Non-Blocking Reads**: The main visualization loop queries `store.get(track_id)` on every frame. If an attribute (e.g., action or emotion) is still being processed, the store returns the last known value immediately without blocking the rendering thread.
* **Independent Refresh Timers**: Each attribute maintains its own timestamp counter. Emotion is updated every 2.0 seconds, age/gender every 4.0 seconds, and action every 16 frames.
* **Stale Track Garbage Collection (`purge_stale`)**: When ByteTrack terminates a track ID, `purge_stale` purges all associated historical buffers from memory, preventing memory leaks during extended operation.

---

## Technical Justifications: Key Engineering Decisions

| Architectural Choice | Decision | Engineering Justification |
| :--- | :--- | :--- |
| **Compute Split** | **GPU**: Detic Swin-B<br>**CPU**: DeepFace + MViT | Preserves GPU VRAM for the 2.6 GB Detic object detector, avoiding CUDA memory spikes and preventing GPU context-switching overhead. |
| **Model Weight Loading** | **Dynamic Runtime Fetching** | Keeps git repository lightweight, avoids GitHub's 100 MB file limit and Git LFS bandwidth quotas, and enables automated single-click deployment in cloud environments. |
| **Enhancement Space** | **LAB Color Space CLAHE** | Applies contrast enhancement strictly to the L-channel (Luminance), preserving original color fidelity and preventing chrominance distortion. |
| **Multi-Object Tracking**| **ByteTrack over DeepSORT** | Uses bounding-box geometry and motion estimation, avoiding the need for a secondary GPU-bound feature extraction network. |
| **Emotion Inference** | **Temporal Majority Voting** | Buffers predictions over a rolling window of 5 frames, filtering out transient single-frame facial expression anomalies. |
| **System Exception Management** | **Supervisor with `self_heal`** | Catches non-fatal exceptions, empties CUDA memory caches, triggers garbage collection, and retries operations without crashing the pipeline. |

---

## Directory Structure & Project Layout

Below is the verified workspace file structure for the EyeOfAI repository:

```
EyeOfAI-System/
├── EyeOfAI_System.py         # Main standalone Python system implementation
├── EyeOfAI_System.ipynb      # Google Colab / Jupyter Notebook implementation
├── requirements.txt          # Complete Python package dependency manifest
├── README.md                 # System documentation & technical reference
├── Videos/                   # Directory containing demo test videos
│   ├── demo_sample_1.mp4     # Pre-uploaded test clip for open-vocabulary evaluation
│   ├── demo_sample_2.mp4     # Pre-uploaded test clip for action & face analytics
│   └── README.md             # Guide on test video formats & usage
└── eyeofai_debug.log         # Auto-generated runtime debug log file
```

---

## Installation & Setup Guide

### Environment Prerequisites
* Linux (Ubuntu 20.04 / 22.04 recommended) or Google Colab Cloud Instance
* Python >= 3.10
* NVIDIA GPU with CUDA 12.1 support (for GPU acceleration)

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Don-Youssef/EyeOfAI-System.git
   cd EyeOfAI-System
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Launch via Python Script**:
   ```bash
   python EyeOfAI_System.py
   ```

5. **Launch via Google Colab**:
   Open `EyeOfAI_System.ipynb` in Google Colab, select a **T4 GPU** runtime environment, and execute all notebook cells sequentially.

---

## Sample Demonstration Videos

Pre-uploaded demonstration videos are included in the `Videos/` directory:
* Users can evaluate system performance, open-vocabulary detection accuracy, and attribute analytics on these sample clips before connecting custom camera feeds or external video files.
* To process a demonstration video, set the video input path in `EyeOfAI_System.py` or the notebook cell to:
  ```python
  video_input_path = "Videos/demo_sample_1.mp4"
  ```

---

## Complete Dependency Specifications (`requirements.txt`)

Below are the package requirements used by the system:

```txt
torch==2.3.0
torchvision==0.18.0
opencv-python-headless>=4.8.0
Pillow>=9.5.0
requests>=2.31.0
pyyaml>=6.0
cython>=3.0.0
matplotlib>=3.7.0
pycocotools>=2.0.6
ninja>=1.11.0
rich>=13.0.0
psutil>=5.9.0
loguru>=0.7.0
lap>=0.4.0
cython_bbox>=0.1.5
thop>=0.1.1
deepface>=0.0.79
tf-keras>=2.15.0
python-jose>=3.3.0
pytorchvideo>=0.1.5
av>=9.0.0
lvis>=0.5.3
```

---

## Model Author Mentions, Credits & Acknowledgments

The EyeOfAI System builds upon open-source research and model architectures. We explicitly acknowledge and thank the following research teams and open-source projects:

* **Facebook AI Research (FAIR) - Detectron2 & Detic**:
  * *Detic: Detecting Twenty-Thousand Classes Teams* (Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, Ishan Misra).
  * [Detic Repository](https://github.com/facebookresearch/Detic) | [Detectron2 Repository](https://github.com/facebookresearch/detectron2)
* **ByteDance / IFZhang - ByteTrack**:
  * *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* (Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang).
  * [ByteTrack Repository](https://github.com/ifzhang/ByteTrack)
* **Facebook AI Research (FAIR) - PyTorchVideo & MViT**:
  * *Multiscale Vision Transformers (MViT)* (Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Christoph Feichtenhofer, Jitendra Malik).
  * [PyTorchVideo Repository](https://github.com/facebookresearch/pytorchvideo)
* **Sefik Ilkin Serengil - DeepFace Framework**:
  * *DeepFace: A Lightweight Facial Analysis Framework for Python*.
  * [DeepFace Repository](https://github.com/serengil/deepface)
* **OpenAI - CLIP**:
  * *Learning Transferable Visual Models From Natural Language Supervision* (Alec Radford, Jong Wook Kim, et al.).

---

## License & Attribution

This project is released under the **MIT License**. You are free to use, modify, distribute, and integrate this software into academic research, personal projects, or commercial systems, provided proper credit and copyright notices are retained.

---
*Maintained with excellence by Youssef Muhammad (Don-Youssef)*
