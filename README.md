# Efficient-Live-Portrait
## 📹 SDXL-Lightning + Controlnet-Open-Pose + Live-Portrait


https://github.com/user-attachments/assets/5ca959f9-8fcc-4233-8a1d-9b201bd042c9



https://github.com/user-attachments/assets/24086eea-7075-45ec-8eef-2a9344185746


## 📹 Video2Video Demo


https://github.com/user-attachments/assets/de259719-d174-4c83-9287-2fa77c3b8fad


## 📹 Video Demo for normal mode
 

https://github.com/user-attachments/assets/ac0e92d7-34e1-4402-a202-d06a2e806abe

## 📹 Video Demo for Face-ID mode
+ Single Face Image
  ![368220873_826368889022136_4472311944594836999_n](https://github.com/user-attachments/assets/25851766-a454-4f16-8d44-f63923cdabf2)

+ Through Face-ID adapter
   

https://github.com/user-attachments/assets/197a8d75-3c56-43f5-ac71-e7110d9e53d1


## Introduction
This repo is the optimize task by converted to ONNX and TensorRT models for [LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control](https://github.com/KwaiVGI/LivePortrait).
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

Also we adding feature: 
+ Real-Time demo with ONNX models
+ TensorRT runtime with latest Tensorrt version. You should run on Colab, this still can't use on Window
+ Face-ID adapter for control Face animation in the Multiple Faces image you want to do
+ Coming soon for ControlNet Stable Diffusion. Stay tuned
## Features
[✅] 20/07/2024: TensorRT Engine code and Demo

[✅] 22/07/2024: Support Multiple Faces

[✅] 22/07/2024: Face-ID Adapter for Control Face Animation

[✅] 24/07/2024: Multiple Face motion in Video for animation multiples Face in image

[✅] 28/07/2024: Supported Video2Video Live Portrait (only use one Face)

[✅] 30/07/2024: Support SDXL-Lightning Controlnet-Open-Pose from 1 to 8 step for change source image to Art Image

[  ] Integrate Animate-Diff Lightning Motion module


## 🔥 Getting Started

### 1. Prerequisites
You no longer need to manage complex Conda environments or compile C++ plugins manually. This repository is now fully containerized!

Make sure you have installed:
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)

### 2. Idempotent Setup & Run
Simply clone the repository and use Docker Compose. The system will automatically:
1. Download all required ONNX weights from HuggingFace (if missing).
2. Patch the GridSample ONNX operators for TensorRT compatibility.
3. Compile all 12 TensorRT engines (FP16 and FP32) optimized specifically for your GPU.
4. Launch the web-based video stream.

```bash
git clone https://github.com/aihacker111/Efficient-Live-Portrait
cd Efficient-Live-Portrait

# Build the environment and start the setup/stream
docker compose up --build
```
> **Note:** The first run will take several minutes to download the ~2GB of weights and build the TensorRT engines. Subsequent runs will start almost instantly!

### 3. Inference and Real-time Demo 🚀

#### Web Streaming (Headless)
Once `docker compose up` finishes the setup loop, it will automatically launch the webcam stream. 

Open your browser and navigate to:
```text
http://localhost:8890
```

By default, it runs in `--mixed` mode (FP32 for motion extraction, FP16 for the rest) to ensure the highest quality without sacrificing real-time speed.

* To edit the setup, modify `run_webcam_stream.py` arguments inside `run.sh`.
* To test different input images, replace `experiment_examples/examples/source/s0.jpg`.

#### Colab Demo
Follow the instructions in the `colab` folder for cloud-based execution.

### 4. Inference speed evaluation 🚀🚀🚀
We'll release it soon.
