# Efficient-Live-Portrait

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
