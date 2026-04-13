# Efficient Live Portrait — lightweight build image
#
# Bakes in: Python deps + GridSample3D TRT plugin
# Everything else (weights, ONNX patching, TRT engines) is handled
# by run.sh at container start via the bind-mounted volume.
#
# Build:  docker compose build
# Run:    docker compose up

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements_25.11.txt /tmp/requirements_25.11.txt
RUN pip install --no-cache-dir -r /tmp/requirements_25.11.txt && \
    rm /tmp/requirements_25.11.txt

# ── Build GridSample3D TRT plugin (SM 75–120) ────────────────────────────────
# Covers: T4, A100, RTX 3090/A40, RTX 4090/L40, H100, B200/GB200/GB300, RTX 5090/5080/5070
RUN git clone --depth 1 https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git /opt/grid-sample3d-trt-plugin && \
    cd /opt/grid-sample3d-trt-plugin && \
    sed -i '/set_target_properties.*CUDA_ARCHITECTURES/d' CMakeLists.txt && \
    sed -i '/add_subdirectory(test)/d' CMakeLists.txt && \
    sed -i '/enable_testing/d' CMakeLists.txt && \
    mkdir -p build && cd build && \
    CUDAARCHS= cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;103;120" && \
    make -j"$(nproc)"

ENV GRID_SAMPLE_3D_PLUGIN=/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so
