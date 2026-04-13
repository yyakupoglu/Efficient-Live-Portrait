# Real-time Live Portrait with web browser streaming (no X display needed)
# Usage: python3 run_webcam_stream.py  -->  open http://localhost:8890 in browser

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
import time
import threading
import warnings
from flask import Flask, Response
from omegaconf import OmegaConf
from LivePortrait import EfficientLivePortrait
from LivePortrait.commons import save_config_to_yaml
import argparse

warnings.filterwarnings("ignore")

# --- Config ---
SOURCE_IMAGE = 'experiment_examples/examples/source/s0.jpg'
WEBCAM_ID = 0
STREAM_PORT = 8890
JPEG_QUALITY = 80

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
fps_display = 0
stop_event = threading.Event()



def run_portrait(args):
    global output_frame, fps_display

    print("Loading models...")
    cfg_yaml = save_config_to_yaml()
    kwargs = OmegaConf.load(cfg_yaml)

    # Point to the newly built 10.14 TensorRT engines
    engine_dir = "live_portrait_weights/live_portrait/10.14"
    if args.fp32:
        kwargs.F_rt  = f"{engine_dir}/appearance_feature_extractor_fp32.engine"
        kwargs.M_rt  = f"{engine_dir}/motion_extractor_fp32.engine"
        kwargs.GW_rt = f"{engine_dir}/generator_fix_grid_fp32.engine"
        kwargs.S_rt  = f"{engine_dir}/stitching_fp32.engine"
        kwargs.SE_rt = f"{engine_dir}/stitching_eye_fp32.engine"
        kwargs.SL_rt = f"{engine_dir}/stitching_lip_fp32.engine"
    else:
        kwargs.F_rt_half  = f"{engine_dir}/appearance_feature_extractor_fp16.engine"
        kwargs.M_rt_half  = f"{engine_dir}/motion_extractor_fp16.engine"
        kwargs.GW_rt_half = f"{engine_dir}/generator_fix_grid_fp16.engine"
        kwargs.S_rt_half  = f"{engine_dir}/stitching_fp16.engine"
        kwargs.SE_rt_half = f"{engine_dir}/stitching_eye_fp16.engine"
        kwargs.SL_rt_half = f"{engine_dir}/stitching_lip_fp16.engine"
        
        # If mixed mode, override ONLY the motion_extractor with its FP32 engine
        if args.mixed:
            kwargs.M_rt_half = f"{engine_dir}/motion_extractor_fp32.engine"

    # TensorRT GridSample plugin built by Dockerfile 
    kwargs.grid_sample_3d = os.getenv('GRID_SAMPLE_3D_PLUGIN', "grid/build/libgrid_sample_3d_plugin.so")

    live_portrait = EfficientLivePortrait(
        use_tensorrt=True, half=not args.fp32, cropping_video=False, **kwargs
    )

    print(f"Preparing source portrait from: {SOURCE_IMAGE}")
    _, _, x_s, f_s, r_s, x_s_info, lip_delta_before_animation, \
        crop_info, img_rgb, _ = live_portrait.prepare_webcam_portrait(
            source_image_path=SOURCE_IMAGE
        )

    # Pre-compute mask once
    mask_ori = live_portrait.prepare_paste_back(
        crop_info['M_c2o'],
        dsize=(img_rgb.shape[1], img_rgb.shape[0])
    )

    cap = cv2.VideoCapture(WEBCAM_ID)
    # C920 performs best using MJPG. We request 60 FPS and 256x256. 
    # (cv2 will auto-fallback to closest supported, e.g., 320x240@30fps if 60fps/256x256 is unsupported by the driver)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
    cap.set(cv2.CAP_PROP_FPS, 60)

    fps_counter = 0
    fps_time = time.time()
    frame_count = 0

    print(f"\n>>> Stream ready at http://localhost:{STREAM_PORT}")
    print(">>> Open that URL in your browser to see the output")
    print(">>> Press Ctrl+C to stop\n")
    
    try:
        while not stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            x_s, x_d_i_new = live_portrait.get_kp_info(
                frame, x_s, r_s, x_s_info, lip_delta_before_animation
            )
            i_p_i = live_portrait.warp_decode(f_s, np.array(x_s), np.array(x_d_i_new))

            result = live_portrait.paste_back(i_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
            result_bgr = np.ascontiguousarray(result[:, :, ::-1])

            # FPS tracking
            fps_counter += 1
            frame_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = now
                print(f'\rFPS: {fps_display} | Frames: {frame_count}', end='', flush=True)

            # Stamp FPS on frame
            cv2.putText(result_bgr, f'FPS: {fps_display}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            with lock:
                output_frame = result_bgr.copy()


    finally:
        print("\nCleaning up portrait engine...")
        cap.release()
        del live_portrait



def generate_mjpeg():
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            flag, encoded = cv2.imencode(
                '.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
        if not flag:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')


@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html><head><title>Live Portrait</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; display: flex; justify-content: center;
         align-items: center; height: 100vh; font-family: sans-serif; }
  img { max-width: 100%; max-height: 100vh; border: 2px solid #333;
        border-radius: 8px; }
  .label { position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
           color: #0f0; font-size: 14px; opacity: 0.7; }
</style></head>
<body>
  <div class="label">Live Portrait — Real-Time Stream</div>
  <img src="/video_feed">
</body></html>'''


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", action="store_true", help="Run with FP32 TensorRT engines instead of FP16")
    parser.add_argument("--mixed", action="store_true", help="Run motion_extractor in FP32, rest in FP16 (fixes FP16 quality)")
    args = parser.parse_args()

    t = threading.Thread(target=run_portrait, args=(args,), daemon=False)
    t.start()
    print(f"Starting stream server on port {STREAM_PORT}...")
    try:
        app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        print("Waiting for thread to finish...")
        t.join(timeout=5)
        print("Done.")

