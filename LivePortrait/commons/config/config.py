# coding: utf-8
# Author: Vo Nguyen An Tin
# Email: tinprocoder0908@gmail.com

import os
from dataclasses import dataclass, asdict
from typing import Literal, Tuple
import torch.cuda
import yaml

# Expected filenames under live_portrait_weights/{live_portrait,insightface}/ (no auto-download).
LOCAL_MODEL_FILENAMES = {
    'live_portrait': {
        'grid_sample_3d': 'libgrid_sample_3d_plugin.so',
        'F_onnx': 'appearance_feature_extractor.onnx',
        'M_onnx': 'motion_extractor.onnx',
        'GW_onnx': 'generator_fix_grid.onnx',
        'S_onnx': 'stitching.onnx',
        'SE_onnx': 'stitching_eye.onnx',
        'SL_onnx': 'stitching_lip.onnx',
        'F_rt': 'appearance_feature_extractor_fp32.engine',
        'M_rt': 'motion_extractor_fp32.engine',
        'GW_rt': 'generator_fp32.engine',
        'S_rt': 'stitching_fp32.engine',
        'SE_rt': 'stitching_eye_fp32.engine',
        'SL_rt': 'stitching_lip_fp32.engine',
        'F_rt_half': 'appearance_feature_extractor_fp16.engine',
        'M_rt_half': 'motion_extractor_fp16.engine',
        'GW_rt_half': 'generator_fp16.engine',
        'S_rt_half': 'stitching_fp16.engine',
        'SE_rt_half': 'stitching_eye_fp16.engine',
        'SL_rt_half': 'stitching_lip_fp16.engine',
    },
    'insightface': {
        'arc_face': 'w600k_r50.onnx',
        '2d106det': '2d106det.onnx',
        'det_10g': 'det_10g.onnx',
        'landmark': 'landmark.onnx',
    },
}


def get_local_model_paths():
    """Resolve paths under ./live_portrait_weights/ from cwd. Does not download."""
    current_dir = os.getcwd()
    face_dir = os.path.join(current_dir, 'live_portrait_weights')
    model_paths = {}
    for main_key, sub_dict in LOCAL_MODEL_FILENAMES.items():
        dir_path = os.path.join(face_dir, main_key)
        model_paths[main_key] = {
            sub_key: os.path.join(dir_path, filename)
            for sub_key, filename in sub_dict.items()
        }
    return model_paths, face_dir


@dataclass(repr=False)  # use repr from PrintableConfig
class Config:
    model_paths, face_dir = get_local_model_paths()
    grid_sample_3d: str = os.getenv('GRID_SAMPLE_3D_PLUGIN', model_paths['live_portrait']['grid_sample_3d'])
    # ONNX
    checkpoint_F: str = model_paths['live_portrait']['F_onnx']  # path to checkpoint
    checkpoint_M: str = model_paths['live_portrait']['M_onnx']  # path to checkpoint
    checkpoint_GW: str = model_paths['live_portrait']['GW_onnx']
    checkpoint_S: str = model_paths['live_portrait']['S_onnx']  # path to checkpoint
    checkpoint_SE: str = model_paths['live_portrait']['SE_onnx']
    checkpoint_SL: str = model_paths['live_portrait']['SL_onnx']

    # TensorRT FP32
    F_rt: str = model_paths['live_portrait']['F_rt']  # path to checkpoint
    M_rt: str = model_paths['live_portrait']['M_rt']  # path to checkpoint
    GW_rt: str = model_paths['live_portrait']['GW_rt']  # path to checkpoint
    S_rt: str = model_paths['live_portrait']['S_rt']  # path to checkpoint
    SE_rt: str = model_paths['live_portrait']['SE_rt']
    SL_rt: str = model_paths['live_portrait']['SL_rt']

    # TensorRT FP16
    F_rt_half: str = model_paths['live_portrait']['F_rt_half']  # path to checkpoint
    M_rt_half: str = model_paths['live_portrait']['M_rt_half']  # path to checkpoint
    GW_rt_half: str = model_paths['live_portrait']['GW_rt_half']  # path to checkpoint
    S_rt_half: str = model_paths['live_portrait']['S_rt_half']  # path to checkpoint
    SE_rt_half: str = model_paths['live_portrait']['SE_rt_half']
    SL_rt_half: str = model_paths['live_portrait']['SL_rt_half']

    flag_use_half_precision: bool = True  # whether to use half precision
    flag_lip_zero: bool = True  # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    lip_zero_threshold: float = 0.03
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True  # we recommend setting it to True!
    flag_relative: bool = True  # whether to use relative motion
    flag_pasteback: bool = True  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    flag_do_crop: bool = True  # whether to crop the source portrait to the face-cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True
    flag_write_result: bool = True  # whether to write output video
    flag_write_gif: bool = False

    anchor_frame: int = 0  # set this value if find_best_frame is True

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    output_fps: int = 30  # fps for output video
    crf: int = 15  # crf for output video
    mask_crop: str = 'None'
    size_gif: int = 256
    ref_max_shape: int = 1280
    ref_shape_n: int = 2

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # crop config
    ckpt_landmark: str = model_paths['insightface']['landmark']
    ckpt_arc_face: str = model_paths['insightface']['arc_face']
    ckpt_landmark_106: str = model_paths['insightface']['2d106det']
    ckpt_det: str = model_paths['insightface']['det_10g']
    ckpt_face: str = face_dir
    dsize: int = 512  # crop size
    scale: float = 2.3  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down


# Function to save the configuration to a YAML file
def save_config_to_yaml(filename="efficient-live-portrait.yaml"):
    # Define the path where the YAML file will be saved
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        # Save the configuration to the YAML file
        with open(file_path, 'w') as file:
            yaml.safe_dump(asdict(Config()), file)
    return file_path
