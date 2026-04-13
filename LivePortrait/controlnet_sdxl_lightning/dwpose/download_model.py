import os
from pathlib import Path


def get_open_pose_model():
    """Resolve paths under $DW_POSE/dw_pose_weights/ (default: ~/dw_pose_weights). Does not download."""
    current_dir = os.getenv('DW_POSE', default=str(Path.home()))
    model_dir = os.path.join(current_dir, 'dw_pose_weights')
    os.makedirs(model_dir, exist_ok=True)
    filenames = {
        'yolox_l_pose': 'yolox_l.onnx',
        'dw_pose': 'dw-ll_ucoco_384.onnx',
    }
    model_paths = {name: os.path.join(model_dir, fn) for name, fn in filenames.items()}
    return model_paths, model_dir
