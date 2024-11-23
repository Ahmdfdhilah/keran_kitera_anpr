# config/camera_config.py
from .settings import CameraConfig
from typing import Dict


def get_camera_configs() -> Dict[str, dict]:
    return {
        "camera1": {
            "name": "Gerbang Utama Motor In",
            "url": 0,
            "username": "",
            "password": "",
            "gate_id": "1",
            "direction": "in",
        },
    #     "camera2": {
    #         "name": "Gerbang Utama Motor Out",
    #         "url": 0,
    #         "username": "",
    #         "password": "",
    #         "gate_id": "1",
    #         "direction": "out",
    # }
    }
