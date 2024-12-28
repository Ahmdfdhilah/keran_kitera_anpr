# config/settings.py
from pydantic import BaseModel, Field
from typing import List, Dict, Union
from functools import lru_cache

class CameraConfig(BaseModel):
    name: str
    url: Union[str, int] 
    gate_id: str
    direction: str
    username: str | None = None
    password: str | None = None
    resize_width: int = 1080
    enabled: bool = True

class ANPRConfig(BaseModel):
    model_path: str = "models/train_with_nms_timelimit_custom_best.pt"  # New YOLOv11 model
    conf_threshold: float = 0.25  # Adjusted for YOLOv11
    nms_threshold: float = 0.45   # Adjusted for YOLOv11
    input_width: int = 640  # YOLOv11 default size
    input_height: int = 640 # YOLOv11 default size
    
    model_config = {
        'protected_namespaces': ()
    }

class Settings(BaseModel):
    cameras: Dict[str, CameraConfig]
    anpr: ANPRConfig
    result_path: str = "result"
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    debug: bool = True
    model_config = {
        'protected_namespaces': ()
    }
