from typing import Dict, Union
from pydantic import BaseModel


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
    config_path: str = "models/yolov4-ANPR.cfg"
    weights_path: str = "models/yolov4-ANPR.weights"
    names_path: str = "models/yolov4-ANPR.names"
    conf_threshold: float = 0.9
    nms_threshold: float = 0.3
    input_width: int = 416
    input_height: int = 416

    model_config = {"protected_namespaces": ()}


class Settings(BaseModel):
    cameras: Dict[str, CameraConfig]
    anpr: ANPRConfig
    result_path: str = "result"
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883

    model_config = {"protected_namespaces": ()}