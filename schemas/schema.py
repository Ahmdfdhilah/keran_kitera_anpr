import os
from dotenv import load_dotenv
from typing import Dict, Union
from pydantic import BaseModel

# Load variabel dari .env
load_dotenv()

class CameraConfig(BaseModel):
    name: str
    url: Union[str, int]
    gate_id: str
    direction: str
    username: str | None = None
    password: str | None = None
    resize_width: int = 640
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
    mqtt_broker: str = os.getenv("MQTT_BROKER", "localhost")
    mqtt_port: int = int(os.getenv("MQTT_PORT", 1883))
    google_vision_token: str = os.getenv("GOOGLE_VISION_TOKEN", "token")
    model_config = {"protected_namespaces": ()}