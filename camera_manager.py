import asyncio
from imutils.video import VideoStream
import cv2
import imutils
from typing import Dict, Optional, Tuple, List
import logging
from schemas.schema import CameraConfig
from datetime import datetime

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, config: CameraConfig, gate_id: str, direction: str):
        self.config = config
        self.gate_id = gate_id
        self.direction = direction
        self.stream: Optional[VideoStream] = None
        self.is_initializing = False
        self.max_retries = 3
        self.retry_interval = 5.0
        
    async def initialize(self):
        """Initialize the camera connection"""
        if self.stream is not None:
            return

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                self.is_initializing = True
                self.stream = VideoStream(src=self.config.url).start()
                await asyncio.sleep(2.0)  # Wait for camera to warm up
                
                # Verify stream is working
                test_frame = self.stream.read()
                if test_frame is None:
                    raise Exception("Failed to read test frame")
                    
                logger.info(f"Initialized camera for gate {self.gate_id} {self.direction}")
                return
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Failed to initialize camera: {e}. Retry {retry_count}/{self.max_retries}")
                if self.stream:
                    self.stream.stop()
                    self.stream = None
                await asyncio.sleep(self.retry_interval)
            finally:
                self.is_initializing = False
                
        raise Exception(f"Failed to initialize camera after {self.max_retries} attempts")
            
    async def get_frame(self, resize_width: int = None) -> Optional[cv2.Mat]:
        """Capture a single frame on demand"""
        if not self.stream and not self.is_initializing:
            await self.initialize()
            
        if not self.stream:
            return None
            
        frame = self.stream.read()
        if frame is not None and resize_width:
            frame = imutils.resize(frame, width=resize_width)
        return frame
        
    def release(self):
        """Release camera resources"""
        if self.stream:
            self.stream.stop()
            self.stream = None

class CameraManager:
    def __init__(self):
        self.cameras: Dict[Tuple[str, str], Camera] = {}
        self.camera_configs: Dict[str, CameraConfig] = {}
        self.camera_mappings: Dict[str, Tuple[str, str]] = {}
        self.initialization_tasks: List[asyncio.Task] = []
        
    async def configure(self, camera_configs: Dict[str, CameraConfig], mappings: Dict[str, Tuple[str, str]]):
        """Configure and initialize all cameras in parallel"""
        self.camera_configs = camera_configs
        self.camera_mappings = mappings
        
        # Create and initialize all cameras in parallel
        init_tasks = []
        for camera_id, (gate_id, direction) in mappings.items():
            if camera_id in camera_configs:
                config = camera_configs[camera_id]
                camera = Camera(config, gate_id, direction)
                self.cameras[(gate_id, direction)] = camera
                
                try:
                    task = asyncio.create_task(camera.initialize())
                    init_tasks.append(task)
                except Exception as e:
                    logger.error(f"Error scheduling camera initialization: {camera_id}, {e}")

                
        # Wait for all cameras to initialize
        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)
            
        logger.info(f"Initialized {len(self.cameras)} cameras")
        
    def get_camera(self, gate_id: str, direction: str) -> Optional[Camera]:
        """Get camera instance for given gate and direction"""
        return self.cameras.get((gate_id, direction))
        
    async def cleanup(self):
        """Cleanup all cameras"""
        for camera in self.cameras.values():
            camera.release()
        self.cameras.clear()