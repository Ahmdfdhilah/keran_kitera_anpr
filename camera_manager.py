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
        self.reconnect_interval = 10.0
        self.is_running = True
        self.reconnect_task = None
        
    async def initialize(self):
        """Initialize the camera connection"""
        if self.stream is not None:
            return

        retry_count = 0
        while retry_count < self.max_retries and self.is_running:
            try:
                self.is_initializing = True
                self.stream = VideoStream(src=self.config.url).start()
                await asyncio.sleep(2.0)  # Wait for camera to warm up
                
                # Verify stream is working
                test_frame = self.stream.read()
                if test_frame is None:
                    raise Exception("Failed to read test frame")
                    
                logger.info(f"Initialized camera for gate {self.gate_id} {self.direction}")
                # Start background reconnection monitor
                self.start_reconnect_monitor()
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

    def start_reconnect_monitor(self):
        """Start background task to monitor camera connection"""
        if self.reconnect_task is None:
            self.reconnect_task = asyncio.create_task(self._monitor_connection())
            
    async def _monitor_connection(self):
        """Monitor camera connection and attempt reconnection if needed"""
        while self.is_running:
            try:
                if self.stream is None:
                    await self.initialize()
                else:
                    # Test connection by attempting to read frame
                    frame = self.stream.read()
                    if frame is None:
                        logger.warning(f"Camera {self.gate_id}-{self.direction} connection lost. Attempting reconnection...")
                        self.stream.stop()
                        self.stream = None
                        await self.initialize()
                        
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                if self.stream:
                    self.stream.stop()
                    self.stream = None
                    
            await asyncio.sleep(self.reconnect_interval)
            
    async def get_frame(self) -> Optional[cv2.Mat]:
        """Capture a single frame on demand with automatic reconnection"""
        if not self.stream and not self.is_initializing:
            try:
                await self.initialize()
            except Exception as e:
                logger.error(f"Failed to reinitialize camera during frame capture: {e}")
                return None
            
        if not self.stream:
            return None
            
        try:
            frame = self.stream.read()
            if frame is None:
                logger.warning("Received null frame, triggering reconnection...")
                self.stream.stop()
                self.stream = None
                return None
                
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            self.stream.stop()
            self.stream = None
            return None

    def resize_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Resize frame to specified width while maintaining aspect ratio"""
        try:
            if frame is not None and self.config.resize_width > 0:
                return imutils.resize(frame, width=self.config.resize_width)
            return frame
        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame
        
    async def release(self):
        """Release camera resources and stop reconnection monitoring"""
        self.is_running = False
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass
            self.reconnect_task = None
            
        if self.stream:
            self.stream.stop()
            self.stream = None

class CameraManager:
    def __init__(self):
        self.cameras: Dict[Tuple[str, str], Camera] = {}
        self.camera_configs: Dict[str, CameraConfig] = {}
        self.camera_mappings: Dict[str, Tuple[str, str]] = {}
        
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
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Camera initialization error: {result}")
            
        logger.info(f"Initialized {len(self.cameras)} cameras")
        
    def get_camera(self, gate_id: str, direction: str) -> Optional[Camera]:
        """Get camera instance for given gate and direction"""
        return self.cameras.get((gate_id, direction))
        
    async def cleanup(self):
        """Cleanup all cameras"""
        cleanup_tasks = []
        for camera in self.cameras.values():
            cleanup_tasks.append(camera.release())
        await asyncio.gather(*cleanup_tasks)
        self.cameras.clear()