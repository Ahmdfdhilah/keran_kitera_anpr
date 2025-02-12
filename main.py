import json
import logging
import cv2
import base64
import paho.mqtt.client as mqtt
from datetime import datetime
import os
import asyncio
from typing import Dict, Optional

from camera_manager import CameraManager
from config.settings import CameraConfig, Settings, ANPRConfig
from config.camera_config import get_camera_configs
from processor import ANPRProcessor
# from processor import ANPRProcessor  # Commented out as we skip ANPR processing

from video_test import VideoTester
import argparse

logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set debug level for specific modules
logging.getLogger('processor').setLevel(logging.DEBUG)
logging.getLogger('video_test').setLevel(logging.DEBUG)

class ANPRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loop = asyncio.get_event_loop()
        
        # Ensure result directory exists
        os.makedirs(self.settings.result_path, exist_ok=True)
        
        # Skip ANPR processor initialization
        self.processor = ANPRProcessor(settings)

        self.camera_manager = CameraManager()
        
        self.mqtt_client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect
        
        self.running = True
        self.qos_level = 2
        
    async def configure_cameras(self):
        camera_mappings = {
            camera_id: (config.gate_id, config.direction)
            for camera_id, config in self.settings.cameras.items()
            if config.enabled
        }
        
        await self.camera_manager.configure(
            {k: v for k, v in self.settings.cameras.items() if v.enabled},
            camera_mappings
        )
        logger.info(f"Configured {len(camera_mappings)} enabled cameras")
        
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
            topic = "portal/anpr/+/+/request"
            client.subscribe(topic, qos=self.qos_level)
            logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT Broker. Return code: {rc}")
            
    def _on_disconnect(self, client, userdata, rc, properties=None):
        if rc != 0:
            logger.warning("Unexpected MQTT disconnection. Will auto-reconnect")
            
    def _on_message(self, client, userdata, message, properties=None):
        asyncio.run_coroutine_threadsafe(
            self._process_message(message), 
            self.loop
        )
            
    async def _process_message(self, message):
        try:
            topic_parts = message.topic.split('/')
            payload = json.loads(message.payload.decode('utf-8'))
            if len(topic_parts) != 5 or topic_parts[0] != 'portal' or topic_parts[1] != 'anpr':
                logger.error(f"Invalid topic format: {message.topic}")
                return
                
            gate_id = topic_parts[2]
            direction = topic_parts[3]
            
            identifier = payload.get("identifier", "unknown")
            error_message = payload.get("error_message", "")
                
       
            await self._process_anpr_request(gate_id, direction, identifier, error_message)
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            
    async def _process_anpr_request(self, gate_id: str, direction: str, identifier: Optional[str], error_message: Optional[str]):
        try:
            camera = self.camera_manager.get_camera(gate_id, direction)
            if not camera:
                logger.error(f"No camera found for gate {gate_id} direction {direction}")
                await self._publish_response(gate_id, direction, None, None, identifier, "Camera not found")
                return

            # Get frame from camera
            frame = await camera.get_frame()
            if frame is None:
                logger.error(f"Failed to get frame from camera {gate_id} {direction}")
                await self._publish_response(gate_id, direction, None, None, identifier, "Failed to get camera frame")
                return

            # Convert frame to base64 for documentation
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Process frame with YOLOv11
            detections = await self.processor.process_frame(frame, f"{gate_id}_{direction}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_data = {
                "gate_id": gate_id,
                "direction": direction,
                "timestamp": timestamp,
                "image": image_base64,
                "identifier": identifier
            }

            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])
                plate_text = best_detection['text']
                confidence = best_detection['confidence']
                
                result_data.update({
                    "plate_number": plate_text,
                    "confidence": confidence,
                    "status": "success"
                })

                logger.info(f"Plate detected at {gate_id}_{direction}: {plate_text} (conf: {confidence:.2f})")
            else:
                result_data.update({
                    "plate_number": None,
                    "confidence": None,
                    "status": "no_plate_detected"
                })
                logger.warning(f"No plate detected at {gate_id}_{direction}")

            # Save result and publish
            result_path = os.path.join(self.settings.result_path, f"{timestamp}_{gate_id}_{direction}.json")
            with open(result_path, 'w') as f:
                json.dump(result_data, f)

            await self._publish_response(
                gate_id, 
                direction, 
                result_data["plate_number"],
                result_data["confidence"],
                identifier,
                error_message if error_message else None,
                image_base64
            )

        except Exception as e:
            logger.error(f"Error processing ANPR request: {e}")
            await self._publish_response(gate_id, direction, None, None, identifier, str(e))
            
    async def _publish_response(
        self, 
        gate_id: str, 
        direction: str, 
        plate_text: Optional[str],
        confidence: Optional[float],
        identifier: Optional[str],
        error_message: Optional[str],
        image: Optional[str] = None
    ):
        try:
            topic = f"portal/anpr/{gate_id}/{direction}/response"
            
            payload = {
                "identifier": identifier if identifier else "",  
                "confidence": 0,  # No confidence as we skip ANPR
                "timestamp": datetime.now().isoformat(),
                "crfid": identifier,
                "error_message": error_message,
                "image": image,
            }
            
            self.mqtt_client.publish(
                topic,
                json.dumps(payload),
                qos=self.qos_level
            )
            logger.info(f"Published ANPR response to {topic}")
            
        except Exception as e:
            logger.error(f"Error publishing response: {e}")
            
    async def start(self):
        try:
            self.loop = asyncio.get_running_loop()
            await self.configure_cameras()
            
            self.mqtt_client.connect(
                self.settings.mqtt_broker,
                self.settings.mqtt_port
            )
            self.mqtt_client.loop_start()
            
            logger.info("ANPR Service started successfully")
            
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in ANPR service: {e}")
        finally:
            await self.stop()
            
    async def stop(self):
        self.running = False
        await self.camera_manager.cleanup()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logger.info("ANPR Service stopped")

if __name__ == "__main__":
    try:
        # Add argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", help="Run in test mode with video file", type=str)
        args = parser.parse_args()

        # Initialize settings
        camera_configs = get_camera_configs()
        settings = Settings(
            cameras={
                camera_id: CameraConfig(**config)
                for camera_id, config in camera_configs.items()
            },
            anpr=ANPRConfig()
        )

        if args.test:
            logger.info("Starting Video Test Mode...")
            tester = VideoTester(settings)
            asyncio.run(tester.run_test(args.test))
        else:
            logger.info("Starting ANPR Service...")
            service = ANPRService(settings)
            asyncio.run(service.start())

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")