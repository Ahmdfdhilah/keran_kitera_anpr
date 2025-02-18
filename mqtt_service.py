import asyncio
import json
import logging
import base64
from datetime import datetime
import time
import cv2
from paho.mqtt.client import Client as MQTTClient
from processor import ANPRProcessor
from camera_manager import CameraManager
import paho.mqtt.client as mqtt
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("MQTTService")

class MQTTService:
    def __init__(
        self, settings, processor: ANPRProcessor, camera_manager: CameraManager
    ):
        self.settings = settings
        self.processor = processor
        self.camera_manager = camera_manager
        self.mqtt_client = MQTTClient(protocol=mqtt.MQTTv311)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect

        # ThreadPoolExecutor to handle blocking MQTT functions
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.running = asyncio.Event()
        self.loop = None

    def _start(self):
        """Start MQTT client in a separate thread to avoid blocking."""
        try:
            self.loop = asyncio.get_running_loop() 
            self.mqtt_client.connect(self.settings.mqtt_broker, self.settings.mqtt_port)
            self.mqtt_client.loop_start()
            logger.info("MQTT Service started")
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
            self._reconnect()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info("Connected to MQTT Broker")
            client.subscribe("portal/anpr/+/+/request", qos=2)
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self._reconnect()

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """
        Handle MQTT disconnection events.
        Added properties parameter to match MQTT v5.0 callback signature.
        
        Args:
            client: MQTT client instance
            userdata: User data of any type
            rc: Return code for disconnect
            properties: MQTT v5.0 properties (optional)
        """
        if rc != 0:
            logger.warning("Unexpected MQTT disconnection. Reconnecting...")
            self._reconnect()

    def _reconnect(self):
        """Handle automatic reconnect."""
        while True:
            try:
                logger.info("Attempting to reconnect to MQTT broker...")
                self.mqtt_client.connect(self.settings.mqtt_broker, self.settings.mqtt_port)
                self.mqtt_client.loop_start()
                logger.info("Reconnected to MQTT Broker")
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                time.sleep(5)

    def _on_message(self, client, userdata, message):
        topic_parts = message.topic.split("/")
        if (
            len(topic_parts) != 5
            or topic_parts[0] != "portal"
            or topic_parts[1] != "anpr"
        ):
            logger.error(f"Invalid MQTT topic: {message.topic}")
            return

        gate_id = topic_parts[2]
        direction = topic_parts[3]
        payload = json.loads(message.payload.decode("utf-8"))
        identifier = payload.get("identifier", "unknown")

        if self.loop is None:
            logger.error("Event loop is not initialized.")
            return

        task = asyncio.run_coroutine_threadsafe(
            self.process_anpr_mqtt(gate_id, direction, identifier), self.loop
        )
        task.add_done_callback(self._handle_task_result)

    def _handle_task_result(self, task):
        if task.exception():
            logger.error(f"Error processing ANPR task: {task.exception()}")
        else:
            logger.info("Successfully processed ANPR")

    async def process_anpr_mqtt(self, gate_id: str, direction: str, identifier: str):
        try:
            # Get camera based on gate_id and direction
            camera = self.camera_manager.get_camera(gate_id, direction)
            if not camera:
                logger.error(f"No camera configured for gate {gate_id} direction {direction}")
                await self.publish_mqtt_response(
                    gate_id,
                    direction,
                    None,
                    identifier,
                    "CAMERA_ERROR",
                    0.0,
                    None,
                    "Camera not configured"
                )
                return

            # Capture frame from camera
            original_frame = await camera.get_frame()
            if original_frame is None:
                logger.error(f"Failed to capture frame from camera {gate_id}-{direction}")
                await self.publish_mqtt_response(
                    gate_id,
                    direction,
                    None,
                    identifier,
                    "CAPTURE_ERROR",
                    0.0,
                    None,
                    "Failed to capture frame"
                )
                return

            # Save original frame for Google Vision API
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = f"{self.settings.result_path}/original_{gate_id}_{direction}_{timestamp}.jpg"
            cv2.imwrite(original_path, original_frame)

            # Convert frame to bytes for Google Vision API
            _, img_encoded = cv2.imencode('.jpg', original_frame)
            img_bytes = img_encoded.tobytes()

            # Process with Google Vision API
            anpr_result = await self.processor.process_image(img_bytes)
            
            # Resize frame for MQTT transmission
            resized_frame = cv2.resize(original_frame, (640, 480))  # Changed from await to direct cv2 call
            screenshot_path = f"{self.settings.result_path}/{gate_id}_{direction}_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, resized_frame)

            # Initialize plate_image_path as None
            plate_image_path = None

            if anpr_result["success"]:
                # If plate detected, save cropped plate image
                if "bounding_poly" in anpr_result:
                    vertices = anpr_result["bounding_poly"].vertices
                    left = min(vertex.x for vertex in vertices)
                    top = min(vertex.y for vertex in vertices)
                    right = max(vertex.x for vertex in vertices)
                    bottom = max(vertex.y for vertex in vertices)
                    
                    plate_image_path = self.save_cropped_plate(
                        original_frame,
                        left, top,
                        right - left,
                        bottom - top,
                        anpr_result["plate_number"]
                    )
            else:
                # Set default values for failed detection
                anpr_result["plate_number"] = "TIDAK TERBACA"
                anpr_result["confidence"] = 0.0

            # Publish MQTT response
            await self.publish_mqtt_response(
                gate_id,
                direction,
                screenshot_path,
                identifier,
                anpr_result["plate_number"],
                anpr_result.get("confidence", 0),
                plate_image_path,
                anpr_result.get("message", "No plate detected")
            )

        except Exception as e:
            logger.error(f"Error in ANPR processing: {e}")
            await self.publish_mqtt_response(
                gate_id,
                direction,
                None,
                identifier,
                "SYSTEM_ERROR",
                0.0,
                None,
                str(e)
            )

    async def publish_mqtt_response(
        self,
        gate_id,
        direction,
        image_path,
        identifier,
        plate_text,
        confidence,
        plate_image_path=None,
        error_message=None
    ):
        try:
            # Initialize image_base64 as None
            image_base64 = None
            if image_path:
                with open(image_path, "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            # Initialize plate_image_base64 as None
            plate_image_base64 = None
            if plate_image_path:
                with open(plate_image_path, "rb") as plate_file:
                    plate_image_base64 = base64.b64encode(plate_file.read()).decode("utf-8")

            response_topic = f"portal/anpr/{gate_id}/{direction}/response"
            payload = {
                "identifier": identifier,
                "plate_text": plate_text,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "image": image_base64,
                "plate_image": plate_image_base64,
                "status": "success" if plate_text not in ["TIDAK TERBACA", "CAMERA_ERROR", "CAPTURE_ERROR", "SYSTEM_ERROR"] else "error",
                "message": error_message
            }

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._publish_mqtt_response,
                response_topic,
                payload
            )

            logger.info(f"Published response to {response_topic} with plate {plate_text}")

        except Exception as e:
            logger.error(f"Error publishing MQTT response: {e}")
            
    def _publish_mqtt_response(self, topic, payload):
        self.mqtt_client.publish(topic, json.dumps(payload), qos=2)

    def save_cropped_plate(self, frame, left, top, width, height, plate_text):
        try:
            cropped = frame[int(top):int(top + height), int(left):int(left + width)]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cropped_path = f"{self.settings.result_path}/plate_{plate_text}_{timestamp}.jpg"
            cv2.imwrite(cropped_path, cropped)
            return cropped_path
        except Exception as e:
            logger.error(f"Error saving cropped plate: {e}")
            return None

    async def stop(self):
        try:
            logger.info("Stopping MQTT service...")
            self.running.clear()
            self.mqtt_client.publish("client/status", "offline", qos=1, retain=True)
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.executor.shutdown(wait=True)
            logger.info("MQTT client stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping MQTT service: {e}")