import asyncio
import json
import logging
import base64
from datetime import datetime
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
        self.mqtt_client = MQTTClient(protocol=mqtt.MQTTv5)
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

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info("Connected to MQTT Broker")
            client.subscribe("portal/anpr/+/+/request", qos=2)
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("Unexpected MQTT disconnection. Reconnecting...")

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

        # Ensure the loop is properly set before scheduling the async task
        if self.loop is None:
            logger.error("Event loop is not initialized.")
            return

        task = asyncio.run_coroutine_threadsafe(
            self.process_anpr_mqtt(gate_id, direction, identifier), self.loop
        )
        task.add_done_callback(self._handle_task_result)

    def _handle_task_result(self, task):
        """Callback to handle result of the task"""
        if task.exception():
            logger.error(f"Error processing ANPR task: {task.exception()}")
        else:
            logger.info("Successfully processed ANPR")

    async def process_anpr_mqtt(self, gate_id: str, direction: str, identifier: str):
        try:
            # Ambil kamera berdasarkan gate_id dan arah
            camera = self.camera_manager.get_camera(gate_id, direction)
            if not camera:
                logger.error(
                    f"No camera configured for gate {gate_id} direction {direction}"
                )
                return

            # Ambil frame dari kamera
            frame = await camera.get_frame()
            if frame is None:
                logger.error(
                    f"Failed to capture frame from camera {gate_id}-{direction}"
                )
                return

            # Simpan frame sebagai screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = (
                f"{self.settings.result_path}/{gate_id}_{direction}_{timestamp}.jpg"
            )
            cv2.imwrite(screenshot_path, frame)
            logger.info(f"Screenshot saved at {screenshot_path}")

            # Publikasikan response MQTT dengan identifier dummy
            await self.publish_mqtt_response(gate_id, direction, screenshot_path)

        except Exception as e:
            logger.error(f"Error in dummy ANPR processing: {e}")

    async def publish_mqtt_response(
        self, gate_id, direction, image_path
    ):
        try:
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            response_topic = f"portal/anpr/{gate_id}/{direction}/response"
            payload = {
                "identifier": "HIDISAO2",
                "plate_text": "HIDISAO2",
                "confidence": 0,
                "timestamp": datetime.now().isoformat(),
                "image": image_base64,
            }

            # Run blocking publish in a thread-safe manner
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._publish_mqtt_response, response_topic, payload
            )

            logger.info(f"Published response to {response_topic}")

        except Exception as e:
            logger.error(f"Error publishing MQTT response: {e}")

    def _publish_mqtt_response(self, topic, payload):
        """Blocking MQTT publish function"""
        self.mqtt_client.publish(topic, json.dumps(payload), qos=2)

    def save_cropped_plate(self, frame, left, top, width, height, plate_text):
        cropped = frame[top : top + height, left : left + width]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cropped_path = f"{self.settings.result_path}/{plate_text}_{timestamp}.jpg"
        cv2.imwrite(cropped_path, cropped)
        return cropped_path

    async def stop(self):
        """Stop MQTT client gracefully"""
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
