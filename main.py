import asyncio
import logging
from processor import ANPRProcessor
from camera_manager import CameraManager
from config.settings import Settings
from mqtt_service import MQTTService

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ANPRService")

from config.settings import get_settings

settings = get_settings()

class ANPRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor = ANPRProcessor(settings)
        self.camera_manager = CameraManager()
        self.mqtt_service = MQTTService(settings, self.processor, self.camera_manager)


# Main ANPR Service Setup
service = ANPRService(settings)

async def main():
    # Mulai MQTT Service di latar belakang
    service.mqtt_service.setup(asyncio.get_event_loop()) 

    await service.camera_manager.configure(
        camera_configs=settings.cameras,
        mappings={
            camera_id: (config.gate_id, config.direction)
            for camera_id, config in settings.cameras.items()
        },
    )

    mqtt_task = asyncio.create_task(service.mqtt_service.start_mqtt_async())

   
    await asyncio.gather(mqtt_task)


# Jalankan aplikasi
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down services...")