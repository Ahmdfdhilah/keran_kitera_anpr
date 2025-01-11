import asyncio
import logging
from processor import ANPRProcessor
from camera_manager import CameraManager
from schemas.schema import Settings
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

    async def start(self):
        # Configure cameras
        await self.camera_manager.configure(
            camera_configs=settings.cameras,
            mappings={
                camera_id: (config.gate_id, config.direction)
                for camera_id, config in settings.cameras.items()
            },
        )
        # Start MQTT service
        self.mqtt_service._start()

    async def stop(self):
        await self.mqtt_service.stop()

async def main():
    service = ANPRService(settings)
    
    try:
        await service.start()
        # Keep the service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())