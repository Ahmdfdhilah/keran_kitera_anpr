import asyncio
import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
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

# FastAPI App Instance
app = FastAPI()

from config.settings import get_settings

settings = get_settings()


class ANPRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor = ANPRProcessor(settings)
        self.camera_manager = CameraManager()
        self.mqtt_service = MQTTService(settings, self.processor, self.camera_manager)

    async def process_http_image(self, image_bytes: bytes):
        try:
            # Process image
            image_np = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if image_np is None:
                raise ValueError("Invalid image data")

            detected_plates = await self.processor.process_frame(
                image_np, "http-upload"
            )
            results = []

            for plate in detected_plates:
                cropped_path = self.mqtt_service.save_cropped_plate(
                    image_np,
                    plate["left"],
                    plate["top"],
                    plate["width"],
                    plate["height"],
                    plate["text"],
                )
                with open(cropped_path, "rb") as img_file:
                    cropped_base64 = base64.b64encode(img_file.read()).decode("utf-8")

                results.append(
                    {
                        "plate_text": plate["text"],
                        "confidence": plate["confidence"],
                        "timestamp": plate["timestamp"].isoformat(),
                        "cropped_image": cropped_base64,
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Error processing HTTP image: {e}")
            raise


# HTTP Endpoint untuk upload gambar
@app.post("/anpr/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        results = await service.process_http_image(image_bytes)
        return JSONResponse(content={"status": "success", "results": results})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /anpr/upload/ endpoint: {e}")
        raise HTTPException(status_code=400, detail="Failed to process image")


# Main ANPR Service Setup
service = ANPRService(settings)
print(settings.cameras)


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

    # Modify the HTTP server task to avoid nested event loops
    http_task = asyncio.create_task(
        asyncio.to_thread(
            uvicorn.run,
            app,
            host=settings.http_host,
            port=settings.http_port,
            log_level="info",
        )
    )

    # Wait for both tasks to complete
    await asyncio.gather(mqtt_task, http_task)


# Jalankan aplikasi
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
