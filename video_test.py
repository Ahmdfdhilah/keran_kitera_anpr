import cv2
import asyncio
import logging
import json
import os
from datetime import datetime
from processor import ANPRProcessor
from config.settings import Settings

logger = logging.getLogger(__name__)

class VideoTester:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor = ANPRProcessor(settings)
        self.dataset_path = "dataset"
        self.result_path = settings.result_path

    async def run_test(self, video_file: str):
        video_path = video_file if os.path.isfile(video_file) else os.path.join(self.dataset_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.imshow('Press S to capture frame, Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                await self._process_test_frame(frame)
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    async def _process_test_frame(self, frame):
        try:
            os.makedirs(self.result_path, exist_ok=True)
            
            # Process frame
            plates = await self.processor.process_frame(frame, "test")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if plates and len(plates) > 0:
                best_detection = max(plates, key=lambda x: x['confidence'])
                
                if best_detection['confidence'] > 0.3:
                    # Save the processed plate image
                    plate_path = os.path.join(
                        self.result_path, 
                        f"plate_{best_detection['text']}_{timestamp}.jpg"
                    )
                    cv2.imwrite(plate_path, best_detection['plate_image'])
                    
                    result = {
                        "timestamp": timestamp,
                        "plate_number": best_detection['text'],
                        "confidence": float(best_detection['confidence']),
                        "plate_path": plate_path
                    }
                    
                    # Log results
                    log_path = os.path.join(self.result_path, "test_results.log")
                    with open(log_path, "a") as f:
                        json.dump(result, f)
                        f.write("\n")
                        
                    logger.info(f"Detected plate: {best_detection['text']} ({best_detection['confidence']:.2f})")
                    logger.info(f"Saved plate to: {plate_path}")
                    logger.info(f"Updated log at: {log_path}")
                else:
                    logger.warning("Detection confidence too low")
            else:
                logger.warning("No plate detected")
                
        except Exception as e:
            logger.error(f"Error processing test frame: {e}", exc_info=True)