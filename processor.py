import torch
import cv2
import numpy as np
from easyocr import Reader
import asyncio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os

from ultralytics import YOLO
from config.settings import Settings

logger = logging.getLogger(__name__)

class ANPRProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings        
        # Load YOLOv11 model
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
        #                           path=settings.anpr.model_path, force_reload=True, trust_repo=True)

        self.model = YOLO(settings.anpr.model_path)
        
        # Configure model settings
        self.model.conf = settings.anpr.conf_threshold
        self.model.iou = settings.anpr.nms_threshold
        
        # Use CUDA if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.reader = Reader(['en'], gpu=torch.cuda.is_available())
        self.executor = ThreadPoolExecutor(max_workers=len(settings.cameras))


    async def process_frame(self, frame, camera_id: str):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        return await self._process_detections(results[0], frame, camera_id)

    async def _process_detections(self, result, frame, camera_id):
        detected_plates = []
        
        if len(result.boxes) > 0:
            box = result.boxes[0]  # Take first detection only
            
            if box.conf[0] >= self.settings.anpr.conf_threshold:
                # Get coordinates with padding
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height, width = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                # Ensure coordinates are within bounds
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(width, x2 + pad_x)
                y2 = min(height, y2 + pad_y)
                
                # Crop the plate region
                plate_region = frame[y1:y2, x1:x2].copy()
                
                # Process the plate for OCR
                processed_plate = self._preprocess_plate(plate_region)
                
                # Perform OCR on the processed plate image
                ocr_result = self.reader.readtext(processed_plate)
                
                if self.settings.debug:
                    timestamp = datetime.now().strftime('%H%M%S')
                    # Save both original and processed plates
                    original_path = os.path.join(
                        self.settings.result_path,
                        f"plate_original_{camera_id}_{timestamp}.jpg"
                    )
                    processed_path = os.path.join(
                        self.settings.result_path,
                        f"plate_processed_{camera_id}_{timestamp}.jpg"
                    )
                    cv2.imwrite(original_path, plate_region)
                    cv2.imwrite(processed_path, processed_plate)
                    logger.debug(f"Original plate saved: {original_path}")
                    logger.debug(f"Processed plate saved: {processed_path}")
                    logger.debug(f"OCR results: {ocr_result}")
                
                # Process OCR results
                if ocr_result:
                    best_ocr = max(ocr_result, key=lambda x: x[2])
                    text = best_ocr[1].upper().strip()
                    
                    if len(text) >= 4:
                        detected_plates.append({
                            'text': text,
                            'confidence': best_ocr[2],
                            'bbox': (x1, y1, x2, y2),
                            'plate_image': processed_plate  # Change this to return processed image
                        })
        
        return detected_plates

    def _preprocess_plate(self, plate_region):
        """Preprocess the plate image for better OCR results"""
        # Resize (upscale)
        processed = cv2.resize(plate_region, (0, 0), fx=2, fy=2)
        # Convert to grayscale
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        # Enhance contrast
        processed = cv2.equalizeHist(processed)
        return processed

    def _save_debug_images(self, original_plate, processed_plate, ocr_result, camera_id):
        """
        Save both original and processed plates for debugging
        """
        timestamp = datetime.now().strftime('%H%M%S')
        debug_dir = self.settings.result_path
        
        # Save original cropped plate
        original_path = os.path.join(
            debug_dir,
            f"plate_original_{camera_id}_{timestamp}.jpg"
        )
        cv2.imwrite(original_path, original_plate)
        
        # Save processed plate
        processed_path = os.path.join(
            debug_dir,
            f"plate_processed_{camera_id}_{timestamp}.jpg"
        )
        cv2.imwrite(processed_path, processed_plate)
        
        logger.debug(f"Original plate saved: {original_path}")
        logger.debug(f"Processed plate saved: {processed_path}")
        logger.debug(f"OCR results: {ocr_result}")

    def _process_ocr_results(self, ocr_result, x1, y1, x2, y2):
        """
        Process OCR results and return the best plate information
        """
        if not ocr_result:
            return None
        
        # Get the best result based on confidence
        best_ocr = max(ocr_result, key=lambda x: x[2])
        text = best_ocr[1].upper().strip()
        
        # Only return results that meet minimum length requirement
        if len(text) >= 4:
            return {
                'text': text,
                'confidence': best_ocr[2],
                'bbox': (x1, y1, x2, y2)
            }
        
        return None
    # async def _process_plate(self, frame, conf, x1, y1, x2, y2, camera_id):
    #     try:
    #         # Extract plate region with padding
    #         height, width = frame.shape[:2]
    #         pad_x = int((x2 - x1) * 0.1)  # 10% padding
    #         pad_y = int((y2 - y1) * 0.1)
            
    #         # Ensure coordinates are within frame bounds
    #         x1 = max(0, x1 - pad_x)
    #         y1 = max(0, y1 - pad_y)
    #         x2 = min(width, x2 + pad_x)
    #         y2 = min(height, y2 + pad_y)
            
    #         # Extract and process plate region
    #         plate_region = frame[y1:y2, x1:x2]
            
    #         # Process the cropped plate image
    #         processed_plate = cv2.resize(plate_region, (0, 0), fx=2, fy=2)  # Upscale
    #         processed_plate = cv2.cvtColor(processed_plate, cv2.COLOR_BGR2GRAY)
    #         processed_plate = cv2.equalizeHist(processed_plate)  # Enhance contrast
            
    #         # Save ONLY the processed and cropped plate for debugging
    #         if self.settings.debug:
    #             debug_path = os.path.join(self.settings.result_path, 
    #                                     f"plate_debug_{camera_id}_{datetime.now().strftime('%H%M%S')}.jpg")
    #             cv2.imwrite(debug_path, processed_plate)
    #             logger.debug(f"Saved cropped and processed plate image: {debug_path}")
            
    #         # Perform OCR ONLY on the processed plate image
    #         ocr_result = self.reader.readtext(processed_plate)
            
    #         if self.settings.debug:
    #             logger.debug(f"OCR results for cropped plate: {ocr_result}")
            
    #         if ocr_result and len(ocr_result) > 0:
    #             # Get the result that covers the largest area of the image
    #             best_result = max(ocr_result, key=lambda x: (
    #                 (x[0][2][0] - x[0][0][0]) * (x[0][2][1] - x[0][0][1])  # Area calculation
    #                 if len(x) >= 3 else 0
    #             ))
                
    #             if len(best_result) >= 3 and best_result[2] > 0.15:  # Lowered threshold since we're using area
    #                 text = best_result[1].upper().strip()
    #                 return {
    #                     'text': text,
    #                     'confidence': best_result[2],
    #                     'bbox': (x1, y1, x2, y2)
    #                 }
            
    #         return None
            
    #     except Exception as e:
    #         logger.error(f"Error processing plate region: {e}", exc_info=True)
    #         return None
    
    def _get_output_names(self):
        layersNames = self.net.getLayerNames()
        return [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    async def _save_results(self, plate_region, text, conf, camera_id, timestamp):
        # Ensure result directory exists
        os.makedirs(self.settings.result_path, exist_ok=True)
        
        # Save cropped plate image
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = f"{camera_id}_{text}_{timestamp_str}.jpg"
        image_path = os.path.join(self.settings.result_path, image_filename)
        
        cv2.imwrite(image_path, plate_region)
        
        # Save OCR result
        ocr_filename = f"{camera_id}_{text}_{timestamp_str}.txt"
        ocr_path = os.path.join(self.settings.result_path, ocr_filename)
        
        with open(ocr_path, 'w') as f:
            f.write(f"Plate: {text}\nConfidence: {conf}\nTimestamp: {timestamp}")