import cv2
import numpy as np
from easyocr import Reader
import asyncio
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
from config.settings import Settings

logger = logging.getLogger(__name__)

class ANPRProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize neural network with CUDA if available
        self.net = cv2.dnn.readNetFromDarknet(
            settings.anpr.config_path,
            settings.anpr.weights_path
        )
        
        # Check for CUDA availability
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("CUDA is available. Using GPU acceleration.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            logger.info("CUDA is not available. Using CPU.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.classes = []
        with open(settings.anpr.names_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
            
        self.reader = Reader(['en'], gpu=cv2.cuda.getCudaEnabledDeviceCount() > 0)
        self.executor = ThreadPoolExecutor(max_workers=len(settings.cameras))

    async def process_frame(self, frame, camera_id: str):
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255, 
            (self.settings.anpr.input_width, self.settings.anpr.input_height),
            [0,0,0],
            1,
            crop=False
        )
        
        self.net.setInput(blob)
        outs = self.net.forward(self._get_output_names())

        frameHeight, frameWidth = frame.shape[:2]
        
        # Use the _process_detections method to handle detection logic
        detected_plates = await self._process_detections(outs, frameHeight, frameWidth, frame, camera_id)

        return detected_plates

    async def _process_detections(self, outs, frameHeight, frameWidth, frame, camera_id):
        classIds, confidences, boxes = [], [], []
        detected_plates = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                
                if confidence > self.settings.anpr.conf_threshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.settings.anpr.conf_threshold,
            self.settings.anpr.nms_threshold
        )

        for i in indices:
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            plate_info = await self._process_plate(
                frame,
                classIds[i],
                confidences[i],
                left,
                top,
                width,
                height,
                camera_id
            )
            if plate_info:
                plate_info.update({
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height
                })
                detected_plates.append(plate_info)

        return detected_plates

    async def _process_plate(self, frame, classId, conf, left, top, width, height, camera_id):
        try:
            plate_region = frame[top:top+height, left:left+width]
            ocr_result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.reader.readtext,
                plate_region
            )
            
            best_text = None
            best_conf = 0
            
            for _, text, prob in ocr_result:
                if prob > best_conf:
                    best_text = text.upper()
                    best_conf = prob
            
            if best_text and best_conf > self.settings.anpr.conf_threshold:
                timestamp = datetime.now()
                
                # Save results
                # await self._save_results(
                #     plate_region,
                #     best_text,
                #     best_conf,
                #     camera_id,
                #     timestamp
                # )
                
                return {
                    'text': best_text,
                    'confidence': best_conf,
                    'timestamp': timestamp,
                    'camera_id': camera_id
                }
                
        except Exception as e:
            logger.error(f"Error processing plate: {e}")
            
        return None

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