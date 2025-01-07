import torch
import cv2
import numpy as np
from easyocr import Reader
import asyncio
import logging
from datetime import datetime
import os
from PIL import Image, ImageEnhance
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import json
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("anpr.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

@dataclass
class Parameters:
    """Configuration parameters for ANPR system"""
    # Model parameters
    imgsz: int = 640
    conf_thres: float = 0.25
    max_det: int = 1000
    model_path: str = "models/train_with_nms_timelimit_custom_best.pt"

    # OCR parameters optimized for license plates
    min_text_length: int = 3
    min_ocr_confidence: float = 0.3
    allowlist: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

    # EasyOCR specific parameters
    canvas_size: int = 2560
    mag_ratio: float = 1.0
    text_threshold: float = 0.7
    link_threshold: float = 0.4
    low_text: float = 0.4
    slope_ths: float = 0.1
    ycenter_ths: float = 0.5
    height_ths: float = 0.5
    width_ths: float = 0.5
    add_margin: float = 0.1

    # Image processing parameters
    target_height: int = 64  # Optimized for license plate height
    contrast_factor: float = 1.5
    sharpness_factor: float = 1.5
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # Visualization parameters
    color_blue: Tuple[int, int, int] = (255, 255, 0)
    color_red: Tuple[int, int, int] = (25, 20, 240)
    font_scale: float = 0.7
    thickness: int = 2
    rect_thickness: int = 3

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LicensePlateOCR:
    """Enhanced OCR processing using EasyOCR techniques"""

    def __init__(self, params: Parameters):
        self.params = params
        self.reader = Reader(
            ["en"],
            gpu=torch.cuda.is_available(),
            model_storage_directory="./models",
            download_enabled=True,
            recog_network="english_g2",
            detect_network="craft",  # Using CRAFT for better text detection
            quantize=True  # Enable quantization for better performance
        )
        self.debug_dir = "debug_images"
        os.makedirs(self.debug_dir, exist_ok=True)

    def preprocess_plate(self, plate_region: np.ndarray, plate_id: str) -> List[np.ndarray]:
        """Enhanced preprocessing pipeline based on EasyOCR techniques"""
        try:
            # Resize with maintained aspect ratio
            aspect_ratio = plate_region.shape[1] / plate_region.shape[0]
            target_width = int(self.params.target_height * aspect_ratio)
            resized = cv2.resize(plate_region, (target_width, self.params.target_height))

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.params.clahe_clip_limit,
                tileGridSize=self.params.clahe_grid_size
            )
            equalized = clahe.apply(gray)

            # Create multiple processing variations
            processed_images = []

            # Original equalized
            processed_images.append(cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB))

            # Otsu thresholding
            _, binary_otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2RGB))

            # Adaptive thresholding
            adaptive = cv2.adaptiveThreshold(
                equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB))

            # Enhanced contrast version
            pil_img = Image.fromarray(equalized)
            enhancer = ImageEnhance.Contrast(pil_img)
            contrast_img = enhancer.enhance(self.params.contrast_factor)
            processed_images.append(cv2.cvtColor(np.array(contrast_img), cv2.COLOR_RGB2BGR))

            # Save debug images
            debug_path = os.path.join(self.debug_dir, f"plate_{plate_id}")
            os.makedirs(debug_path, exist_ok=True)

            for idx, img in enumerate(processed_images):
                cv2.imwrite(os.path.join(debug_path, f"variation_{idx}.jpg"), img)

            return processed_images

        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return [cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)]

    def recognize_plate(self, image: np.ndarray, plate_id: str) -> List[Tuple[str, float]]:
        """Improved OCR using EasyOCR techniques"""
        try:
            all_results = []
            image_variations = self.preprocess_plate(image, plate_id)

            for img_variant in image_variations:
                # Using EasyOCR's advanced parameters
                results = self.reader.readtext(
                    img_variant,
                    decoder='beamsearch',
                    beamWidth=5,
                    batch_size=1,
                    workers=0,
                    allowlist=self.params.allowlist,
                    paragraph=False,
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                    text_threshold=self.params.text_threshold,
                    low_text=self.params.low_text,
                    link_threshold=self.params.link_threshold,
                    canvas_size=self.params.canvas_size,
                    mag_ratio=self.params.mag_ratio,
                    slope_ths=self.params.slope_ths,
                    ycenter_ths=self.params.ycenter_ths,
                    height_ths=self.params.height_ths,
                    width_ths=self.params.width_ths,
                    add_margin=self.params.add_margin
                )

                for bbox, text, conf in results:
                    if len(text) >= self.params.min_text_length and conf >= self.params.min_ocr_confidence:
                        # Clean and format the text
                        cleaned_text = ''.join(c for c in text if c in self.params.allowlist)
                        if cleaned_text:
                            all_results.append((cleaned_text, conf))

            # Sort by confidence and remove duplicates
            unique_results = []
            seen_texts = set()

            # Sort results by confidence
            sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)

            for text, conf in sorted_results:
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_results.append((text, conf))

            if unique_results:
                logger.info(f"OCR results for plate {plate_id}: {unique_results}")
            else:
                logger.warning(f"No valid OCR results for plate {plate_id}")

            return unique_results

        except Exception as e:
            logger.error(f"OCR error for plate {plate_id}: {str(e)}")
            return []

class ANPRProcessor:
    """Main ANPR processing class"""

    def __init__(self, settings: Dict):
        self.settings = settings
        self.params = Parameters()
        self.model = self.load_model()
        self.ocr_processor = LicensePlateOCR(self.params)
        self.setup_directories()

    def load_model(self) -> YOLO:
        model = YOLO(self.params.model_path)
        model.conf = self.params.conf_thres
        model.max_det = self.params.max_det
        model.to(self.params.device)
        return model

    def setup_directories(self):
        self.output_dir = "output"
        self.crops_dir = os.path.join(self.output_dir, "crops")
        self.results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def save_result(self, result: Dict, image: np.ndarray, plate_id: str):
        crop_path = os.path.join(
            self.crops_dir,
            f"{plate_id}_{result['plate_number']}_{result['confidence']:.2f}.jpg"
        )
        cv2.imwrite(crop_path, image)
        
        result["crop_path"] = crop_path
        result_path = os.path.join(self.results_dir, f"{plate_id}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    async def process_frame(self, frame: np.ndarray) -> List[Dict]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        return await self._process_detections(results[0], frame)

    async def _process_detections(self, result, frame: np.ndarray) -> List[Dict]:
        detected_plates = []

        for box in result.boxes:
            if box.cls[0].item() == 2 and box.conf[0].item() >= self.params.conf_thres:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_region = frame[y1:y2, x1:x2].copy()
                    if plate_region.size == 0:
                        continue

                    plate_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    ocr_results = self.ocr_processor.recognize_plate(plate_region, plate_id)

                    if ocr_results:
                        plate_text, confidence = ocr_results[0]
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "plate_number": plate_text,
                            "confidence": float(confidence),
                            "detection_confidence": float(box.conf[0]),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        }

                        self.save_result(result, plate_region, plate_id)
                        detected_plates.append(result)

                        # Draw detection results
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            self.params.color_blue,
                            self.params.rect_thickness,
                        )
                        cv2.putText(
                            frame,
                            f"{plate_text} ({confidence:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.params.font_scale,
                            self.params.color_blue,
                            self.params.thickness,
                        )

                        logger.info(f"Detected plate: {plate_text} with confidence: {confidence:.2f}")

                except Exception as e:
                    logger.error(f"Error processing detection: {str(e)}")

        return detected_plates