import logging
import re
from typing import Optional, Tuple, Dict
from google.cloud import vision
from google.oauth2 import service_account
from config.settings import Settings
import os
logger = logging.getLogger(__name__)

class ANPRProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.token_path = os.path.join(os.path.dirname(__file__), self.settings.google_vision_token)
        self.client = None
        self.initialize_vision_client()
        
        # Indonesian license plate patterns
        self.plate_patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',    # Standard format (B 1234 ABC)
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,2}$',    # Shorter format (B 1234 AB)
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',          # No spaces (B1234ABC)
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,2}$'           # No spaces shorter (B1234AB)
        ]

    def initialize_vision_client(self):
        """Initialize Google Cloud Vision client with credentials"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.token_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
            logger.info("Successfully initialized Google Cloud Vision client")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision client: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocess text to handle common OCR issues"""
        # Convert to uppercase and remove extra spaces
        text = text.upper().strip()
        
        # Remove common OCR mistakes
        replacements = {
            'O': '0',
            'I': '1',
            'S': '5',
            'Z': '2',
            'G': '6',
            'B': 'B',  # Preserve B as it's common in Indonesian plates
            'D': 'D'   # Preserve D as it's common in Indonesian plates
        }
        
        # Apply replacements only to the numeric portion
        processed_text = ''
        in_number_section = False
        
        for char in text:
            if char.isdigit():
                in_number_section = True
                processed_text += char
            elif in_number_section and char in replacements:
                processed_text += replacements[char]
            else:
                in_number_section = False
                processed_text += char
                
        # Remove any non-alphanumeric characters except spaces
        processed_text = re.sub(r'[^A-Z0-9\s]', '', processed_text)
        
        # Normalize spaces
        processed_text = ' '.join(processed_text.split())
        
        return processed_text

    def is_valid_indonesian_plate(self, text: str) -> bool:
        """Validate if text matches Indonesian license plate format"""
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Try with and without spaces
        texts_to_check = [
            cleaned_text,
            ''.join(cleaned_text.split())  # Remove all spaces
        ]
        
        for test_text in texts_to_check:
            for pattern in self.plate_patterns:
                if re.match(pattern, test_text):
                    return True
        return False

    def select_best_plate_candidate(self, annotations: list) -> Optional[Tuple[str, float]]:
        """Select the best license plate candidate from OCR results"""
        candidates = []
        
        # First, check the full text annotation
        full_text = annotations[0].description if annotations else ""
        
        # Split the full text into lines and check each line
        for line in full_text.split('\n'):
            if self.is_valid_indonesian_plate(line):
                candidates.append((self.format_plate_number(line), 1.0))
        
        # Then check individual text annotations
        for annotation in annotations[1:]:  # Skip the first (full text) annotation
            text = annotation.description.strip()
            confidence = annotation.confidence
            
            if self.is_valid_indonesian_plate(text):
                candidates.append((self.format_plate_number(text), confidence))
        
        # Sort by confidence and return the best match
        if candidates:
            return sorted(candidates, key=lambda x: x[1], reverse=True)[0]
        return None

    async def process_image(self, image_bytes: bytes) -> Dict:
        """Process image and extract license plate"""
        try:
            if self.client is None:
                self.initialize_vision_client()

            image = vision.Image(content=image_bytes)
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(
                    f'{response.error.message}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'
                )

            annotations = response.text_annotations
            
            if not annotations:
                return {"success": False, "message": "No text detected"}

            # Find best plate candidate
            best_candidate = self.select_best_plate_candidate(annotations)
            
            if best_candidate:
                plate_number, confidence = best_candidate
                return {
                    "success": True,
                    "plate_number": plate_number,
                    "confidence": confidence,
                    "raw_text": annotations[0].description
                }
            else:
                return {
                    "success": False,
                    "message": "No valid license plate detected",
                    "raw_text": annotations[0].description
                }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"success": False, "message": f"Error processing image: {str(e)}"}

    def format_plate_number(self, plate: str) -> str:
        """Format license plate number to standard format"""
        # Clean and normalize the plate text first
        plate = self.preprocess_text(plate)
        
        # Remove all whitespace
        plate = ''.join(plate.split())
        
        # Find the first number position
        number_pos = -1
        for i, char in enumerate(plate):
            if char.isdigit():
                number_pos = i
                break
                
        if number_pos == -1:
            return plate
            
        # Split into parts
        area_code = plate[:number_pos]
        rest = plate[number_pos:]
        
        # Find where numbers end
        letters_pos = -1
        for i, char in enumerate(rest):
            if char.isalpha():
                letters_pos = i
                break
                
        if letters_pos == -1:
            numbers = rest
            suffix = ''
        else:
            numbers = rest[:letters_pos]
            suffix = rest[letters_pos:]
            
        # Combine with proper spacing
        return f"{area_code} {numbers} {suffix}".strip()