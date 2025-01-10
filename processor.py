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
        pass