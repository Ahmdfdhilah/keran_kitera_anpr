import logging
from config.settings import Settings

logger = logging.getLogger(__name__)

class ANPRProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        pass