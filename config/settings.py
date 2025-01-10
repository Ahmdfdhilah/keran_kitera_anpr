from functools import lru_cache
from schemas.schema import Settings, ANPRConfig
from db.postgresql import get_cameras_from_db

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        cameras=get_cameras_from_db(),
        anpr=ANPRConfig(),
    )