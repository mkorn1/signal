from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_model: str = "anthropic/claude-sonnet-4"
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"
    debug: bool = False
    
    # Audio generation settings
    stability_api_key: Optional[str] = None  # Required for Stable Audio 2
    replicate_api_token: Optional[str] = None  # Required for Demucs stem separation
    
    # Feature flags
    use_mock_audio: bool = True  # When True (default), return mock audio instead of calling Stable Audio API. Set to False to enable real API calls.

    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra env vars in .env


@lru_cache()
def get_settings() -> Settings:
    return Settings()
