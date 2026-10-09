import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "Legal Metrology Vision API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Uploads storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    
    # Candidate Gemini multimodal vision models in priority order
    CANDIDATE_MODELS: List[str] = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash"
    ]
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]


settings = Settings()


def get_api_keys() -> List[str]:
    """
    Fetch configured Gemini API keys from environment variables.
    Supports single GEMINI_API_KEY or comma-separated GEMINI_API_KEYS with dynamic reloading.
    """
    load_dotenv(override=True)
    raw = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    keys = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
    return keys
