"""
Configuration settings for PersonalDataAI
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Settings:
    """Application settings loaded from environment variables"""
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    
    # API Configuration
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    data_gov_api_base: str = "https://catalog.data.gov/api/3"
    
    # Application Settings
    max_datasets_to_fetch: int = 20
    cache_duration_hours: int = 24
    log_level: str = "INFO"
    
    # File Paths
    cache_dir: Path = Path("./cache")
    data_dir: Path = Path("./data")
    logs_dir: Path = Path("./logs")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.cache_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

def load_settings() -> Settings:
    """Load settings from environment variables"""
    
    settings = Settings(
        # Ollama
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        
        # APIs
        kaggle_username=os.getenv("KAGGLE_USERNAME"),
        kaggle_key=os.getenv("KAGGLE_KEY"),
        data_gov_api_base=os.getenv("DATA_GOV_API_BASE", "https://catalog.data.gov/api/3"),
        
        # App settings
        max_datasets_to_fetch=int(os.getenv("MAX_DATASETS_TO_FETCH", "20")),
        cache_duration_hours=int(os.getenv("CACHE_DURATION_HOURS", "24")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        
        # Paths
        cache_dir=Path(os.getenv("CACHE_DIR", "./cache")),
        data_dir=Path(os.getenv("DATA_DIR", "./data")),
        logs_dir=Path(os.getenv("LOGS_DIR", "./logs"))
    )
    
    return settings

def validate_settings(settings: Settings) -> bool:
    """Validate that required settings are properly configured"""
    
    # Check if Ollama is accessible
    try:
        import requests
        response = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception(f"Ollama not accessible at {settings.ollama_host}")
    except Exception as e:
        raise Exception(f"Cannot connect to Ollama: {e}")
    
    # Check if model is available
    try:
        import requests
        response = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
        models = response.json()
        available_models = [model["name"] for model in models.get("models", [])]
        if settings.ollama_model not in available_models:
            raise Exception(f"Model {settings.ollama_model} not found. Available: {available_models}")
    except Exception as e:
        raise Exception(f"Error checking Ollama models: {e}")
    
    return True