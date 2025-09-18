"""
Configuration file for ResearchGPT Assistant

    1. Mistral API configuration
    2. Processing parameters (chunk size, overlap)
    3. Directory paths for data and results
    4. Model parameters (temperature, max tokens)
    5. Logging configuration
    6. Error handling
"""

import os
import requests
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import time
from pathlib import Path

class Config:
    def __init__(self):
        # Logging configuration
        self._setup_logging()

        # Load environment variables
        load_dotenv()
        
        # Mistral API settings
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # API key for Mistral
        self.MODEL_NAME = os.getenv("MODEL_NAME")  # Model name for Mistral
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))  # Temperature setting
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))  # Max tokens for response

        # API request settings
        self.MAX_RETRIES = self._get_int_env("MAX_RETRIES", 3, min_val=1, max_val=5)
        self.RETRY_DELAY = self._get_float_env("RETRY_DELAY", 1.0, min_val=0.1, max_val=5.0)
        self.REQUEST_TIMEOUT = self._get_float_env("REQUEST_TIMEOUT", 10.0, min_val=1.0, max_val=30.0)

        # Directory paths
        self.DATA_DIR = Path(os.getenv("DATA_DIR", "data/"))
        self.SAMPLE_PAPERS_DIR = self.DATA_DIR / "sample_papers"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results/"))
        
        # Processing parameters
        self.CHUNK_SIZE = self._get_int_env("CHUNK_SIZE", 500, min_val=100, max_val=2000)
        self.OVERLAP = self._get_int_env("OVERLAP", 100, min_val=0, max_val=self.CHUNK_SIZE//2)

        # Validate configuration
        self._validate_directories()
        self.validate_api_key()

    def _setup_logging(self) -> None:
        """Set up logging with console and file handlers."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            
            # File handler with rotation
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "researchgpt.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5 # keep last 5 logs
            )
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %I:%M %p'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def _get_float_env(self, key: str, default: float, min_val: float, max_val: float) -> float:
        """Get float environment variable with validation."""
        try:
            value = float(os.getenv(key, default))
            if not (min_val <= value <= max_val):
                self.logger.error(f"{key} must be between {min_val} and {max_val}")
                raise ValueError(f"{key} must be between {min_val} and {max_val}")
            return value
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid {key} value: {str(e)}")
            raise ValueError(f"Invalid {key} value: {str(e)}")

    def _get_int_env(self, key: str, default: int, min_val: int, max_val: int) -> int:
        """Get integer environment variable with validation."""
        try:
            value = int(os.getenv(key, default))
            if not (min_val <= value <= max_val):
                self.logger.error(f"{key} must be between {min_val} and {max_val}")
                raise ValueError(f"{key} must be between {min_val} and {max_val}")
            return value
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid {key} value: {str(e)}")
            raise ValueError(f"Invalid {key} value: {str(e)}")

    def _validate_directories(self) -> None:
        """Validate and create necessary directories."""
        directories = [self.DATA_DIR, self.SAMPLE_PAPERS_DIR, self.PROCESSED_DIR, self.RESULTS_DIR]
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                if not os.access(directory, os.W_OK):
                    self.logger.error(f"No write permission for directory: {directory}")
                    raise PermissionError(f"No write permission for directory: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create or validate directory {directory}: {str(e)}")
                raise RuntimeError(f"Failed to set up directory {directory}: {str(e)}")

    def validate_api_key(self) -> bool:
        """
        Validates the Mistral API key by checking its presence and making a test request.
        Returns True if valid, raises an exception if invalid or on failure.
        
        Raises:
            ValueError: If API key is missing or invalid
            RequestException: If API request fails after retries
        """
        if not self.MISTRAL_API_KEY or not self.MISTRAL_API_KEY.strip():
            self.logger.error("MISTRAL_API_KEY is not set or empty in environment variables")
            raise ValueError("MISTRAL_API_KEY is not set or empty in environment variables")

        api_key = self.MISTRAL_API_KEY.strip()
        headers = {"Authorization": f"Bearer {api_key}"}
        url = "https://api.mistral.ai/v1/models"

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                
                if response.status_code == 200:
                    self.logger.debug("MISTRAL_API_KEY validated successfully")
                    return True
                    
            except ConnectionError as e:
                self.logger.warning(f"Connection error on attempt {attempt + 1}/{self.MAX_RETRIES}: {str(e)}")
                if attempt == self.MAX_RETRIES - 1:
                    self.logger.error("Max retries reached for API key validation")
                    raise RequestException("Failed to validate API key due to connection issues")
                    
            except Timeout as e:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}/{self.MAX_RETRIES}: {str(e)}")
                if attempt == self.MAX_RETRIES - 1:
                    self.logger.error("Max retries reached for API key validation")
                    raise RequestException("Failed to validate API key due to timeout")
                    
            except HTTPError as e:
                if response.status_code in (401, 403):
                    self.logger.error("Invalid MISTRAL_API_KEY")
                    raise ValueError("Invalid MISTRAL_API_KEY")
                self.logger.warning(f"HTTP error on attempt {attempt + 1}/{self.MAX_RETRIES}: {str(e)}")
                
            except RequestException as e:
                self.logger.warning(f"Request failed on attempt {attempt + 1}/{self.MAX_RETRIES}: {str(e)}")
                
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY)
                
        raise RequestException("Failed to validate API key after multiple attempts")
