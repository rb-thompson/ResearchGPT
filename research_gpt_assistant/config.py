import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import requests
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from dotenv import load_dotenv
import time


class Config:
    """Configuration class for ResearchGPT Assistant.

    Manages API settings, file paths, logging, and processing parameters.
    Loads environment variables and validates configuration on initialization.
    """

    def __init__(self):
        """Initialize configuration with logging, environment variables, and validation."""
        # Setup logging
        self._setup_logging()

        # Load environment variables
        load_dotenv()

        # Mistral API settings
        self.MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
        self.MODEL_NAME: Optional[str] = os.getenv("MODEL_NAME")
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.1))
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 1000))

        # API request settings
        self.MAX_RETRIES: int = self._get_int_env("MAX_RETRIES", 3, 1, 5)
        self.RETRY_DELAY: float = self._get_float_env("RETRY_DELAY", 1.0, 0.1, 5.0)
        self.REQUEST_TIMEOUT: float = self._get_float_env("REQUEST_TIMEOUT", 10.0, 1.0, 30.0)

        # Directory paths
        self.DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data/"))
        self.SAMPLE_PAPERS_DIR: Path = self.DATA_DIR / "sample_papers"
        self.PROCESSED_DIR: Path = self.DATA_DIR / "processed"
        self.RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR", "results/"))

        # Processing parameters
        self.CHUNK_SIZE: int = self._get_int_env("CHUNK_SIZE", 500, 100, 2000)
        self.OVERLAP: int = self._get_int_env("OVERLAP", 100, 0, self.CHUNK_SIZE // 2)
        self.MIN_CHUNK_SIZE: int = self._get_int_env("MIN_CHUNK_SIZE", 250, 50, self.CHUNK_SIZE)

        # Validate configuration
        self._validate_directories()
        self.validate_api_key()

    def _setup_logging(self) -> None:
        """Configure logging with console and rotating file handlers."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Suppress pdf parsing warnings
        logging.getLogger("pdfplumber").setLevel(logging.ERROR)
        logging.getLogger("pdfminer").setLevel(logging.ERROR)

        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            # File handler with rotation
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "researchgpt.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setLevel(logging.DEBUG)

            # Formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %I:%M %p",
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def _get_float_env(self, key: str, default: float, min_val: float, max_val: float) -> float:
        """Retrieve and validate a float environment variable.

        Args:
            key: Environment variable name.
            default: Default value if variable is not set.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            Validated float value.

        Raises:
            ValueError: If the value is invalid or out of range.
        """
        try:
            value = float(os.getenv(key, default))
            if not min_val <= value <= max_val:
                raise ValueError(f"{key} must be between {min_val} and {max_val}")
            return value
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid {key} value: {e}")
            raise ValueError(f"Invalid {key} value: {e}")

    def _get_int_env(self, key: str, default: int, min_val: int, max_val: int) -> int:
        """Retrieve and validate an integer environment variable.

        Args:
            key: Environment variable name.
            default: Default value if variable is not set.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            Validated integer value.

        Raises:
            ValueError: If the value is invalid or out of range.
        """
        try:
            value = int(os.getenv(key, default))
            if not min_val <= value <= max_val:
                raise ValueError(f"{key} must be between {min_val} and {max_val}")
            return value
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid {key} value: {e}")
            raise ValueError(f"Invalid {key} value: {e}")

    def _validate_directories(self) -> None:
        """Validate and create necessary directories.

        Raises:
            PermissionError: If a directory is not writable.
            RuntimeError: If a directory cannot be created.
        """
        directories = [self.DATA_DIR, self.SAMPLE_PAPERS_DIR, self.PROCESSED_DIR, self.RESULTS_DIR]
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                if not os.access(directory, os.W_OK):
                    self.logger.error(f"No write permission for directory: {directory}")
                    raise PermissionError(f"No write permission for directory: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
                raise RuntimeError(f"Failed to create directory {directory}: {e}")

    def validate_api_key(self) -> bool:
        """Validate the Mistral API key with a test request.

        Returns:
            True if the API key is valid.

        Raises:
            ValueError: If the API key is missing or invalid.
            RequestException: If the API request fails after retries.
        """
        if not self.MISTRAL_API_KEY or not self.MISTRAL_API_KEY.strip():
            self.logger.error("MISTRAL_API_KEY is not set or empty")
            raise ValueError("MISTRAL_API_KEY is not set or empty")

        headers = {"Authorization": f"Bearer {self.MISTRAL_API_KEY.strip()}"}
        url = "https://api.mistral.ai/v1/models"

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                self.logger.info("MISTRAL_API_KEY validated successfully")
                return True
            except ConnectionError as e:
                self.logger.warning(f"Connection error on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise RequestException("Failed to validate API key due to connection issues")
            except Timeout as e:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise RequestException("Failed to validate API key due to timeout")
            except HTTPError as e:
                if response.status_code in (401, 403):
                    self.logger.error("Invalid MISTRAL_API_KEY")
                    raise ValueError("Invalid MISTRAL_API_KEY")
                self.logger.warning(f"HTTP error on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}")
            except RequestException as e:
                self.logger.warning(f"Request failed on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}")
            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY)
        raise RequestException("Failed to validate API key after multiple attempts")