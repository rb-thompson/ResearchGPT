"""
Script to test Config class initialization and validation
"""

from config import Config
import logging

def main():
    # Use the existing logger from Config
    logger = logging.getLogger("config")
    
    try:
        # Initialize Config
        config = Config()
        logger.info("Configuration initialized successfully")
        logger.info("Configuration check completed successfully")
        logger.info(f"Model: {config.MODEL_NAME}")
        logger.info(f"Temperature: {config.TEMPERATURE}")
        logger.info(f"Max Tokens: {config.MAX_TOKENS}")
        logger.info(f"Data Directory: {config.DATA_DIR}")
        logger.info(f"Results Directory: {config.RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Configuration check failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()