#!/usr/bin/env python3
"""
Faiz Chatbot - Main application entry point
This script launches the chatbot interface and initializes all necessary components.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    logger.info("Starting Faiz Chatbot...")
    
    try:
        # Import here to avoid circular imports
        from frontend.app import run_app
        
        # Start the web interface
        run_app()
        
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()