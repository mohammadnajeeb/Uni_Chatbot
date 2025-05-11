#!/usr/bin/env python3
"""
Script to run the entire Faiz Chatbot pipeline from data collection to interface.
"""
import os
import sys
import argparse
import subprocess
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the Faiz Chatbot pipeline.')
    parser.add_argument(
        '--skip-scrape', 
        action='store_true',
        help='Skip the web scraping step'
    )
    parser.add_argument(
        '--skip-ingest', 
        action='store_true',
        help='Skip the data ingestion step'
    )
    parser.add_argument(
        '--max-pages', 
        type=int, 
        default=int(os.getenv('MAX_PAGES_TO_SCRAPE', 100)),
        help='Maximum number of pages to scrape per site'
    )
    
    return parser.parse_args()

def run_command(command, description):
    """Run a shell command and log the output."""
    logger.info(f"Starting: {description}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        )
        
        logger.info(f"Output: {result.stdout}")
        logger.info(f"Completed: {description}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    
    # Step 1: Web scraping
    if not args.skip_scrape:
        logger.info("Step 1: Web scraping")
        
        scrape_command = [
            sys.executable,
            "scrapers/main_scraper.py",
            "--max-pages", str(args.max_pages)
        ]
        
        if not run_command(scrape_command, "Web scraping"):
            logger.error("Web scraping failed. Aborting pipeline.")
            return 1
    else:
        logger.info("Skipping web scraping step.")
    
    # Step 2: Data ingestion
    if not args.skip_ingest:
        logger.info("Step 2: Data ingestion")
        
        ingest_command = [
            sys.executable,
            "chatbot/ingest.py"
        ]
        
        if not run_command(ingest_command, "Data ingestion"):
            logger.error("Data ingestion failed. Aborting pipeline.")
            return 1
    else:
        logger.info("Skipping data ingestion step.")
    
    # Step 3: Run the chatbot interface
    logger.info("Step 3: Starting the chatbot interface")
    
    interface_command = [
        "streamlit", "run",
        "frontend/app.py",
        "--server.port=8501",
        "--browser.serverAddress=localhost",
        "--server.headless=false"
    ]
    
    run_command(interface_command, "Chatbot interface")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())