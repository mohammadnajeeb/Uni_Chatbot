#!/usr/bin/env python3
"""
Main script to run all scrapers and collect data for the Faiz Chatbot.
"""
import os
import sys
import time
import logging
import argparse
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('data', 'scraper.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import scrapers
from scrapers.amu_scraper import AMUScraper
from scrapers.amu_exams_scraper import AMUExamsScraper

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run web scrapers for Faiz Chatbot.')
    parser.add_argument(
        '--site', 
        choices=['amu', 'exams', 'all'], 
        default='all',
        help='Which site to scrape: amu, exams, or all'
    )
    parser.add_argument(
        '--max-pages', 
        type=int, 
        default=int(os.getenv('MAX_PAGES_TO_SCRAPE', 1000)),
        help='Maximum number of pages to scrape per site'
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=float(os.getenv('SCRAPE_DELAY', 1.5)),
        help='Delay between requests in seconds'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=os.path.join('data', 'raw'),
        help='Directory to save scraped data'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the scrapers."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    total_files = 0
    
    # Run AMU scraper if requested
    if args.site in ['amu', 'all']:
        logger.info("Starting AMU website scraper...")
        amu_output_dir = os.path.join(args.output_dir, 'amu')
        amu_scraper = AMUScraper(
            output_dir=amu_output_dir,
            delay=args.delay,
            max_pages=args.max_pages
        )
        amu_files = amu_scraper.scrape()
        total_files += len(amu_files)
        logger.info(f"AMU scraper finished. Collected {len(amu_files)} files.")
    
    # Run AMU Exams scraper if requested
    if args.site in ['exams', 'all']:
        logger.info("Starting AMU Controller of Exams website scraper...")
        exams_output_dir = os.path.join(args.output_dir, 'amu_exams')
        exams_scraper = AMUExamsScraper(
            output_dir=exams_output_dir,
            delay=args.delay,
            max_pages=args.max_pages
        )
        exams_files = exams_scraper.scrape()
        total_files += len(exams_files)
        logger.info(f"AMU Exams scraper finished. Collected {len(exams_files)} files.")
    
    # Log summary
    elapsed_time = time.time() - start_time
    logger.info(f"Scraping completed in {elapsed_time:.2f} seconds.")
    logger.info(f"Total files collected: {total_files}")
    
    # Return success
    return 0

if __name__ == "__main__":
    sys.exit(main())