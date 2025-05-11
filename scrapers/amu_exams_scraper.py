"""
AMU Controller of Exams website scraper for amucontrollerexams.com and its subdomains.
"""
import os
import logging
from typing import List, Set
from queue import Queue
from urllib.parse import urlparse

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

class AMUExamsScraper(BaseScraper):
    """Scraper for AMU Controller of Exams website and its subdomains."""
    
    def __init__(self, output_dir: str, delay: float = 1.5, max_pages: int = 1000):
        """Initialize the AMU Exams scraper."""
        super().__init__(
            base_url="https://www.amucontrollerexams.com/",
            output_dir=output_dir,
            delay=delay,
            max_pages=max_pages
        )
        # Additional patterns to skip
        self.skip_patterns = [
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.zip', '.rar', '.jpg', '.jpeg', '.png', '.gif',
            'login', 'register', 'download', 'wp-admin', 'wp-content'
        ]
        
    def is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid for the AMU Exams scraper.
        Overrides base method to implement AMU Exams-specific rules.
        """
        # First apply the base rules
        if not super().is_valid_url(url):
            return False
            
        # Skip files and certain patterns
        for pattern in self.skip_patterns:
            if pattern in url.lower():
                return False
                
        # Parse URL
        parsed = urlparse(url)
        
        # Only allow AMU Exams domains
        if parsed.netloc and not parsed.netloc.endswith('amucontrollerexams.com'):
            return False
            
        return True
        
    def scrape(self) -> List[str]:
        """
        Scrape the AMU Exams website and its subdomains.
        
        Returns:
            List of paths to saved content files
        """
        logger.info(f"Starting scrape of {self.base_url}")
        
        saved_files = []
        queue = Queue()
        queue.put(self.base_url)
        
        # Keep track of visited URLs to avoid duplicates
        self.visited_urls: Set[str] = set()
        
        # Process the queue
        pages_scraped = 0
        
        while not queue.empty() and pages_scraped < self.max_pages:
            # Get the next URL from the queue
            url = queue.get()
            
            # Skip if we've already visited this URL
            if url in self.visited_urls:
                continue
                
            # Add to visited URLs
            self.visited_urls.add(url)
            
            # Fetch and parse the page
            logger.info(f"Scraping: {url}")
            soup = self.get_page(url)
            
            if not soup:
                continue
                
            # Extract and save content
            content = self.extract_text_content(soup)
            
            if content:
                saved_file = self.save_content(url, content, prefix="amu_exams")
                saved_files.append(saved_file)
                
            # Extract links and add to queue
            links = self.extract_links(soup, url)
            
            for link in links:
                if link not in self.visited_urls:
                    queue.put(link)
                    
            # Update counter
            pages_scraped += 1
            
            # Log progress
            if pages_scraped % 10 == 0:
                logger.info(f"Pages scraped: {pages_scraped}, Queue size: {queue.qsize()}")
                
        logger.info(f"Scraping complete. Pages scraped: {pages_scraped}, Files saved: {len(saved_files)}")
        
        return saved_files