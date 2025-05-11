"""
Web scraper for the AMU main website.
"""
import os
import logging
from typing import List, Set
from queue import Queue
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

class AMUScraper(BaseScraper):
    """Scraper for the main AMU website."""
    
    def __init__(self, output_dir: str, max_pages: int = 100, delay: float = 1.0):
        """
        Initialize the AMU scraper.
        
        Args:
            output_dir: Directory to save scraped content
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests in seconds
        """
        super().__init__(
            base_url="https://www.amu.ac.in/",
            output_dir=output_dir,
            delay=delay,
            max_pages=max_pages
        )
        
        # Define priority patterns for important pages
        self.priority_patterns = [
            'admission',
            'course',
            'program',
            'department',
            'faculty',
            'research',
            'about',
            'contact',
            'hostel',
            'library',
            'campus',
        ]
        
        # Additional patterns to skip
        self.skip_patterns = [
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.zip', '.rar', '.jpg', '.jpeg', '.png', '.gif',
            'login', 'register', 'wp-admin', 'wp-content'
        ]
    
    def is_valid_url(self, url: str) -> bool:
        """
        Override base method to implement AMU-specific rules.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if URL should be scraped
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
        
        # Only allow AMU domains
        valid_domains = ['amu.ac.in', 'aligarh.ac.in']
        if parsed.netloc and not any(parsed.netloc.endswith(domain) for domain in valid_domains):
            return False
            
        return True
        
    def prioritize_links(self, links: List[str]) -> List[str]:
        """
        Prioritize links based on predefined patterns.
        
        Args:
            links: List of URLs to prioritize
            
        Returns:
            Prioritized list of URLs
        """
        high_priority = []
        normal_priority = []
        
        for link in links:
            is_priority = any(pattern in link.lower() for pattern in self.priority_patterns)
            if is_priority:
                high_priority.append(link)
            else:
                normal_priority.append(link)
                
        return high_priority + normal_priority
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract relevant content from the HTML.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            
        Returns:
            Extracted text content
        """
        try:
            # Remove script, style, and hidden elements
            for element in soup(['script', 'style', 'meta', 'link', '[style*="display:none"]', '[style*="display: none"]']):
                element.extract()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.text.strip()
            
            # Identify main content area
            main_content = None
            
            # Try to find main content containers by common IDs and classes
            for container_id in ['main-content', 'content', 'main', 'page-content', 'article']:
                main_content = soup.find(id=container_id)
                if main_content:
                    break
                    
            for container_class in ['main-content', 'content', 'main', 'page-content', 'article', 'entry-content']:
                if not main_content:
                    main_content = soup.find(class_=container_class)
                if main_content:
                    break
            
            # If no main content area found, use the body
            if not main_content:
                main_content = soup.body
            
            # If still no content, use the entire soup
            if not main_content:
                main_content = soup
            
            # Process content
            paragraphs = []
            
            # Extract text from paragraphs
            for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = p.get_text().strip()
                if text:
                    paragraphs.append(text)
            
            # Extract data from tables
            for table in main_content.find_all('table'):
                table_text = []
                for row in table.find_all('tr'):
                    row_texts = [cell.get_text().strip() for cell in row.find_all(['th', 'td'])]
                    row_text = ' | '.join(text for text in row_texts if text)
                    if row_text:
                        table_text.append(row_text)
                if table_text:
                    paragraphs.append('\n'.join(table_text))
            
            # Extract text from list items
            for ul in main_content.find_all(['ul', 'ol']):
                for li in ul.find_all('li', recursive=False):  # Only direct children
                    text = li.get_text().strip()
                    if text:
                        paragraphs.append('• ' + text)
            
            # Combine and format the content
            extracted_text = f"TITLE: {title}\nURL: {url}\n\nCONTENT:\n"
            extracted_text += '\n\n'.join(paragraphs)
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return f"Error extracting content: {e}"
        
    def scrape(self) -> List[str]:
        """
        Scrape the AMU website.
        
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
            content = self.extract_content(soup, url)
            
            if content:
                saved_file = self.save_content(url, content, prefix="amu")
                saved_files.append(saved_file)
                
            # Extract links and add to queue
            links = self.extract_links(soup, url)
            prioritized_links = self.prioritize_links(links)
            
            for link in prioritized_links:
                if link not in self.visited_urls:
                    queue.put(link)
                    
            # Update counter
            pages_scraped += 1
            
            # Log progress
            if pages_scraped % 10 == 0:
                logger.info(f"Pages scraped: {pages_scraped}, Queue size: {queue.qsize()}")
                
        logger.info(f"Scraping complete. Pages scraped: {pages_scraped}, Files saved: {len(saved_files)}")
        
        return saved_files