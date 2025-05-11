"""
Base scraper module containing common functionality for all scrapers.
"""
import os
import time
import logging
import requests
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class BaseScraper:
    """Base class for web scrapers with common functionality."""
    
    def __init__(
        self, 
        base_url: str,
        output_dir: str,
        delay: float = 1.5,
        max_pages: int = 1000,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the base scraper.
        
        Args:
            base_url: The root URL to start scraping from
            output_dir: Directory to save scraped data
            delay: Time to wait between requests in seconds
            max_pages: Maximum number of pages to scrape
            headers: Custom headers for HTTP requests
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay = delay
        self.max_pages = max_pages
        self.visited_urls = set()
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
        }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a web page and parse it with BeautifulSoup.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if the request failed
        """
        try:
            # Rate limiting
            time.sleep(self.delay)
            
            # Make the request
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            return BeautifulSoup(response.text, 'html.parser')
        
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid and should be scraped.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if the URL should be scraped
        """
        if not url or url in self.visited_urls:
            return False
            
        # Parse the URL
        parsed = urlparse(url)
        
        # Check if it's a relative URL
        if not parsed.netloc:
            return True
            
        # Check if it's part of the target domain
        base_domain = urlparse(self.base_url).netloc
        return parsed.netloc == base_domain or parsed.netloc.endswith(f".{base_domain}")
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL (handle relative URLs, remove fragments, etc.)
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        # Handle relative URLs
        if not urlparse(url).netloc:
            url = urljoin(self.base_url, url)
            
        # Remove fragments
        url = url.split('#')[0]
        
        # Ensure trailing slash for consistency
        if not url.endswith('/') and '.' not in url.split('/')[-1]:
            url += '/'
            
        return url
    
    def extract_links(self, soup: BeautifulSoup, url: str) -> List[str]:
        """
        Extract all links from a page.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the current page
            
        Returns:
            List of normalized URLs
        """
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            
            # Skip empty links, javascript, and mailto links
            if (
                not href or 
                href.startswith(('javascript:', 'mailto:', 'tel:')) or
                href == '#'
            ):
                continue
                
            # Normalize the URL
            normalized_url = self.normalize_url(href)
            
            # Check if it's a valid URL to crawl
            if self.is_valid_url(normalized_url):
                links.append(normalized_url)
                
        return links
    
    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """
        Extract meaningful text content from a page.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Extracted text content
        """
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
            
        # Get text and clean it
        text = soup.get_text()
        
        # Normalize whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def save_content(self, url: str, content: str, prefix: str = "page") -> str:
        """
        Save the extracted content to a file.
        
        Args:
            url: Source URL
            content: Text content to save
            prefix: Filename prefix
            
        Returns:
            Path to the saved file
        """
        # Create a filename from the URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        if path_parts and path_parts[-1]:
            filename = path_parts[-1]
            if '.' not in filename:
                filename += '.txt'
        else:
            # Use domain as filename if path is empty
            filename = f"{parsed.netloc.replace('.', '_')}.txt"
        
        # Ensure the filename is valid
        filename = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in filename)
        filepath = os.path.join(self.output_dir, f"{prefix}_{filename}")
        
        # Write the content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n\n")
            f.write(content)
            
        logger.info(f"Saved content from {url} to {filepath}")
        return filepath
            
    def scrape(self) -> List[str]:
        """
        Main scraping method to be implemented by subclasses.
        
        Returns:
            List of paths to saved content files
        """
        raise NotImplementedError("Subclasses must implement this method")