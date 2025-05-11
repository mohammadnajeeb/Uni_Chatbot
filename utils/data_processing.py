"""
Data processing utilities for the Faiz Chatbot.
"""
import os
import re
import glob
import logging
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def load_text_files(glob_pattern: str) -> List[Tuple[str, str]]:
    """
    Load text files matching the glob pattern.
    
    Args:
        glob_pattern: Glob pattern to match files
        
    Returns:
        List of tuples containing (file_path, content)
    """
    logger.info(f"Loading files matching {glob_pattern}")
    
    files = []
    for file_path in glob.glob(glob_pattern, recursive=True):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
                # Skip empty files
                if not content.strip():
                    continue
                    
                # Skip files with encoding issues (more than 20% non-printable characters)
                non_printable = sum(1 for char in content if ord(char) < 32 or ord(char) > 126)
                if content and (non_printable / len(content)) > 0.2:
                    logger.warning(f"Skipping file with encoding issues: {file_path}")
                    continue
                
                files.append((file_path, content))
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    logger.info(f"Loaded {len(files)} files from {glob_pattern}")
    return files

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Replace non-printable characters
    text = ''.join(char if (ord(char) >= 32 and ord(char) < 127) or char.isspace() else ' ' for char in text)
    
    # Replace multiple spaces, newlines, and tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize quotes and apostrophes
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    
    return text.strip()

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata from text content.
    
    Args:
        text: Text content
        
    Returns:
        Dictionary of metadata
    """
    metadata = {}
    
    # Extract URL
    url_match = re.search(r'URL:\s*(https?://[^\s]+)', text)
    if url_match:
        metadata['url'] = url_match.group(1)
    
    # Extract title
    title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', text)
    if title_match:
        metadata['title'] = title_match.group(1)
    
    # Determine category based on URL or content
    if 'url' in metadata:
        url = metadata['url'].lower()
        if 'admission' in url:
            metadata['category'] = 'admission'
        elif 'exam' in url:
            metadata['category'] = 'examination'
        elif 'result' in url or 'answer-key' in url:
            metadata['category'] = 'results'
        elif 'phd' in url or 'research' in url:
            metadata['category'] = 'research'
        elif 'course' in url or 'program' in url:
            metadata['category'] = 'courses'
        else:
            metadata['category'] = 'general'
    
    return metadata

def create_documents(files: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Create documents from loaded files.
    
    Args:
        files: List of tuples containing (file_path, content)
        
    Returns:
        List of documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    documents = []
    
    for file_path, content in files:
        try:
            # Clean content
            cleaned_content = clean_text(content)
            
            # Extract metadata
            metadata = extract_metadata(content)
            metadata['source'] = file_path
            
            # Split content into chunks
            chunks = text_splitter.split_text(cleaned_content)
            
            # Create documents from chunks
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                doc = {
                    'text': chunk,
                    'metadata': {
                        **metadata,
                        'chunk': i
                    }
                }
                documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    return documents