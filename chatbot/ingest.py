"""
Script to ingest scraped data into the vector database.
"""
import os
import sys
import uuid
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
        logging.FileHandler(os.path.join('data', 'ingest.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from utils.data_processing import load_text_files, create_documents
from utils.vector_store import VectorStore

# Import HuggingFace for embeddings
from langchain_huggingface import HuggingFaceEmbeddings

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Ingest scraped data into the vector database.')
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default=os.path.join('data', 'raw'),
        help='Directory containing scraped data'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=os.getenv('VECTOR_DB_PATH', os.path.join('data', 'vectordb')),
        help='Directory for the vector database'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        help='HuggingFace model to use for embeddings'
    )
    
    return parser.parse_args()

def main():
    """Main function to ingest data."""
    args = parse_arguments()
    
    # Initialize HuggingFace embeddings
    try:
        logger.info(f"Initializing embeddings with model: {args.embedding_model}")
        embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        return 1
    
    # Initialize the vector store
    vector_store = VectorStore(args.output_dir)
    
    # Process each directory of scraped data
    for site_dir in ['amu', 'amu_exams']:
        input_dir = os.path.join(args.input_dir, site_dir)
        
        if not os.path.exists(input_dir):
            logger.warning(f"Directory not found: {input_dir}")
            continue
            
        logger.info(f"Processing data from {input_dir}")
        
        # Load text files - gets list of (file_path, content) tuples
        file_path_content_tuples = load_text_files(os.path.join(input_dir, "*.txt"))
        
        if not file_path_content_tuples:
            logger.warning(f"No files found in {input_dir}")
            continue
            
        logger.info(f"Loaded {len(file_path_content_tuples)} files from {input_dir}")
        
        # Process files and create documents for the vector store using the create_documents function
        documents = create_documents(file_path_content_tuples)
        
        logger.info(f"Created {len(documents)} document chunks from {len(file_path_content_tuples)} files")
        
        # Add documents to the vector store
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        try:
            # Generate document IDs
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Use the add_documents method with properly formatted documents
            formatted_docs = [
                {'id': ids[i], 'text': texts[i], 'metadata': metadatas[i]}
                for i in range(len(texts))
            ]
            
            vector_store.add_documents(formatted_docs, embeddings.embed_documents)
            logger.info(f"Added {len(texts)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
    
    # Log summary
    total_docs = vector_store.count()
    logger.info(f"Ingestion complete. Total documents in vector store: {total_docs}")
    
    # Return success
    return 0

if __name__ == "__main__":
    sys.exit(main())