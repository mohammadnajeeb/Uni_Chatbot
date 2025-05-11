"""
Vector database utilities for storing and retrieving document embeddings.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(self, db_path: str):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to store the vector database
        """
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collection for AMU content
        self.collection = self.client.get_or_create_collection(
            name="amu_content",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized vector store at {db_path}")
        
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        embed_fn
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'id', 'text', and 'metadata'
            embed_fn: Function to convert text to embedding vectors
        """
        # Prepare batches for efficient insertion
        batch_size = 100
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract data from the batch
            ids = [doc['id'] for doc in batch]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            
            # Generate embeddings
            embeddings = embed_fn(texts)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            logger.info(f"Added batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} to vector store")
            
        logger.info(f"Added {total_docs} documents to vector store")
        
    def search(
        self, 
        query: str, 
        embed_fn, 
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store for relevant documents.
        
        Args:
            query: Search query
            embed_fn: Function to convert query to embedding vector
            n_results: Number of results to return
            
        Returns:
            List of documents with their metadata and similarity scores
        """
        # Generate query embedding
        query_embedding = embed_fn([query])[0]
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format the results
        documents = []
        
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            document = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            documents.append({
                'id': doc_id,
                'text': document,
                'metadata': metadata,
                'score': 1.0 - distance  # Convert distance to similarity score
            })
            
        return documents
    
    def count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Document count
        """
        return self.collection.count()