import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
import uuid

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./.chromadb"
        ))
        self.collection = self.chroma_client.get_or_create_collection(name="resume_jd_embeddings")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text string"""
        return self.model.encode(text)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embedding1 = self.generate_embedding(text1).reshape(1, -1)
        embedding2 = self.generate_embedding(text2).reshape(1, -1)
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return max(0, min(1, similarity))  # Ensure between 0 and 1
    
    def store_embedding(self, text: str, metadata: dict, id: str = None):
        """Store text embedding in vector database"""
        if id is None:
            id = str(uuid.uuid4())
        
        embedding = self.generate_embedding(text).tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[id]
        )
        return id
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """Perform semantic search on stored embeddings"""
        query_embedding = self.generate_embedding(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return list(zip(
            results['documents'][0], 
            results['distances'][0]
        ))
    
    def compare_documents(self, doc1: str, doc2: str) -> dict:
        """Compare two documents and return detailed similarity analysis"""
        # Generate embeddings
        emb1 = self.generate_embedding(doc1)
        emb2 = self.generate_embedding(doc2)
        
        # Calculate overall similarity
        overall_similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        
        # Split into sentences and compare
        # For simplicity, we'll just return the overall similarity
        # In a more advanced implementation, we could do chunk-based comparison
        
        return {
            "overall_similarity": overall_similarity,
            "semantic_score": overall_similarity * 100  # Convert to percentage
        }