"""
Embedding service for generating and storing vector embeddings of assessments
"""

import os
# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle
from data_processor import prepare_catalog_for_embedding, load_catalog, filter_individual_tests, validate_catalog_fields


class EmbeddingService:
    """Service for generating and managing embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service with a sentence transformer model.
        all-MiniLM-L6-v2 is a lightweight, fast model good for semantic search.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.catalog_items = None
        
    def generate_embeddings(self, catalog_items: List[Dict]) -> np.ndarray:
        """Generate embeddings for catalog items"""
        texts = [item['text'] for item in catalog_items]
        print(f"Generating embeddings for {len(texts)} items...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, catalog_path: str = "shl_catalog.csv", 
                   train_path: str = "train.csv",
                   cache_path: str = "embeddings_cache.pkl"):
        """
        Build embedding index from catalog + train set assessments.
        Uses assessments from both catalog and train set to ensure coverage.
        Loads from cache if available, otherwise generates new embeddings.
        """
        # Check if cache exists
        if os.path.exists(cache_path):
            print(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.embeddings = cache_data['embeddings']
                self.catalog_items = cache_data['catalog_items']
                print(f"Loaded {len(self.catalog_items)} items from cache")
                return
        
        # Generate new embeddings
        print("Generating new embeddings...")
        
        # Load catalog - get ALL assessments
        catalog_df = load_catalog(catalog_path)
        filtered_df = filter_individual_tests(catalog_df)
        validated_df = validate_catalog_fields(filtered_df)
        
        print(f"Catalog assessments (after filtering): {len(validated_df)}")
        
        # Load train set - get ALL unique assessments
        train_assessments_list = []
        if os.path.exists(train_path):
            try:
                train_df = pd.read_csv(train_path)
                train_urls = train_df['Assessment_url'].unique()
                train_urls_set = set(train_urls)
                catalog_urls_set = set(validated_df['url'].tolist())
                
                print(f"Train set unique URLs: {len(train_urls_set)}")
                print(f"Overlap with catalog: {len(train_urls_set & catalog_urls_set)}")
                
                # Include ALL train URLs, even if they overlap with catalog
                # This ensures we have embeddings for all assessments referenced in train set
                for url in train_urls:
                    if url not in catalog_urls_set:
                        # Create entry for train URL not in catalog
                        name = url.split('/')[-2].replace('-', ' ').title()
                        if name.endswith('/'):
                            name = name[:-1]
                        train_assessments_list.append({
                            'url': url,
                            'title': name,
                            'description': f"Assessment: {name}",
                            'test_type': 'P,K'
                        })
                    # Note: URLs in both are already in validated_df, so we don't duplicate
                
                if train_assessments_list:
                    print(f"Adding {len(train_assessments_list)} assessments from train set not in catalog")
            except Exception as e:
                print(f"Warning: Could not load train set: {e}")
        
        # Combine: catalog assessments + train assessments not in catalog
        # This gives us all unique assessments from both sources
        if train_assessments_list:
            train_df_combined = pd.DataFrame(train_assessments_list)
            combined_df = pd.concat([validated_df, train_df_combined], ignore_index=True)
        else:
            combined_df = validated_df
        
        # Final count: catalog assessments + train-only assessments
        print(f"Total unique assessments for embedding: {len(combined_df)}")
        print(f"  - From catalog: {len(validated_df)}")
        if train_assessments_list:
            print(f"  - From train (not in catalog): {len(train_assessments_list)}")
        
        self.catalog_items = prepare_catalog_for_embedding(combined_df)
        self.embeddings = self.generate_embeddings(self.catalog_items)
        
        # Save to cache
        print(f"Saving embeddings to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'catalog_items': self.catalog_items
            }, f)
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        return self.model.encode([query])[0]
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search for similar assessments using cosine similarity
        Returns top_k most similar items with their similarity scores
        """
        if self.embeddings is None or self.catalog_items is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get query embedding
        query_emb = self.get_query_embedding(query)
        
        # Compute cosine similarities
        # Normalize embeddings for cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results with similarity scores
        results = []
        for idx in top_indices:
            item = self.catalog_items[idx].copy()
            item['similarity'] = float(similarities[idx])
            results.append(item)
        
        return results


if __name__ == "__main__":
    # Test the embedding service
    print("Initializing embedding service...")
    service = EmbeddingService()
    
    print("\nBuilding index...")
    service.build_index()
    
    print("\nTesting search...")
    test_query = "Java developer with collaboration skills"
    results = service.search(test_query, top_k=5)
    
    print(f"\nTop 5 results for query: '{test_query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Test Type: {result['test_type']}")

