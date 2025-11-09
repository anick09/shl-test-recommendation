"""
Data processing module for SHL Assessment Recommender
Handles loading, filtering, and validation of catalog and dataset files
"""

import pandas as pd
import re
import os
from typing import List, Dict, Tuple
import requests
from urllib.parse import urlparse


def load_catalog(catalog_path: str = "shl_catalog.csv") -> pd.DataFrame:
    """Load the SHL catalog CSV file"""
    df = pd.read_csv(catalog_path)
    return df


def filter_individual_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out 'Pre-packaged Job Solutions' and keep only individual test solutions.
    
    IMPORTANT: Based on train set analysis, individual tests can be under:
    - /products/product-catalog/view/ (individual tests)
    - /solutions/products/product-catalog/view/ (also individual tests!)
    
    Excludes:
    - General category/landing pages (empty titles or very generic descriptions)
    - Pages that are clearly category pages (no specific assessment name)
    """
    # Start with all rows that have a title (exclude category pages)
    filtered = df[df['title'].notna() & (df['title'] != '')].copy()
    
    # Keep individual tests - they can be under either:
    # - /products/product-catalog/view/ 
    # - /solutions/products/product-catalog/view/
    # Both patterns contain individual test solutions (not pre-packaged)
    filtered = filtered[
        filtered['url'].str.contains('/product-catalog/view/', na=False)
    ].copy()
    
    # Exclude very generic category pages (those with empty or very short descriptions)
    # Category pages typically have no specific assessment description
    filtered = filtered[
        (filtered['description'].notna()) & 
        (filtered['description'].str.len() > 50)  # Real assessments have substantial descriptions
    ].copy()
    
    # Exclude the main product catalog landing page
    filtered = filtered[
        filtered['url'] != 'https://www.shl.com/products/product-catalog/'
    ].copy()
    
    return filtered.reset_index(drop=True)


def validate_catalog_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that all required fields are present and non-empty"""
    required_fields = ['url', 'title', 'description', 'test_type']
    
    # Check if all required fields exist
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Remove rows with missing critical fields
    validated = df[
        df['url'].notna() & 
        (df['url'] != '') &
        df['title'].notna() & 
        (df['title'] != '')
    ].copy()
    
    # Fill empty descriptions with empty string
    validated['description'] = validated['description'].fillna('')
    
    # Ensure test_type is present (fill with empty string if missing)
    validated['test_type'] = validated['test_type'].fillna('')
    
    return validated


def load_train_test_data(csv_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from CSV files"""
    if csv_path is None:
        # Try CSV files first, fallback to Excel
        if os.path.exists('train.csv') and os.path.exists('test.csv'):
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
        else:
            # Fallback to Excel
            train_df = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
            test_df = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name='Test-Set')
    else:
        # If specific path provided, assume it's Excel
        train_df = pd.read_excel(csv_path, sheet_name='Train-Set')
        test_df = pd.read_excel(csv_path, sheet_name='Test-Set')
    
    return train_df, test_df


def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL (for job description URLs)"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Simple text extraction - in production, use BeautifulSoup for better parsing
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit to 5000 characters
    except Exception as e:
        print(f"Error extracting text from URL {url}: {e}")
        return ""


def process_query(query: str, url: str = None) -> str:
    """
    Process a query - if URL is provided, extract text from it and combine with query.
    Otherwise, return the query as-is.
    """
    if url:
        url_text = extract_text_from_url(url)
        if url_text:
            return f"{query}\n\n{url_text}" if query else url_text
    return query


def prepare_catalog_for_embedding(df: pd.DataFrame) -> List[Dict]:
    """
    Prepare catalog data for embedding generation.
    Creates a combined text field from title and description.
    Preserves all catalog fields for API responses.
    """
    catalog_items = []
    
    for _, row in df.iterrows():
        # Combine title and description for embedding
        text = f"{row['title']}. {row['description']}".strip()
        
        # Create item with all available fields
        item = {
            'url': row['url'],
            'title': row['title'],
            'description': row['description'],
            'test_type': row.get('test_type', ''),
            'text': text  # Combined text for embedding
        }
        
        # Add optional fields if they exist in the dataframe
        if 'adaptive_support' in row.index:
            item['adaptive_support'] = row['adaptive_support']
        if 'duration' in row.index:
            item['duration'] = row['duration'] if pd.notna(row['duration']) else None
        if 'remote_support' in row.index:
            item['remote_support'] = row['remote_support']
        
        catalog_items.append(item)
    
    return catalog_items


if __name__ == "__main__":
    # Test the data processing pipeline
    print("Loading catalog...")
    catalog_df = load_catalog()
    print(f"Original catalog size: {len(catalog_df)} rows")
    
    print("\nFiltering individual tests...")
    filtered_df = filter_individual_tests(catalog_df)
    print(f"Filtered catalog size: {len(filtered_df)} rows")
    
    print("\nValidating fields...")
    validated_df = validate_catalog_fields(filtered_df)
    print(f"Validated catalog size: {len(validated_df)} rows")
    
    print("\nSample filtered data:")
    print(validated_df[['url', 'title', 'test_type']].head(10))
    
    print("\nLoading train/test data...")
    train_df, test_df = load_train_test_data()
    print(f"Train set: {len(train_df)} rows, {train_df['Query'].nunique()} unique queries")
    print(f"Test set: {len(test_df)} rows")
    
    print("\nData processing complete!")

