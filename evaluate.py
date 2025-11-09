"""
Evaluation script for Mean Recall@10 metric
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pandas as pd
from recommender import SHLRecommender
from data_processor import load_train_test_data
from typing import List, Set


def calculate_recall_at_k(recommended_urls: List[str], relevant_urls: Set[str], k: int = 10) -> float:
    """
    Calculate Recall@K metric
    
    Args:
        recommended_urls: List of recommended assessment URLs (top K)
        relevant_urls: Set of relevant assessment URLs (ground truth)
        k: Number of top recommendations to consider
    
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_urls:
        return 0.0
    
    # Take top k recommendations
    top_k_recommended = recommended_urls[:k]
    
    # Count how many relevant items are in top k
    relevant_in_top_k = len(set(top_k_recommended) & relevant_urls)
    
    # Recall = relevant items retrieved / total relevant items
    recall = relevant_in_top_k / len(relevant_urls)
    
    return recall


def evaluate_on_train_set(recommender: SHLRecommender, train_path: str = "train.csv") -> dict:
    """
    Evaluate the recommender on the training set
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load train data
    train_df, _ = load_train_test_data(train_path)
    
    # Group by query to get relevant URLs for each query
    query_groups = train_df.groupby('Query')
    
    recalls = []
    query_results = []
    
    print(f"Evaluating on {len(query_groups)} unique queries...")
    print("=" * 80)
    
    for query, group_df in query_groups:
        # Get ground truth relevant URLs
        relevant_urls = set(group_df['Assessment_url'].tolist())
        
        # Get recommendations
        recommendations = recommender.recommend(query, max_results=10)
        recommended_urls = [rec['assessment_url'] for rec in recommendations]
        
        # Calculate Recall@10
        recall = calculate_recall_at_k(recommended_urls, relevant_urls, k=10)
        recalls.append(recall)
        
        # Store results
        query_results.append({
            'query': query,
            'recall@10': recall,
            'num_relevant': len(relevant_urls),
            'num_recommended': len(recommended_urls),
            'relevant_retrieved': len(set(recommended_urls) & relevant_urls)
        })
        
        print(f"\nQuery: {query[:60]}...")
        print(f"  Relevant assessments: {len(relevant_urls)}")
        print(f"  Recommended: {len(recommended_urls)}")
        print(f"  Relevant retrieved: {len(set(recommended_urls) & relevant_urls)}")
        print(f"  Recall@10: {recall:.4f}")
    
    # Calculate Mean Recall@10
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    print("\n" + "=" * 80)
    print(f"Mean Recall@10: {mean_recall:.4f}")
    print(f"Individual Recall@10 scores: {[f'{r:.4f}' for r in recalls]}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(query_results)
    
    return {
        'mean_recall@10': mean_recall,
        'individual_recalls': recalls,
        'query_results': results_df
    }


if __name__ == "__main__":
    print("Initializing recommender...")
    recommender = SHLRecommender()
    
    print("\nRunning evaluation on train set...")
    results = evaluate_on_train_set(recommender)
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Mean Recall@10: {results['mean_recall@10']:.4f}")
    print(f"\nPer-query results:")
    print(results['query_results'].to_string(index=False))
    
    # Save results
    results['query_results'].to_csv('evaluation_results.csv', index=False)
    print(f"\nResults saved to evaluation_results.csv")

