"""
Generate predictions on test.csv and output in required submission format
"""
import pandas as pd
import csv
from recommender import SHLRecommender
import time

def generate_predictions(test_file='test.csv', output_file='test_predictions.csv'):
    """
    Generate predictions for test queries and save in submission format
    
    Format:
    Query, Assessment_url
    Query 1, Recommendation 1 (URL)
    Query 1, Recommendation 2 (URL)
    ...
    Query 2, Recommendation 1 (URL)
    """
    print("="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)
    
    # Read test queries
    print(f"\n[1/4] Reading test queries from {test_file}...")
    df = pd.read_csv(test_file)
    queries = df['Query'].tolist()
    print(f"[+] Found {len(queries)} queries to process")
    
    # Initialize recommender
    print(f"\n[2/4] Initializing recommender system...")
    recommender = SHLRecommender()
    print("[+] Recommender initialized")
    
    # Generate recommendations
    print(f"\n[3/4] Generating recommendations...")
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Processing Query {i}/{len(queries)}")
        print(f"{'='*80}")
        print(f"Query preview: {query[:100]}...")
        
        try:
            # Get recommendations (minimum 5, maximum 10)
            recommendations = recommender.recommend(query, min_results=5, max_results=10)
            
            print(f"\n[+] Generated {len(recommendations)} recommendations for query {i}")
            
            # Add to results in required format
            for rec in recommendations:
                results.append({
                    'Query': query,
                    'Assessment_url': rec['assessment_url']
                })
            
            # Small delay between queries to avoid rate limiting
            if i < len(queries):
                time.sleep(1)
                
        except Exception as e:
            print(f"[!] Error processing query {i}: {e}")
            # Add at least one empty result to maintain structure
            results.append({
                'Query': query,
                'Assessment_url': ''
            })
    
    # Save to CSV
    print(f"\n[4/4] Saving predictions to {output_file}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\n{'='*80}")
    print(f"PREDICTION GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"[+] Total rows in output: {len(results)}")
    print(f"[+] Output saved to: {output_file}")
    print(f"[+] Format verification:")
    print(f"    - Columns: {list(results_df.columns)}")
    print(f"    - Shape: {results_df.shape}")
    
    # Display summary
    print(f"\n[i] Summary by query:")
    query_counts = results_df.groupby('Query').size()
    for idx, (query, count) in enumerate(query_counts.items(), 1):
        print(f"    Query {idx}: {count} recommendations")
    
    return results_df


if __name__ == "__main__":
    # Generate predictions
    results = generate_predictions()
    
    # Show sample output
    print(f"\n{'='*80}")
    print("SAMPLE OUTPUT (first 15 rows):")
    print(f"{'='*80}")
    print(results.head(15).to_string(index=False))

