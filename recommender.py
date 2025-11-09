"""
Recommendation engine using RAG (Retrieval-Augmented Generation) with Gemini API re-ranking
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Load environment variables from .env file
try:
    from load_env import load_env
    load_env()
except Exception as e:
    print(f"Note: Could not load .env file: {e}")

import google.generativeai as genai
from typing import List, Dict, Optional
from embedding_service import EmbeddingService
from data_processor import process_query
import re
import time


class SHLRecommender:
    """SHL Assessment Recommendation System"""
    
    def __init__(self, gemini_api_key: Optional[str] = None, 
                 embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize the recommender system
        
        Args:
            gemini_api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
            embedding_service: Pre-initialized embedding service (optional)
        """
        # Initialize Gemini API
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_api_key_here':
            try:
                genai.configure(api_key=api_key)
                # Try different model names (API requires "models/" prefix)
                model_names = [
                    'models/gemini-2.0-flash-exp',  # Latest experimental
                    'models/gemini-1.5-flash',  # Fast and reliable
                    'models/gemini-1.5-pro',  # More capable
                    'models/gemini-1.5-flash-latest',  # Stable fallback
                ]
                self.model = None
                for model_name in model_names:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        print(f"✓ Using Gemini model: {model_name}")
                        break
                    except Exception as e:
                        continue
                if self.model is None:
                    print("⚠ Warning: Could not initialize any Gemini model. LLM re-ranking will be disabled.")
                    self.model = None
            except Exception as e:
                print(f"⚠ Warning: Failed to configure Gemini API: {e}")
                print("  LLM re-ranking will be disabled.")
                self.model = None
        else:
            print("\n" + "="*80)
            print("⚠ GEMINI API KEY NOT CONFIGURED")
            print("="*80)
            print("LLM re-ranking is DISABLED. To enable:")
            print()
            print("1. Get a FREE API key from: https://aistudio.google.com/app/apikey")
            print()
            print("2. Set it in one of these ways:")
            print("   Option A - Create .env file:")
            print("      Copy env_template.txt to .env")
            print("      Edit .env and add: GEMINI_API_KEY=your_actual_key")
            print()
            print("   Option B - Set environment variable:")
            print("      PowerShell: $env:GEMINI_API_KEY='your_actual_key'")
            print("      CMD:        set GEMINI_API_KEY=your_actual_key")
            print()
            print("3. Run recommender.py again")
            print("="*80 + "\n")
            self.model = None
        
        # Initialize embedding service
        if embedding_service is None:
            self.embedding_service = EmbeddingService()
            self.embedding_service.build_index()
        else:
            self.embedding_service = embedding_service
    
    def _extract_test_types(self, test_type_str: str) -> List[str]:
        """Extract test types from string (e.g., 'P,K' -> ['P', 'K'])"""
        if not test_type_str:
            return []
        return [t.strip() for t in test_type_str.split(',') if t.strip()]
    
    def _balance_recommendations(self, candidates: List[Dict], query: str, 
                                target_count: int = 10) -> List[Dict]:
        """
        Balance recommendations across test types (K and P) when query spans multiple domains.
        Uses LLM to identify which test types are relevant, then balances accordingly.
        """
        if not candidates:
            return []
        
        # Analyze query to determine relevant test types
        query_lower = query.lower()
        has_technical = any(keyword in query_lower for keyword in 
                          ['java', 'python', 'sql', 'javascript', 'coding', 'programming', 
                           'technical', 'developer', 'engineer', 'skills', 'knowledge', 
                           'technology', 'code', 'software'])
        has_behavioral = any(keyword in query_lower for keyword in 
                           ['collaborate', 'team', 'personality', 'behavior', 'behavioral',
                            'communication', 'stakeholder', 'interpersonal', 'soft skills',
                            'work with', 'interact'])
        
        # Determine target test types
        target_types = []
        if has_technical:
            target_types.append('K')
        if has_behavioral:
            target_types.append('P')
        
        # If no clear indicators, use all types
        if not target_types:
            target_types = ['K', 'P']
        
        # Group candidates by test type
        type_groups = {'K': [], 'P': [], 'both': []}
        for candidate in candidates:
            test_types = self._extract_test_types(candidate.get('test_type', ''))
            if 'K' in test_types and 'P' in test_types:
                type_groups['both'].append(candidate)
            elif 'K' in test_types:
                type_groups['K'].append(candidate)
            elif 'P' in test_types:
                type_groups['P'].append(candidate)
            else:
                # If no test type specified, add to both
                type_groups['both'].append(candidate)
        
        # Balance selection
        balanced = []
        remaining = target_count
        
        # If query spans both domains, balance between them
        if len(target_types) > 1:
            per_type = max(1, remaining // len(target_types))
            for test_type in target_types:
                if test_type in type_groups and type_groups[test_type]:
                    balanced.extend(type_groups[test_type][:per_type])
                    remaining -= min(per_type, len(type_groups[test_type]))
            
            # Fill remaining slots with 'both' type or highest similarity
            if remaining > 0:
                remaining_candidates = [c for c in candidates if c not in balanced]
                balanced.extend(remaining_candidates[:remaining])
        else:
            # Single domain - prioritize that type, but include 'both' type items
            primary_type = target_types[0] if target_types else 'K'
            if primary_type in type_groups:
                balanced.extend(type_groups[primary_type][:remaining])
                remaining -= len(type_groups[primary_type])
            
            # Add 'both' type items
            if remaining > 0 and type_groups['both']:
                balanced.extend(type_groups['both'][:remaining])
                remaining -= len(type_groups['both'])
            
            # Fill with other type if still needed
            if remaining > 0:
                other_type = 'P' if primary_type == 'K' else 'K'
                if other_type in type_groups:
                    balanced.extend(type_groups[other_type][:remaining])
        
        # Ensure we have at least some results
        if not balanced:
            balanced = candidates[:target_count]
        
        return balanced[:target_count]
    
    def _llm_rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Use Gemini API to re-rank and select the most relevant assessments
        """
        print(f"\n{'='*80}")
        print(f"STEP 3: LLM RE-RANKING")
        print(f"{'='*80}")
        
        if not self.model:
            print("[!] LLM NOT AVAILABLE - Skipping re-ranking (using similarity-based ranking)")
            print(f"[i] Returning top {top_k} candidates by similarity score")
            return candidates[:top_k]
        
        if not candidates:
            print("[!] No candidates to re-rank")
            return candidates[:top_k]
        
        print(f"[+] LLM IS BEING INVOKED")
        print(f"[i] Model: {self.model._model_name if hasattr(self.model, '_model_name') else 'Gemini'}")
        print(f"[i] Input: {len(candidates[:20])} candidates (limited to top 20)")
        print(f"[i] Requested: Top {top_k} most relevant assessments")
        
        # Prepare candidate descriptions
        candidate_texts = []
        for i, candidate in enumerate(candidates[:20]):  # Limit to top 20 for LLM
            text = f"{i+1}. {candidate['title']}\n   Description: {candidate['description'][:200]}\n   Test Type: {candidate.get('test_type', '')}"
            candidate_texts.append(text)
        
        candidates_str = "\n".join(candidate_texts)
        
        prompt = f"""You are an expert at matching job requirements with appropriate assessment tests.

Given the following query and a list of assessment tests, select the {top_k} most relevant assessments.

Query: {query}

Available Assessments:
{candidates_str}

Please return ONLY the numbers (1-{len(candidate_texts)}) of the {top_k} most relevant assessments, separated by commas, in order of relevance.
For example: 3, 7, 1, 5, 9

Your selection:"""
        
        # Try with timeout and retries
        max_retries = 2
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to LLM (attempt {attempt + 1}/{max_retries})...")
                start_time = time.time()
                
                # Set a shorter timeout to fail fast
                import google.generativeai as genai
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=100,  # Short response needed
                        temperature=0.3
                    )
                )
                result_text = response.text.strip()
                elapsed_time = time.time() - start_time
                
                print(f"[+] LLM RESPONSE RECEIVED (took {elapsed_time:.2f}s)")
                print(f"[i] Raw response: {result_text}")
                
                # Extract numbers from response
                numbers = re.findall(r'\d+', result_text)
                selected_indices = [int(n) - 1 for n in numbers[:top_k] if int(n) <= len(candidates) and int(n) > 0]
                
                print(f"[i] LLM selected indices: {[i+1 for i in selected_indices]}")
                print(f"[i] Number of selections: {len(selected_indices)}")
                
                # If we got valid selections, use them
                if selected_indices:
                    reranked = [candidates[i] for i in selected_indices if 0 <= i < len(candidates)]
                    print(f"[+] Successfully re-ranked {len(reranked)} assessments")
                    
                    # Fill remaining slots with original order if needed
                    remaining = top_k - len(reranked)
                    if remaining > 0:
                        print(f"[i] Filling {remaining} remaining slots with similarity-based candidates")
                        used_indices = set(selected_indices)
                        for candidate in candidates:
                            if len(reranked) >= top_k:
                                break
                            idx = candidates.index(candidate)
                            if idx not in used_indices:
                                reranked.append(candidate)
                    
                    print(f"\n[+] LLM RE-RANKING COMPLETE")
                    print(f"[i] Final count after LLM re-ranking: {len(reranked[:top_k])} assessments")
                    return reranked[:top_k]
                else:
                    print(f"[!] LLM returned no valid selections, retrying...")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"[!] LLM Error (attempt {attempt + 1}): {error_msg[:200]}")
                if attempt < max_retries - 1:
                    # Wait before retry
                    print(f"[*] Waiting 1s before retry...")
                    time.sleep(1)
                    continue
                # On final attempt or if error persists, fall back
                if "Timeout" not in error_msg and "503" not in error_msg and "failed to connect" not in error_msg:
                    print(f"[!] LLM re-ranking failed: {error_msg[:100]}")
        
        # Fallback to similarity-based ranking
        print(f"\n[!] LLM RE-RANKING FAILED - FALLING BACK TO SIMILARITY-BASED RANKING")
        print(f"[i] Returning top {top_k} by similarity score")
        return candidates[:top_k]
    
    def recommend(self, query: str, url: Optional[str] = None, 
                  min_results: int = 5, max_results: int = 10) -> List[Dict]:
        """
        Generate recommendations for a given query or job description URL
        
        Args:
            query: Natural language query or job description text
            url: Optional URL to extract job description from
            min_results: Minimum number of recommendations (default: 5)
            max_results: Maximum number of recommendations (default: 10)
        
        Returns:
            List of recommended assessments with title, url, and other metadata
        """
        print(f"\n{'='*80}")
        print(f"RECOMMENDATION PIPELINE STARTED")
        print(f"{'='*80}")
        
        # Process query (extract text from URL if provided)
        print(f"\n{'='*80}")
        print(f"STEP 1: QUERY PROCESSING")
        print(f"{'='*80}")
        print(f"[i] Original query: {query[:200]}{'...' if len(query) > 200 else ''}")
        if url:
            print(f"[i] URL provided: {url}")
        processed_query = process_query(query, url)
        print(f"[i] Processed query length: {len(processed_query)} characters")
        
        # Retrieve top candidates using semantic search
        print(f"\n{'='*80}")
        print(f"STEP 2: EMBEDDING-BASED RETRIEVAL")
        print(f"{'='*80}")
        print(f"[*] Searching embedding space for similar assessments...")
        print(f"[i] Total assessments in index: {len(self.embedding_service.catalog_items)}")
        print(f"[i] Retrieving top 20 candidates by cosine similarity")
        
        candidates = self.embedding_service.search(processed_query, top_k=20)
        
        print(f"[+] EMBEDDING MATCHING COMPLETE")
        print(f"[i] Candidates extracted: {len(candidates)} assessments")
        
        if candidates:
            print(f"\n[i] Top 5 candidates by similarity:")
            for i, c in enumerate(candidates[:5], 1):
                print(f"    {i}. {c['title'][:60]} (similarity: {c.get('similarity', 0):.4f})")
        
        if not candidates:
            print(f"[!] No candidates found!")
            return []
        
        # Re-rank using LLM
        reranked = self._llm_rerank(processed_query, candidates, top_k=max_results)
        
        # Balance across test types
        print(f"\n{'='*80}")
        print(f"STEP 4: BALANCING ACROSS TEST TYPES")
        print(f"{'='*80}")
        print(f"[*] Balancing recommendations across K (Knowledge) and P (Personality) tests...")
        balanced = self._balance_recommendations(reranked, processed_query, target_count=max_results)
        print(f"[+] Balanced results: {len(balanced)} assessments")
        
        # Count test types
        k_count = sum(1 for item in balanced if 'K' in item.get('test_type', ''))
        p_count = sum(1 for item in balanced if 'P' in item.get('test_type', ''))
        print(f"[i] Test type distribution: {k_count} Knowledge tests, {p_count} Personality tests")
        
        # Ensure minimum results
        if len(balanced) < min_results:
            print(f"\n[*] Adding more results to meet minimum of {min_results}...")
            # Add more from original candidates if needed
            used_urls = {item['url'] for item in balanced}
            for candidate in candidates:
                if len(balanced) >= min_results:
                    break
                if candidate['url'] not in used_urls:
                    balanced.append(candidate)
            print(f"[+] Final count: {len(balanced)} assessments")
        
        # Format results
        print(f"\n{'='*80}")
        print(f"STEP 5: FORMATTING FINAL RESULTS")
        print(f"{'='*80}")
        results = []
        for item in balanced[:max_results]:
            results.append({
                'assessment_name': item['title'],
                'assessment_url': item['url'],
                'description': item.get('description', ''),
                'test_type': item.get('test_type', ''),
                'adaptive_support': item.get('adaptive_support', 'No'),
                'duration': item.get('duration'),
                'remote_support': item.get('remote_support', 'Yes'),
                'similarity': item.get('similarity', 0.0)
            })
        
        print(f"[+] RECOMMENDATION PIPELINE COMPLETE")
        print(f"[i] Total recommendations: {len(results)}")
        print(f"{'='*80}\n")
        
        return results


if __name__ == "__main__":
    # Test the recommender
    import os
    
    print("Initializing recommender...")
    recommender = SHLRecommender()
    
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script."
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        recommendations = recommender.recommend(query)
        print(f"\nFound {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['assessment_name']}")
            print(f"   URL: {rec['assessment_url']}")
            print(f"   Test Type: {rec['test_type']}")

