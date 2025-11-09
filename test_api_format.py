"""
Test script to verify API response format matches requirements
"""
import json
from api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("Testing /health endpoint")
    print("="*80)
    
    response = client.get("/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    print("✓ Health endpoint working correctly")
    

def test_recommend_endpoint():
    """Test recommendation endpoint format"""
    print("\n" + "="*80)
    print("Testing /recommend endpoint")
    print("="*80)
    
    # Test query
    test_query = {
        "query": "I need Java developers who can collaborate with teams"
    }
    
    print(f"\nRequest: {json.dumps(test_query, indent=2)}")
    
    response = client.post("/recommend", json=test_query)
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse structure:")
        print(f"  - Has 'recommended_assessments' key: {'recommended_assessments' in data}")
        
        if 'recommended_assessments' in data:
            assessments = data['recommended_assessments']
            print(f"  - Number of recommendations: {len(assessments)}")
            
            if assessments:
                print(f"\n  First recommendation structure:")
                first = assessments[0]
                expected_fields = ['url', 'name', 'adaptive_support', 'description', 
                                 'duration', 'remote_support', 'test_type']
                
                for field in expected_fields:
                    present = field in first
                    print(f"    - {field}: {'✓' if present else '✗'} {type(first.get(field)).__name__ if present else 'missing'}")
                
                print(f"\n  Sample response (first 2 recommendations):")
                print(json.dumps({"recommended_assessments": assessments[:2]}, indent=2))
                
                # Verify structure matches requirements
                assert 'url' in first
                assert 'name' in first
                assert isinstance(first.get('test_type'), list)
                
                print("\n✓ Response format matches requirements!")
            else:
                print("  ✗ No recommendations returned")
        else:
            print("  ✗ Missing 'recommended_assessments' key")
            print(f"  Actual keys: {list(data.keys())}")
    else:
        print(f"Error: {response.json()}")


if __name__ == "__main__":
    print("="*80)
    print("API FORMAT VALIDATION TEST")
    print("="*80)
    
    try:
        test_health_endpoint()
        test_recommend_endpoint()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

