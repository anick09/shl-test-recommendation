"""
FastAPI REST API for SHL Assessment Recommendation System
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Load environment variables from .env file if it exists
try:
    from load_env import load_env
    load_env()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from recommender import SHLRecommender
import uvicorn

app = FastAPI(title="SHL Assessment Recommendation API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    frontend_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    return {"message": "Frontend not found. API is running."}

# Initialize recommender (lazy loading)
recommender = None

def get_recommender():
    """Lazy initialization of recommender"""
    global recommender
    if recommender is None:
        recommender = SHLRecommender()
    return recommender


class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint"""
    query: Optional[str] = None
    url: Optional[str] = None


class RecommendationItem(BaseModel):
    """Individual recommendation item"""
    url: str
    name: str
    adaptive_support: Optional[str] = "No"
    description: Optional[str] = ""
    duration: Optional[int] = None
    remote_support: Optional[str] = "Yes"
    test_type: Optional[List[str]] = []


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint"""
    recommended_assessments: List[RecommendationItem]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: RecommendationRequest):
    """
    Recommendation endpoint that accepts a query or URL and returns relevant assessments
    
    Args:
        request: Request containing either 'query' (text) or 'url' (job description URL)
    
    Returns:
        JSON response with list of recommended assessments (5-10 items)
    """
    if not request.query and not request.url:
        raise HTTPException(
            status_code=400, 
            detail="Either 'query' or 'url' must be provided"
        )
    
    try:
        rec = get_recommender()
        results = rec.recommend(
            query=request.query or "",
            url=request.url,
            min_results=5,
            max_results=10
        )
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No recommendations found"
            )
        
        # Format response with all required fields
        recommendations = []
        for item in results:
            # Parse test_type field - convert "P,K" to list of full names
            test_type_str = item.get('test_type', '')
            test_types = []
            if test_type_str:
                type_codes = [t.strip() for t in test_type_str.split(',')]
                type_map = {
                    'K': 'Knowledge & Skills',
                    'P': 'Personality & Behaviour',
                    'C': 'Competencies'
                }
                test_types = [type_map.get(code, code) for code in type_codes if code]
            
            recommendations.append(
                RecommendationItem(
                    url=item['assessment_url'],
                    name=item['assessment_name'],
                    adaptive_support=item.get('adaptive_support', 'No'),
                    description=item.get('description', ''),
                    duration=item.get('duration'),
                    remote_support=item.get('remote_support', 'Yes'),
                    test_type=test_types
                )
            )
        
        
        return RecommendationResponse(recommended_assessments=recommendations)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

