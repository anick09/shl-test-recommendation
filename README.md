# SHL Assessment Recommendation System

An intelligent recommendation system that helps hiring managers and recruiters find the right SHL assessments for their roles using natural language queries or job descriptions.

## Features

- **Natural Language Processing**: Accepts natural language queries or job description URLs
- **RAG-based Retrieval**: Uses semantic search with sentence transformers for initial candidate retrieval
- **LLM Re-ranking**: Leverages Google Gemini API for intelligent re-ranking
- **Balanced Recommendations**: Ensures balanced recommendations across test types (Knowledge/Skills and Personality/Behavior)
- **REST API**: FastAPI-based RESTful API with `/health` and `/recommend` endpoints
- **Web Frontend**: Modern, responsive web interface for testing

## Project Structure

```
.
├── api.py                      # FastAPI REST API
├── data_processor.py           # Data loading and preprocessing
├── embedding_service.py        # Embedding generation and semantic search
├── recommender.py              # Core recommendation engine
├── evaluate.py                 # Evaluation script (Mean Recall@10)
├── generate_predictions.py    # Generate test set predictions
├── requirements.txt           # Python dependencies
├── frontend/
│   └── index.html             # Web frontend
└── README.md                  # This file
```

## Setup

### Prerequisites

- Python 3.8+
- Google Gemini API key (get one at https://ai.google.dev/gemini-api/docs/pricing)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd shl_assesment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file or set environment variable
export GEMINI_API_KEY="your-api-key-here"
```

On Windows:
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

4. Prepare the catalog data:
```bash
# The catalog data should be in shl_catalog.csv
# Run data processor to validate and filter
python data_processor.py
```

5. Build embeddings (first run will download the model):
```bash
python embedding_service.py
```

## Usage

### Running the API

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```

Response:
```json
{"status": "healthy"}
```

#### Get Recommendations
```bash
POST /recommend
Content-Type: application/json

{
  "query": "I am hiring for Java developers who can collaborate effectively",
  "url": null  // optional
}
```

Response:
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/assessment-name/",
      "name": "Assessment Name",
      "adaptive_support": "No",
      "description": "Assessment: Assessment Name",
      "duration": null,
      "remote_support": "Yes",
      "test_type": [
        "Personality & Behaviour",
        "Knowledge & Skills"
      ]
    },
    ...
  ]
}
```

### Using the Web Frontend

1. Start the API server:
```bash
python api.py
```

2. Open `frontend/index.html` in a web browser

3. Enter your query or job description URL and click "Get Recommendations"

### Evaluation

Evaluate on the training set:
```bash
python evaluate.py
```

Generate predictions for test set:
```bash
python generate_predictions.py
```

## Architecture

### Data Pipeline
1. **Catalog Loading**: Loads and filters SHL catalog (excludes Pre-packaged Job Solutions)
2. **Data Validation**: Ensures all required fields are present
3. **Embedding Generation**: Creates vector embeddings for all assessments using sentence-transformers

### Recommendation Engine
1. **Semantic Search**: Uses cosine similarity to find top 20 candidate assessments
2. **LLM Re-ranking**: Gemini API re-ranks candidates based on query relevance
3. **Test Type Balancing**: Ensures balanced recommendations when query spans multiple domains
4. **Result Formatting**: Returns 5-10 recommendations with assessment name and URL

### Evaluation
- **Mean Recall@10**: Measures how many relevant assessments were retrieved in top 10 recommendations
- Evaluated on provided training set with 10 labeled queries

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (required for LLM re-ranking)

### Model Configuration
- Embedding Model: `all-MiniLM-L6-v2` (sentence-transformers)
- LLM Model: `gemini-pro` (Google Gemini)

## Deployment

### Local Development
```bash
python api.py
```

### Production Deployment
The API can be deployed to platforms like:
- Railway
- Render
- Heroku
- AWS/GCP/Azure

Example for Railway:
1. Create a `Procfile`:
```
web: python api.py
```

2. Set environment variables in Railway dashboard

3. Deploy from GitHub

## Notes

- The system requires internet connection for Gemini API calls
- First run will download the embedding model (~80MB)
- Embeddings are cached in `embeddings_cache.pkl` for faster subsequent runs
- Protobuf compatibility: The system sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to handle compatibility issues

## License

This project is for assessment purposes.


