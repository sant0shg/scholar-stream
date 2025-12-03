import os
import torch
import pandas as pd
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient

from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Configuration ---
# Single Milvus Lite DB file path
MILVUS_DB_PATH = os.environ.get("MILVUS_DB_PATH", "/Users/admin/Documents/7. IISc/Subjects/04. Data Science In Practice/Projects/arxiv research paper/cloud_run/app/milvus_demo.db")
EMBEDDING_DIM = 384
INPUT_CSV_PATH = os.environ.get("INPUT_CSV_PATH", "./app/papers.csv")
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
FINETUNED_MODEL_PATH = os.environ.get("FINETUNED_MODEL_PATH", "./app/finetuned_model_from_analysis")
TOP_K = int(os.environ.get("TOP_K", 10))

# Two distinct collection names within the single Milvus DB
BASE_COLLECTION_NAME = "research_papers"
CUSTOM_COLLECTION_NAME = "research_papers_custom"

# --- FLASK APP SETUP ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global Variables (Loaded ONCE at startup)
paper_lookup = {}
default_model = None
finetuned_model = None
milvus_client = None


def load_models_and_data():
    """Loads models, data, and initializes a single MilvusClient for the Lite file."""
    global paper_lookup, default_model, finetuned_model, milvus_client
    
    # 1. Load Data for Metadata Lookup
    try:
        papers_df = pd.read_csv(INPUT_CSV_PATH)
        paper_lookup = papers_df.set_index('id').to_dict('index')
        app.logger.info(f"Loaded {len(papers_df)} papers for metadata lookup.")
    except Exception as e:
        app.logger.error(f"Error loading data from {INPUT_CSV_PATH}: {e}")
        return False

    # 2. Load Sentence Transformer Models
    try:
        default_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        # Assuming finetuned model path contains the model files
        finetuned_model = SentenceTransformer(FINETUNED_MODEL_PATH, device=DEVICE) 
        app.logger.info("Models loaded successfully.")
    except Exception as e:
        app.logger.error(f"Error loading models: {e}")
        return False
        
    # 3. Initialize Single Milvus Lite Client from File
    try:
        # Connects to the single Milvus Lite DB file
        milvus_client = MilvusClient(MILVUS_DB_PATH)
        # connections.connect(alias="default", uri=MILVUS_DB_PATH)
        app.logger.info(f"Single Milvus Lite client initialized from {MILVUS_DB_PATH}.")
        
    except Exception as e:
        app.logger.error(f"Error initializing Milvus Lite client from {MILVUS_DB_PATH}: {e}")
        return False
        
    return True


def setup_milvus_collection(collection_name):
    """Connects to Milvus and ensures the collection is created."""
    print("Connecting to Milvus...")
    connections.connect(alias="default", uri=MILVUS_DB_PATH)
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return Collection(collection_name)

    print(f"Creating collection '{collection_name}'...")
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, "Research paper embeddings")
    collection = Collection(collection_name, schema)
    index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 384}}
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Milvus setup complete.")
    return collection

def search_milvus_files(query: str) -> dict:
    """Encodes query and searches two collections within the single Milvus Lite instance."""
    
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    query_result = {"query": query, "base": [], "custom": []}



    # Define the two search runs, pointing to the same client but different collections
    search_runs = [
        # Run 1: Base Model embedding search in Base Collection
        (default_model, BASE_COLLECTION_NAME, "base"),
        # Run 2: Fine-Tuned Model embedding search in Custom Collection
        (finetuned_model, CUSTOM_COLLECTION_NAME, "custom")
    ]


    for model_to_use, collection_name, result_key in search_runs:
        

        try:
            # Encode the query using the correct model for the embedding space
            query_vector = model_to_use.encode(query).tolist() 
            
            app.logger.info(f"Searching {collection_name} with {result_key} model...")
            # collection = setup_milvus_collection(collection_name)
            # The search using MilvusClient.search()
            results = milvus_client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=TOP_K,
                search_params=search_params,
                # Output fields must include 'id' (primary key) and 'title', 'description' (metadata)
                output_fields=["id"]
            )
            
            # Process results 
            if results and results[0]:
                for hit in results[0]:
                    entity = hit.get('entity', {})
                    paper_id = entity.get('id', 'N/A')
                    
                    # Ensure description is extracted and formatted
                    description_raw = entity.get('description', 'N/A')
                    description_snippet = description_raw[:100].replace('\n', ' ') + '...'
                    paper_data = paper_lookup.get(paper_id, {})

                    hit_data = {
                        "id": paper_id,
                        "title": paper_data.get('title', 'N/A'),
                        "description": paper_data.get('description', 'N/A')[:100],
                        "score": round(hit.get('distance', 0.0), 4)
                    }
                    query_result[result_key].append(hit_data)
        
        except Exception as e:
            app.logger.error(f"Error searching {collection_name} ({result_key}): {e}")
            query_result[result_key].append({"error": f"Search failed on {collection_name}: {str(e)}"})

    return query_result


# --- ROUTES ---

@app.route('/')
def home():
    # Simple HTML form for the UI
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scholar Stream</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            #results { margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .base-row { background-color: #e6f7ff; }
            .custom-row { background-color: #fff0f6; }
            .score-base { color: blue; font-weight: bold; }
            .score-custom { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>Scholar Stream (Single Milvus DB, Two Collections)</h2>
        <p>Searching Base Collection: <b>research_papers</b> (using base model embeddings)</p>
        <p>Searching Custom Collection: <b>research_papers_custom</b> (using fine-tuned model embeddings)</p>
        
        <form onsubmit="searchModels(event)">
            <input type="text" id="query" placeholder="Enter search query (e.g., LLM and reasoning)" style="width: 80%; padding: 10px;">
            <button type="submit" style="padding: 10px;">Search</button>
        </form>
        <div id="results"></div>

        <script>
            async function searchModels(event) {
                event.preventDefault();
                const query = document.getElementById('query').value;
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<h3>Searching...</h3>';

                try {
                    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                    const jsonResponse = await response.json();
                    
                    if (jsonResponse.error) {
                        resultsDiv.innerHTML = `<h3>API Error: ${jsonResponse.error}</h3>`;
                        return;
                    }

                    displayResults(jsonResponse.data);
                } catch (error) {
                    resultsDiv.innerHTML = `<h3>Network Error: ${error.message}</h3>`;
                }
            }

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                if (!data || data.query === undefined) {
                    resultsDiv.innerHTML = '<h3>No results found or data structure is invalid.</h3>';
                    return;
                }

                let html = `<h3>Results for: "${data.query}"</h3>`;
                
                // Combine and sort results by score (descending)
                const allResults = [
                    ...data.base.map(r => ({...r, source: 'Base Model (research_papers)', scoreClass: 'score-base', rowClass: 'base-row'})),
                    ...data.custom.map(r => ({...r, source: 'Custom Model (research_papers_custom)', scoreClass: 'score-custom', rowClass: 'custom-row'}))
                ].sort((a, b) => b.score - a.score);

                if (allResults.length === 0) {
                    resultsDiv.innerHTML = '<h3>No results found.</h3>';
                    return;
                }

                html += '<table><thead><tr><th>Source/Collection</th><th>Score (IP)</th><th>Title</th><th>Description Snippet</th></tr></thead><tbody>';
                
                allResults.forEach(r => {
                    html += `
                        <tr class="${r.rowClass}">
                            <td>${r.source}</td>
                            <td class="${r.scoreClass}">${r.score}</td>
                            <td>${r.title}</td>
                            <td>${r.description}</td>
                        </tr>
                    `;
                });

                html += '</tbody></table>';
                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

@app.route('/api/search', methods=['GET'])
def search_api():
    """API endpoint to handle search requests."""
    query = request.args.get('q', default='', type=str)
    if not query:
        return jsonify({"error": "Missing query parameter 'q'."}), 400
    
    # Ensure models and clients are loaded
    if not all([default_model, finetuned_model, milvus_client]):
        # Attempt to load if not already loaded
        if not load_models_and_data():
             return jsonify({"error": "Application failed to load models or data on startup."}), 500

    # Execute the search logic
    try:
        results = search_milvus_files(query)
        return jsonify({"query": query, "data": results}), 200
    except Exception as e:
        app.logger.error(f"Search API failed: {e}")
        return jsonify({"error": "Internal server error during search execution."}), 500

# Run the loader once when the application starts
with app.app_context():
    if not load_models_and_data():
        app.logger.error("Initial load failed. Application may not function correctly.")

# if __name__ == '__main__':
#     # Cloud Run/Gunicorn will manage the port binding, but for local testing:
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host='0.0.0.0', port=port, debug=True)
