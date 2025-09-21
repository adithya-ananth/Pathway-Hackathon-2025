# Dynamic RAG Implementation with Search Fallback

## 🎯 Overview

This implementation provides a **truly dynamic RAG (Retrieval-Augmented Generation) pipeline** using Pathway that:

1. **Streams content in real-time** from files, databases, or APIs
2. **Processes queries dynamically** as they arrive
3. **Automatically triggers external search** when insufficient results are found
4. **Integrates seamlessly** with your team's search function
5. **Updates embeddings incrementally** as new content arrives

## 🔧 Key Dynamic Features Implemented

### 1. **Real-Time Content Streaming**
```python
# Automatically updates when files change
content_stream = pw.io.jsonlines.read(
    "./content_stream/",
    schema=ContentSchema,
    mode="streaming",           # 🔥 Real-time updates
    autocommit_duration_ms=1000 # Updates every 1 second
)
```

### 2. **Dynamic Query Processing**  
```python
# Queries processed as they arrive
query_stream = pw.io.jsonlines.read(
    "./query_stream/",
    schema=QuerySchema,
    mode="streaming",
    autocommit_duration_ms=500  # Faster query processing
)
```

### 3. **Search Fallback Integration**
```python
# Your team's search function integration
pipeline = DynamicRAGPipeline(search_function=your_search_function)

# Automatically triggers external search when needed
results = pipeline.search_with_fallback(
    query="user query",
    keywords=["keyword1", "keyword2"],
    min_results=3  # Triggers search if < 3 local results
)
```

## 🏗️ Architecture Flow

```
📥 Content Sources → 🧠 Vector Store → 🔍 Query Processing → 📊 Results
      ↓                    ↓                    ↓              ↓
  [Files/APIs]        [Embeddings]      [Semantic Search]  [Filtered]
      ↓                    ↓                    ↓              ↓
  [Streaming]         [Real-time]       [+ Keyword Filter] [+ Search]
                                              ↓              ↓
                                        [Insufficient?] → [External]
                                              ↓              ↓
                                           [No: Return]  [Yes: Search]
                                                           ↓
                                                    [Add New Content]
                                                           ↓
                                                     [Re-run Query]
```

## 🚀 Integration with Your Team's Search Function

### Step 1: Implement Your Search Function

Your team needs to implement this interface:

```python
def your_search_function(query: str, keywords: list[str]) -> list[dict]:
    """
    Your team's search function that returns data in ContentSchema format
    
    Args:
        query: The search query string
        keywords: List of keywords for filtering
        
    Returns:
        List of dictionaries matching ContentSchema structure
    """
    
    # Your implementation here - could call:
    # - External APIs (arXiv, PubMed, Google Scholar, etc.)
    # - Web scraping services
    # - Database queries
    # - Other search services
    
    # Example return format:
    return [
        {
            "id": "unique_id",
            "title": "Paper Title",
            "abstract": "Paper abstract...",
            "authors": ["Author1", "Author2"],
            "published_date": "2024-09-21",
            "url": "https://source-url.com",
            "pdf_url": "https://pdf-url.com",
            "primary_category": "category",
            "secondary_categories": ["cat1", "cat2"],
            "text": "Full paper text content...",
            "citations": ["Reference1", "Reference2"]
        }
        # ... more results
    ]
```

### Step 2: Initialize the Dynamic Pipeline

```python
from rag.main import DynamicRAGPipeline

# Initialize with your search function
pipeline = DynamicRAGPipeline(search_function=your_search_function)

# Load initial content (optional)
pipeline.load_content("path/to/initial/content.jsonl")
# OR
pipeline.load_content(list_of_documents)

# Setup vector store
pipeline.setup_vector_store()
```

### Step 3: Use the Pipeline

```python
# Search with automatic fallback
results = pipeline.search_with_fallback(
    query="machine learning in healthcare",
    keywords=["machine learning", "healthcare", "medical"],
    top_k=10,
    min_results=5  # Triggers search if fewer than 5 local results
)

# Execute the pipeline
pipeline.run_pipeline()

# Results will be in query_results.jsonl
```

## 🌊 Truly Dynamic Usage (Real-time Streaming)

For production use with real-time streaming:

### Setup Streaming Directories

```bash
mkdir -p content_stream query_stream live_results
```

### Add Content Dynamically

```bash
# Add new content files (auto-detected)
echo '{"id": "new1", "title": "New Paper", ...}' >> content_stream/new_papers.jsonl
```

### Add Queries Dynamically  

```bash
# Add new queries (auto-processed)
echo '{"query": "AI safety", "top_k": 5, "keywords": ["safety", "AI"]}' >> query_stream/queries.jsonl
```

### Use Advanced Dynamic Pipeline

```python
from rag.main import TrulyDynamicRAG

# Initialize with your search function
dynamic_rag = TrulyDynamicRAG(search_function=your_search_function)

# Run streaming pipeline
dynamic_rag.run_dynamic_pipeline()  # Runs indefinitely, processing streams
```

## 📊 Monitoring & Production Features

### Enable Monitoring Dashboard

```python
# Run with web dashboard
pw.run(
    monitoring_level=pw.MonitoringLevel.ALL,
    with_http_server=True
)
```

### Multiple Output Streams

```python
# Stream results to multiple destinations
pw.io.jsonlines.write(results, "./results.jsonl")     # File
pw.io.kafka.write(results, kafka_settings)            # Kafka  
pw.io.http.rest_connector_serve(results)              # REST API
```

## 🔄 How Search Fallback Works

1. **Query Arrives**: User submits a query with keywords
2. **Local Search**: System searches existing vector store
3. **Evaluation**: Checks if results are sufficient (configurable threshold)
4. **Trigger Search**: If insufficient, calls your search function
5. **Content Addition**: New content added to pipeline dynamically  
6. **Re-run Query**: Query executed again with expanded content
7. **Return Results**: Combined local + external results returned

## 🛠️ Configuration Options

### Streaming Settings

```python
# Adjust streaming sensitivity
content_stream = pw.io.jsonlines.read(
    "./content_stream/",
    schema=ContentSchema,
    mode="streaming",
    autocommit_duration_ms=500  # Faster updates (500ms)
)
```

### Search Thresholds

```python
# Configure when to trigger external search
results = pipeline.search_with_fallback(
    query="query",
    min_results=3,      # Minimum results before triggering search
    top_k=10,          # Maximum results to return
    keywords=["key1"]   # Required keywords
)
```

### Embedding Models

```python
# Use different embedding models
pipeline = DynamicRAGPipeline(
    embedder_model="sentence-transformers/all-mpnet-base-v2"  # Higher quality
)
```

## 🧪 Testing Your Integration

Run the provided test script:

```bash
poetry run python test_dynamic_rag.py
```

This will:
- ✅ Test basic pipeline setup
- ✅ Test local search functionality  
- ✅ Test external search fallback
- ✅ Demonstrate integration points

## 📝 Data Format Requirements

Your search function must return data in this exact format:

```python
{
    "id": str,                          # Unique identifier
    "title": str,                       # Document title
    "abstract": str,                    # Document summary
    "authors": list[str],              # List of author names
    "published_date": str,             # Date string (YYYY-MM-DD)
    "url": str,                        # Document URL
    "pdf_url": str,                    # PDF download URL
    "primary_category": str,           # Main category
    "secondary_categories": list[str], # Additional categories
    "text": str,                       # Full document text
    "citations": list[str]             # List of references
}
```

## 🎯 Production Deployment

### 1. Container Setup
```dockerfile
FROM python:3.13
COPY . /app
WORKDIR /app
RUN poetry install --no-root
CMD ["poetry", "run", "python", "-m", "rag.main"]
```

### 2. Environment Variables
```bash
export PATHWAY_MONITORING=1
export PATHWAY_LOG_LEVEL=INFO
export SEARCH_API_KEY=your_api_key
```

### 3. Service Integration
```python
# Service wrapper
class RAGService:
    def __init__(self, search_function):
        self.pipeline = DynamicRAGPipeline(search_function)
        
    def search(self, query, keywords=None):
        return self.pipeline.search_with_fallback(query, keywords)
        
    def health_check(self):
        return {"status": "healthy", "pipeline": "ready"}
```

## 🚨 Important Notes

1. **First Run**: Initial embedding computation takes time
2. **Memory**: Large document collections require sufficient RAM  
3. **Streaming**: Results update in real-time as content changes
4. **Search Function**: Must be thread-safe and handle errors gracefully
5. **Rate Limits**: Consider API rate limits in your search function

## 🔧 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   poetry install --no-root
   ```

2. **Streaming Directory Issues**
   ```bash
   mkdir -p content_stream query_stream
   ```

3. **Memory Issues**
   - Reduce batch size
   - Use lighter embedding models
   - Implement content chunking

4. **Search Function Errors**
   - Add proper error handling
   - Implement retries with backoff
   - Log failures for debugging

This implementation provides a complete foundation for dynamic RAG with search integration. Your team just needs to implement the search function interface and configure the streaming sources as needed!
