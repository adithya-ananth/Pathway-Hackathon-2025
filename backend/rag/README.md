# RAG Module - Clean Architecture

This folder contains the modularized RAG (Retrieval-Augmented Generation) pipeline.

## 📁 Module Structure

```
rag/
├── __init__.py              # Package initialization
├── config.py                # Environment configuration & API setup
├── schemas.py               # Pathway data schemas (ContentSchema, QuerySchema)
├── utils.py                 # Helper functions (file I/O, path resolution, safe getters)
├── vector_store.py          # Vector store setup and management
├── query.py                 # Query processing and retrieval logic
├── formatting.py            # Document formatting and metadata extraction
├── answer_generation.py     # LLM-based answer generation using Gemini
├── output.py                # Console output and printing functions
└── main.py                  # Main orchestration and pipeline execution
```

## 📋 Module Responsibilities

### `config.py`
- Loads environment variables
- Configures Gemini API
- Stores global state variables

### `schemas.py`
- Defines `ContentSchema` for paper data
- Defines `QuerySchema` for user queries
- Pathway schema definitions

### `utils.py`
- File path resolution
- Text file reading
- Safe data extraction from Pathway objects
- Type conversion utilities

### `vector_store.py`
- Content pipeline setup (reading from JSONL)
- Vector store initialization with embeddings
- Sentence transformer embedder configuration

### `query.py`
- Query pipeline setup
- Vector similarity search
- Keyword-based filtering
- Result processing

### `formatting.py`
- Metadata extraction from retrieval results
- Similarity score extraction
- Document formatting for output
- Keyword matching logic

### `answer_generation.py`
- Gemini LLM integration
- Comprehensive answer generation
- Theme extraction from papers
- Related keyword suggestions
- Research conclusion generation

### `output.py`
- Pretty printing of query results
- Console formatting
- Final summary generation
- Query information display

### `main.py`
- Pipeline orchestration
- Pathway table joins and transformations
- JSONL output writing
- Debug snapshot generation
- Main entry point

## 🚀 Usage

```python
from rag import main, create_rag_system

# Run the complete pipeline
vector_store, answer, top5_docs = main()

# Or use directly
vector_store, answer, top5_docs = create_rag_system()
```

## 🔄 Data Flow

1. **Content Ingestion** (`vector_store.py`)
   - Read papers from `content_stream/enriched_papers.jsonl`
   - Extract text from file paths
   - Create embeddings

2. **Query Processing** (`query.py`)
   - Read queries from `query_stream/input_query.jsonl`
   - Perform vector similarity search
   - Apply keyword filtering

3. **Answer Generation** (`answer_generation.py`)
   - Extract insights from top documents
   - Generate comprehensive answer using Gemini
   - Format with sources and recommendations

4. **Output** (`output.py`)
   - Print results to console
   - Write to `query_results.jsonl`
   - Save comprehensive answers to `comprehensive_answers.jsonl`

## 📝 Key Improvements

- **Separation of Concerns**: Each module has a single, clear responsibility
- **Cleaner Code**: Removed debug comments and bloat
- **Easier Testing**: Modular functions can be tested independently
- **Better Maintainability**: Changes are isolated to specific modules
- **Ready for Migration**: Easy to swap Pathway with LangChain later
