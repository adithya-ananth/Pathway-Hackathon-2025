- [] Refactor whole codebase to use Langchain (in order):
  
  - [] **Step 1: requirements.txt** or **pyproject.toml** - Update dependencies
    - Remove: `pathway`
    - Add: `langchain`
    - Add: `langchain-community`
    - Add: `langchain-google-genai`
    - Add: `faiss-cpu` (or `faiss-gpu`)
    - Add: `sentence-transformers` (if not already present)
  
  - [] **Step 2: rag/config.py** - Update configuration for LangChain
    - Keep: Existing API key configuration
    - Remove: Pathway-related global variables (if any)
    - Add: LangChain-specific settings (if needed)
    - Update: Global state variables for LangChain compatibility
  
  - [] **Step 3: rag/schemas.py** - Replace Pathway schemas with standard Python structures
    - Remove: `pathway.Schema` imports
    - Remove: `ContentSchema` and `QuerySchema` classes
    - Add: Python dataclasses or TypedDict for data structures
    - Document: Expected structure for papers and queries
  
  - [] **Step 4: rag/formatting.py** - Update to work with LangChain Document objects
    - Update: `extract_metadata_from_result()` for LangChain Document format
    - Update: `format_document()` to handle `Document.metadata` and `Document.page_content`
    - Simplify: Remove Pathway-specific Json wrapper handling
    - Update: Functions to work with standard dicts instead of Pathway types
  
  - [] **Step 5: rag/vector_store.py** - Replace Pathway VectorStoreServer with FAISS/Chroma
    - Remove: `pathway.xpacks.llm.embedders.SentenceTransformerEmbedder`
    - Remove: `pathway.xpacks.llm.vector_store.VectorStoreServer`
    - Remove: `pw.io.jsonlines.read` with standard file reading
    - Add: `langchain_community.embeddings.HuggingFaceEmbeddings`
    - Add: `langchain_community.vectorstores.FAISS` (or `Chroma`)
    - Add: `langchain.schema.Document` for document creation
    - Update: `setup_content_pipeline()` to read JSONL with standard Python
    - Update: `setup_vector_store()` to create and return FAISS instance
    - Add: Option to save/load vector store from disk
  
  - [] **Step 6: rag/query.py** - Replace Pathway query processing with LangChain retrievers
    - Remove: `pathway as pw` imports
    - Remove: `pw.io.jsonlines.read` for queries
    - Remove: `vector_store.retrieve_query()` calls
    - Remove: Pathway table operations
    - Add: `langchain.schema.BaseRetriever` (optional custom retriever)
    - Add: Standard Python file I/O for query reading
    - Update: `setup_query_pipeline()` to read queries with standard Python
    - Update: `process_query_results()` to use `vector_store.similarity_search()` or `retriever.get_relevant_documents()`
    - Update: Return format to use standard Python lists/dicts
  
  - [] **Step 7: rag/answer_generation.py** - Replace direct Gemini API with LangChain wrapper
    - Remove: Direct `google.generativeai` imports and usage
    - Add: `langchain_google_genai.ChatGoogleGenerativeAI`
    - Add: `langchain.prompts.PromptTemplate` for structured prompts
    - Add: `langchain.schema.output_parser.StrOutputParser`
    - Add: `langchain.chains.LLMChain` (optional)
    - Update: `generate_answer_with_context()` to use LangChain ChatGoogleGenerativeAI
    - Update: Prompt construction to use PromptTemplate
  
  - [] **Step 8: rag/main.py** - Replace Pathway pipeline orchestration with standard Python
    - Remove: All `pathway as pw` imports and usage
    - Remove: `pw.io.jsonlines.write()` calls
    - Remove: `pw.apply()`, `pw.this`, `pw.Table`, `pw.join()` operations
    - Remove: `pw.run()` execution
    - Remove: Pathway-specific pipeline construction
    - Add: Standard Python file I/O for writing results
    - Add: LangChain `Runnable` chains (optional, for complex workflows)
    - Update: `create_rag_system()` to orchestrate with standard Python functions
    - Update: Replace streaming pipeline with synchronous execution
    - Simplify: Remove debug snapshot files (or use standard logging)
  
  - [] **Step 9: main.py** (root) - Update RAG module import and usage
    - Update: Import statements if RAG interface changes
    - Test: Ensure FastAPI endpoints still work correctly
    - Update: Error handling for new LangChain-based flow

  - [] **Step 10: parser.py** - Replace Pathway-based DoclingParser with standard implementation
    - Remove: All `pathway as pw` imports and `pw.UDF` inheritance
    - Remove: `from pathway.internals import udfs`
    - Remove: `from pathway.internals.config import _check_entitlements`
    - Remove: `from pathway.optional_import import optional_imports`
    - Remove: `from pathway.xpacks.llm import llms, prompts`
    - Remove: `from pathway.xpacks.llm._utils import _prepare_executor`
    - Remove: `from pathway.xpacks.llm.constants import DEFAULT_VISION_MODEL`
    - Remove: `from pathway.xpacks.llm import _parser_utils`
    - Update: Convert `DoclingParser` from `pw.UDF` to regular Python class
    - Update: Replace `cache_strategy: udfs.CacheStrategy` with standard caching
    - Update: Replace `async_mode` with standard async patterns
    - Add: Standard Docling library usage without Pathway wrapper
    - Add: LangChain document loaders for PDF parsing (PyPDFLoader, UnstructuredPDFLoader)
    - Keep: `extract_text()` fallback function as-is (no Pathway dependency)
    - Simplify: Remove Pathway-specific UDF execution patterns

  - [] **Step 11: keyword_extractor.py** - Migrate to LangChain LLM wrapper
    - Remove: Direct `google.generativeai as genai` usage
    - Add: `langchain_google_genai.ChatGoogleGenerativeAI`
    - Add: `langchain.prompts.PromptTemplate` for keyword extraction prompt
    - Add: `langchain.schema.output_parser.StrOutputParser` or `CommaSeparatedListOutputParser`
    - Update: `extract_keywords_from_prompt()` to use LangChain LLM
    - Update: Use PromptTemplate for structured few-shot examples
    - Optional: Add `LLMChain` for better prompt management

  - [] **Step 12: enrich_papers.py** - Migrate to LangChain LLM wrapper
    - Remove: Direct `google.generativeai as genai` usage
    - Add: `langchain_google_genai.ChatGoogleGenerativeAI`
    - Add: `langchain.prompts.PromptTemplate` for reference extraction and subcategory generation
    - Add: `langchain.schema.output_parser.JsonOutputParser` for structured JSON outputs
    - Update: `extract_references_with_llm()` to use LangChain LLM
    - Update: `build_prompt()` to use PromptTemplate
    - Update: API calls to use LangChain's invoke/batch methods
    - Optional: Add retry strategies using LangChain's built-in retry mechanisms

  - [] **Step 13: scraper.py** - No changes needed (already LangChain-compatible)
    - ✅ No Pathway dependencies
    - ✅ No direct LLM API calls
    - ✅ Uses standard Python libraries (requests, xmltodict, json)
    - Note: This file is already compatible and requires no refactoring

- [] Remove Emojis from codebase
  - [] **rag/vector_store.py** - Remove 📥 emoji from print statement (line 20)
  - [] **rag/query.py** - Remove 📨 emoji from print statement (line 14)
  - [] **rag/main.py** - Remove ✅ emojis from print statements (lines 21, 24, 27, 32)
  - [] **rag/main.py** - Remove 📋 emoji from print statement (line 151)

- [] Additional Refactoring for Post-Migration Compatibility:
  
  - [] **Step 14: pyproject.toml** - Update Poetry dependencies
    - Remove: `pathway = {extras = ["xpack-llm"], version = "^0.26.1"}`
    - Add: `langchain = "^0.2.0"` (or latest stable version)
    - Add: `langchain-community = "^0.2.0"`
    - Add: `langchain-google-genai = "^0.1.0"`
    - Add: `faiss-cpu = "^1.8.0"` (or `faiss-gpu` for GPU support)
    - Keep: `sentence-transformers = "^5.1.0"` (already present)
    - Note: Run `poetry lock --no-update` after changes to update lock file
  
  - [] **Step 15: Data Flow & File Dependencies** - Verify integration points
    - Verify: `keyword_extractor.run_keyword_extraction()` creates `config.json` and `query_stream/input_query.jsonl`
    - Verify: `scraper.fetch_and_save_arxiv_papers()` reads `config.json` and writes `arxiv_papers.jsonl`
    - Verify: `parser.parse_and_save_papers()` reads `arxiv_papers.jsonl` and writes to `papers_text/*.txt`
    - Verify: `enrich_papers.main()` reads `arxiv_papers.jsonl` and `papers_text/*.txt`, writes `content_stream/enriched_papers.jsonl`
    - Verify: `rag.main.main()` reads `query_stream/input_query.jsonl` and `content_stream/enriched_papers.jsonl`
    - Verify: Return value format from `rag.main.main()` matches `(vector_store, answer, documents)` expected by root `main.py`
    - Test: Ensure all intermediate files are created in correct directories
  
  - [] **Step 16: rag/config.py** - Update global state management
    - Review: `LAST_COMPREHENSIVE_ANSWER` and `LAST_TOP5_DOCS` usage pattern
    - Update: Ensure these globals are properly set in LangChain version
    - Consider: Refactoring to return values directly instead of using globals
    - Verify: `config.LAST_COMPREHENSIVE_ANSWER` and `config.LAST_TOP5_DOCS` are accessible from `main.py`
  
  - [] **Step 17: Error Handling & Edge Cases**
    - Add: Proper error handling in `main.py` FastAPI endpoint for each pipeline step
    - Add: Validation that files exist before reading (e.g., check `config.json` exists before scraper runs)
    - Add: Graceful fallbacks if any step fails (currently the whole pipeline fails)
    - Add: Logging instead of print statements for production readiness
    - Consider: Making pipeline steps resumable (skip steps if output files already exist)
  
  - [] **Step 18: Return Value Alignment**
    - Verify: `rag.main.main()` returns tuple `(vector_store, answer, top5_docs)`
    - Update: Root `main.py` expects `documents` but gets `top5_docs` - ensure naming consistency
    - Verify: `answer` is the comprehensive answer string (matches `LAST_COMPREHENSIVE_ANSWER`)
    - Verify: `top5_docs` is a list of formatted document dicts (matches `LAST_TOP5_DOCS`)
    - Test: FastAPI response structure matches frontend expectations
  
  - [] **Step 19: Directory Structure & File Paths**
    - Verify: All hardcoded paths use relative paths from project root
    - Verify: Directory creation (`os.makedirs`) happens before file writes in all modules
    - Verify: File path resolution in `rag/utils.py` works correctly for all modules
    - Test: Pipeline works when run from different working directories
    - Consider: Using pathlib for cross-platform compatibility
  
  - [] **Step 20: Testing & Validation**
    - Test: End-to-end pipeline with sample prompt
    - Test: Each module independently after LangChain migration
    - Verify: No Pathway imports remain (run `grep -r "import pathway" *.py`)
    - Verify: All emojis removed (run `grep -rP "[🔧📥📨✅📋]" *.py`)
    - Test: FastAPI `/prompt` endpoint returns expected JSON structure
    - Test: Error scenarios (missing API keys, network failures, invalid inputs)
    - Performance: Compare speed with/without vector store caching 