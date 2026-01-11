- [] Refactor whole codebase to use Langchain
  - [] **rag/vector_store.py** - Replace Pathway VectorStoreServer with FAISS/Chroma
    - Remove: `pathway.xpacks.llm.embedders.SentenceTransformerEmbedder`
    - Add: `langchain_community.embeddings.HuggingFaceEmbeddings`
    - Remove: `pathway.xpacks.llm.vector_store.VectorStoreServer`
    - Add: `langchain_community.vectorstores.FAISS` or `Chroma`
    - Replace: `pw.io.jsonlines.read` with standard file reading
    - Update: `setup_vector_store()` function to return FAISS instance
  
  - [] **rag/query.py** - Replace Pathway query processing with LangChain retrievers
    - Remove: `pw.io.jsonlines.read` for queries
    - Add: LangChain `Retriever` interface
    - Remove: `vector_store.retrieve_query()` calls
    - Add: `retriever.get_relevant_documents()` or `vector_store.similarity_search()`
    - Update: `process_query_results()` to work with LangChain Documents
  
  - [] **rag/schemas.py** - Replace Pathway schemas with LangChain Documents
    - Remove: `pathway.Schema` imports
    - Remove: `ContentSchema` and `QuerySchema` classes
    - Add: `langchain.schema.Document` usage
    - Update: Data structures to use standard Python dicts/dataclasses
  
  - [] **rag/main.py** - Replace Pathway pipeline orchestration with LangChain chains
    - Remove: All `pathway as pw` imports and usage
    - Remove: `pw.io.jsonlines.write()` calls
    - Remove: `pw.apply()`, `pw.this`, `pw.Table` operations
    - Remove: `pw.run()` execution
    - Add: LangChain `Runnable` chains
    - Add: Standard Python file I/O for results
    - Update: `create_rag_system()` to use LangChain components
  
  - [] **rag/answer_generation.py** - Replace direct Gemini API with LangChain wrapper
    - Remove: Direct `google.generativeai` usage
    - Add: `langchain_google_genai.ChatGoogleGenerativeAI`
    - Add: `PromptTemplate` for structured prompts
    - Add: `StrOutputParser` for response parsing
    - Update: `generate_answer_with_context()` to use LangChain LLM
  
  - [] **rag/formatting.py** - Update to work with LangChain Document objects
    - Update: `extract_metadata_from_result()` for LangChain Document format
    - Update: `format_document()` to handle Document.metadata
    - Simplify: Remove Pathway-specific Json wrapper handling
  
  - [] **rag/config.py** - Update configuration for LangChain
    - Keep: Existing API key configuration
    - Add: LangChain-specific settings (if needed)
    - Update: Global state variables for LangChain compatibility
  
  - [] **requirements.txt** or **pyproject.toml** - Update dependencies
    - Remove: `pathway`
    - Add: `langchain`
    - Add: `langchain-community`
    - Add: `langchain-google-genai`
    - Add: `faiss-cpu` (or `faiss-gpu`)
    - Add: `sentence-transformers` (if not already present)
  
  - [] **main.py** (root) - Update RAG module import and usage
    - Update: Import statements if RAG interface changes
    - Test: Ensure FastAPI endpoints still work correctly

- [] Remove Emojis from codebase
  - [] **rag/vector_store.py** - Remove 📥 emoji from print statements
  - [] **rag/query.py** - Remove 📨 emoji from print statements
  - [] **rag/main.py** - Remove ✅ and 📋 emojis from print statements
  - [] Check all other Python files for emoji usage 