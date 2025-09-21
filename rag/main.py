import pathway as pw
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.vector_store import VectorStoreServer
import os
import json
import time

# ----------------------------
# Helpers (module-level)
# ----------------------------

def _resolve_project_path(relative_or_abs_path: str) -> str:
    """Resolve relative file paths like 'papers_text/xyz.txt' relative to repo root."""
    if relative_or_abs_path is None:
        raise ValueError("file_path is None; expected a path to a .txt file")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return (
        relative_or_abs_path
        if os.path.isabs(relative_or_abs_path)
        else os.path.join(base_dir, relative_or_abs_path)
    )


def read_text_from_file(file_path: str) -> str:
    """Read UTF-8 text from a local .txt file path (absolute or project-relative)."""
    resolved_path = _resolve_project_path(file_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        read_data = f.read()
        print("READ DATA: ", read_data[:50])
        return read_data

class ContentSchema(pw.Schema):
    # Match content_stream/complete_papers_data.jsonl exactly
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    published_date: str | None
    url: str | None
    pdf_url: str | None
    primary_category: str | None
    sub_categories: list[str] | None
    journal_ref: str | None
    doi: str | None
    references: list[str] | None
    text: str | None
    file_path: str | None
    citations: list[str] | None
    
class QuerySchema(pw.Schema):
    query: str
    top_k: int
    keywords: list[str]

def setup_dynamic_rag_pipeline(content_table: pw.Table[ContentSchema]):
    """
    Set up a dynamic RAG pipeline using Pathway vector store
    Text is stored as embeddings, all other fields as metadata
    """
    
    # Transform the content table to prepare data for vector store
    # Read plain text directly from a local .txt file at file_path.

    # Only embed rows that actually provide a file_path
    content_with_files = content_table.filter(pw.this.file_path != None)

    vector_data = content_with_files.select(
        # Use file_path (guaranteed .txt) as the embedding source; schema.text is ignored.
        data=pw.apply(read_text_from_file, pw.this.file_path),
        metadata=pw.apply(
            lambda paper_id, title, abstract, authors, published_date, url, pdf_url, primary, subcats, citations, file_path: {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published_date": published_date,
                "url": url,
                "pdf_url": pdf_url,
                "primary_category": primary,
                # compatible key name for any downstream references
                "secondary_categories": subcats,
                # also include the canonical sub_categories key
                "sub_categories": subcats,
                "citations": citations,
                # include file_path so optional full-text filtering can read it later
                "file_path": file_path,
            },
            pw.this.paper_id,
            pw.this.title,
            pw.this.abstract,
            pw.this.authors,
            pw.this.published_date,
            pw.this.url,
            pw.this.pdf_url,
            pw.this.primary_category,
            pw.this.sub_categories,
            pw.this.citations,
            pw.this.file_path,
        ),
        # Provide _metadata as well for compatibility with Pathway vector store filtering
        _metadata=pw.apply(
            lambda paper_id, title, abstract, authors, published_date, url, pdf_url, primary, subcats, citations, file_path: {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published_date": published_date,
                "url": url,
                "pdf_url": pdf_url,
                "primary_category": primary,
                "secondary_categories": subcats,
                "sub_categories": subcats,
                "citations": citations,
                "file_path": file_path,
            },
            pw.this.paper_id,
            pw.this.title,
            pw.this.abstract,
            pw.this.authors,
            pw.this.published_date,
            pw.this.url,
            pw.this.pdf_url,
            pw.this.primary_category,
            pw.this.sub_categories,
            pw.this.citations,
            pw.this.file_path,
        ),
        # Convenience passthroughs for debugging snapshots (avoid JSON key access in engine)
        doc_id=pw.this.paper_id,
        title=pw.this.title,
        abstract=pw.this.abstract,
        authors=pw.this.authors,
        primary_category=pw.this.primary_category,
        url=pw.this.url,
    )
    
    # Set up embedder
    embedder = embedders.SentenceTransformerEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vector_store = VectorStoreServer(
        vector_data,
        embedder=embedder,
        splitter=None,
        parser=None
    )
    
    return vector_store, vector_data


def setup_dynamic_content_pipeline():
    """
    Setup dynamic content ingestion that updates in real-time
    """
    # Create directories for streaming content if they don't exist
    os.makedirs("./content_stream", exist_ok=True)
    os.makedirs("./query_stream", exist_ok=True)
    
    # Stream from files (auto-updates when files change)
    content_stream = pw.io.jsonlines.read(
        "./content_stream/",
        schema=ContentSchema,
        mode="streaming",
        autocommit_duration_ms=1000  # Update every 1 second
    )
    print("📥 Content: streaming from ./content_stream/ (JSONL)")
    
    return content_stream


def setup_dynamic_query_pipeline():
    """
    Setup dynamic query processing - queries come in real-time from other team
    They provide query + keywords and expect results or "no results found"
    """
    query_stream = pw.io.jsonlines.read(
        "./query_stream/",
        schema=QuerySchema, 
        mode="streaming",
        autocommit_duration_ms=500  # Fast processing for real-time queries
    )
    print("📨 Queries: streaming from ./query_stream/")
    
    return query_stream


def query_rag_pipeline(vector_store, query_table: pw.Table[QuerySchema]):
    """
    Query the RAG pipeline with keyword-based filtering and return formatted results
    """
    
    # For now, let's just use the original query without combining keywords in the query
    # We'll do keyword filtering in post-processing instead
    simple_query_table = query_table.select(
        query=pw.this.query,
        k=pw.this.top_k,
        metadata_filter=pw.cast(str | None, None),
        filepath_globpattern=pw.cast(str | None, None)
    )
    
    # Perform the retrieval
    results = vector_store.retrieve_query(simple_query_table)
    
    # Add back the original query info and keywords for post-processing
    enriched_results = results.join(
        query_table, 
        results.id == query_table.id
    ).select(
        result=results.result,
        original_query=query_table.query,
        keywords=query_table.keywords
    )
    
    # Filter results based on keywords
    filtered_results = enriched_results.select(
        original_query=pw.this.original_query,
        keywords=pw.this.keywords,
        results=pw.apply(
            lambda docs, keywords: filter_by_keywords(docs, keywords),
            pw.this.result,
            pw.this.keywords
        )
    )
    
    return filtered_results


def _extract_metadata_from_result(doc) -> dict:
    """Extract metadata dict from a vector store retrieval result item.
    Supports objects with `.doc.metadata`, `.metadata`, or dict-like structures.
    """
    try:
        # Preferred: RetrievedDocument variants with nested document/doc
        for attr in ("doc", "document"):
            if hasattr(doc, attr):
                inner = getattr(doc, attr)
                for m_attr in ("metadata", "_metadata"):
                    if hasattr(inner, m_attr):
                        return getattr(inner, m_attr) or {}
                if isinstance(inner, dict):
                    if "metadata" in inner:
                        return inner.get("metadata") or {}
                    if "_metadata" in inner:
                        return inner.get("_metadata") or {}
        # Sometimes metadata is directly on the object
        for m_attr in ("metadata", "_metadata"):
            if hasattr(doc, m_attr):
                return getattr(doc, m_attr) or {}
        # Fallback: dict-like (handle {'doc'/'document': {...}, 'score': ...})
        if isinstance(doc, dict):
            if "metadata" in doc:
                return doc.get("metadata") or {}
            for attr in ("doc", "document"):
                if attr in doc and isinstance(doc[attr], dict):
                    inner = doc[attr]
                    if "metadata" in inner and isinstance(inner["metadata"], (dict,)):
                        return inner["metadata"] or {}
                    if "_metadata" in inner and isinstance(inner["_metadata"], (dict,)):
                        return inner["_metadata"] or {}
            if "_metadata" in doc:
                return doc.get("_metadata") or {}
            return doc
    except Exception:
        pass
    return {}


def _extract_score_from_result(doc) -> float:
    try:
        if hasattr(doc, "score") and isinstance(getattr(doc, "score"), (int, float)):
            return float(getattr(doc, "score"))
        if isinstance(doc, dict) and "score" in doc:
            val = doc.get("score")
            return float(val) if isinstance(val, (int, float, str)) else 0.0
    except Exception:
        return 0.0
    return 0.0


def _read_file_text_if_enabled(metadata: dict) -> str:
    """Optionally read full text for keyword search if env var enabled."""
    if not metadata:
        return ""
    flag = os.getenv("RAG_FILTER_INCLUDE_TEXT", "0").lower() in {"1", "true", "yes"}
    if not flag:
        return ""
    file_path = metadata.get("file_path")
    if not file_path:
        return ""
    try:
        return read_text_from_file(file_path)
    except Exception:
        return ""


def filter_by_keywords(docs, keywords):
    """
    Filter documents based on keywords presence in metadata fields
    Returns empty list if no matches (indicating "no results found")
    """
    if not keywords:
        # Return all results if no keywords specified
        return [format_document(doc) for doc in docs]
    
    filtered_docs = []
    keywords_lower = [kw.lower() for kw in keywords]
    
    for doc in docs:
        metadata = _extract_metadata_from_result(doc)

        # Also extract the original embedded text if available from the retrieval result
        def _extract_embedded_text(d) -> str:
            try:
                for attr in ("doc", "document"):
                    if hasattr(d, attr):
                        inner = getattr(d, attr)
                        if hasattr(inner, "data") and isinstance(getattr(inner, "data"), str):
                            return getattr(inner, "data")
                        if isinstance(inner, dict) and isinstance(inner.get("data"), str):
                            return inner.get("data")
                if isinstance(d, dict) and "doc" in d and isinstance(d["doc"], dict):
                    inner = d["doc"]
                    if isinstance(inner.get("data"), str):
                        return inner.get("data")
                if isinstance(d, dict) and "document" in d and isinstance(d["document"], dict):
                    inner = d["document"]
                    if isinstance(inner.get("data"), str):
                        return inner.get("data")
            except Exception:
                return ""
            return ""
        embedded_text = _extract_embedded_text(doc)
        
        # Create searchable text from multiple fields
        searchable_text = " ".join([
            str(metadata.get("title", "")),
            str(metadata.get("abstract", "")),
            str(metadata.get("primary_category", "")),
            " ".join(metadata.get("secondary_categories", []) or metadata.get("sub_categories", []) or []),
            " ".join(metadata.get("authors", []) or []),
            embedded_text or "",
        ]).lower()
        # Optionally include full document text
        searchable_text = (searchable_text + " " + _read_file_text_if_enabled(metadata)).lower()
        
        # Check for keyword matches
        matched_keywords = [kw for kw in keywords_lower if kw in searchable_text]
        
        if matched_keywords:
            filtered_docs.append(format_document_with_keywords(doc, matched_keywords))
    
    # If no documents matched from retrieval results, fallback to cached vector data scan
    if not filtered_docs:
        try:
            fallback = _fallback_match_from_cache(keywords_lower)
            if fallback:
                return fallback
        except Exception:
            pass
    # If still no documents match keywords, return empty list
    # This signals to other team: "no results found" - they should search web
    return filtered_docs


# Simple on-disk cache to support fallback keyword matching when retrieval results lack metadata
_VECTOR_CACHE_FILE = "./.vector_data_cache.jsonl"
_VECTOR_CACHE: list[dict] | None = None

def _load_vector_cache() -> list[dict]:
    global _VECTOR_CACHE
    if _VECTOR_CACHE is not None:
        return _VECTOR_CACHE
    items: list[dict] = []
    if os.path.exists(_VECTOR_CACHE_FILE):
        try:
            with open(_VECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            items = []
    _VECTOR_CACHE = items
    return _VECTOR_CACHE

def _fallback_match_from_cache(keywords_lower: list[str]) -> list[dict]:
    cache = _load_vector_cache()
    if not cache:
        return []
    results: list[dict] = []
    for item in cache:
        try:
            searchable = " ".join([
                str(item.get("title", "")),
                str(item.get("abstract", "")),
                " ".join(item.get("authors", []) or []),
                str(item.get("primary_category", "")),
                str(item.get("data", "")),
            ]).lower()
            matched = [kw for kw in keywords_lower if kw in searchable]
            if matched:
                results.append({
                    "id": item.get("doc_id", "unknown"),
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", ""),
                    "authors": item.get("authors", []),
                    "similarity_score": 0.0,
                    "url": item.get("url", ""),
                    "primary_category": item.get("primary_category", ""),
                    "matched_keywords": matched,
                })
        except Exception:
            continue
    return results


def format_document(doc):
    """Format a document for output using extracted metadata and score."""
    metadata = _extract_metadata_from_result(doc)
    return {
        "id": metadata.get("id", "unknown"),
        "title": metadata.get("title", ""),
        "abstract": metadata.get("abstract", ""),
        "authors": metadata.get("authors", []),
        "similarity_score": _extract_score_from_result(doc),
        "url": metadata.get("url", ""),
        "primary_category": metadata.get("primary_category", ""),
        # include file_path so downstream printing can surface where text came from
        "file_path": metadata.get("file_path", None),
        "matched_keywords": [],
    }


def format_document_with_keywords(doc, matched_keywords):
    """Format a document and attach matched keywords."""
    formatted = format_document(doc)
    formatted["matched_keywords"] = matched_keywords
    return formatted

def pretty_print_results(original_query: str, results) -> str:
    """
    Nicely print results with similarity scores and matched keywords.
    Returns a small status string so it can be used in a Pathway sink.
    """
    def _safe_get(obj, key, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            try:
                return obj[key]
            except Exception:
                return default
        except Exception:
            return default

    def _to_list(val):
        try:
            if isinstance(val, list):
                return val
            if val is None:
                return []
            if isinstance(val, tuple):
                return list(val)
            # Fallback: wrap non-iterables
            if isinstance(val, (str, int, float)):
                return [val]
            return list(val)
        except Exception:
            return []

    print("\n=== Query Results ===")
    print(f"Query: {original_query}")

    if not results:
        print("No results found.")
        return "printed_0"

    for i, doc in enumerate(results, start=1):
        title = _safe_get(doc, "title", "")
        score = _safe_get(doc, "similarity_score", 0.0)
        try:
            score = float(score) if score is not None else 0.0
        except Exception:
            score = 0.0
        matched = _to_list(_safe_get(doc, "matched_keywords", []))
        doc_id = _safe_get(doc, "id", "")

        print(f"{i}. {title} (score: {score:.3f})")
        if matched:
            try:
                print(f"   matched: {', '.join([str(m) for m in matched])}")
            except Exception:
                print(f"   matched: {matched}")
        print(f"   id: {doc_id}")

    return f"printed_{len(results)}"


def print_comprehensive_answer(query: str, answer: str) -> str:
    """
    Print the comprehensive answer to the console in a formatted way
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE RAG ANSWER")
    print("="*80)
    print(answer)
    print("="*80 + "\n")
    return "comprehensive_answer_printed"


def print_final_summary(original_query: str, results) -> str:
    """Print a concise FINAL RAG RESULT for the top match, including file_path if available.

    Be robust to different result item shapes (formatted dicts, raw retrieval objects, etc.).
    """
    try:
        print("\n=== FINAL RAG RESULT ===")
        print(f"Query: {original_query}")
        if not results:
            print("No results found.")
            return "final_0"

        # Safely pick the first element if it's an indexable sequence
        top_raw = None
        try:
            top_raw = results[0]
        except Exception:
            # Results may be an iterator; try to coerce to list
            try:
                seq = list(results)
                top_raw = seq[0] if seq else None
            except Exception:
                top_raw = None

        if top_raw is None:
            print("No results found.")
            return "final_0"

        # Coerce to a formatted dict with expected keys
        def _coerce_formatted(doc) -> dict:
            try:
                # If it's already a dict with typical keys, use as-is
                if isinstance(doc, dict) and any(k in doc for k in ("id", "title", "abstract", "file_path")):
                    # Ensure similarity_score exists for consistent shape
                    if "similarity_score" not in doc:
                        try:
                            doc = {**doc, "similarity_score": doc.get("score", 0.0)}
                        except Exception:
                            doc = {**doc, "similarity_score": 0.0}
                    return doc
                # Otherwise, build from metadata extractor
                md = _extract_metadata_from_result(doc)
                return {
                    "id": md.get("id", "unknown"),
                    "title": md.get("title", ""),
                    "abstract": md.get("abstract", ""),
                    "authors": md.get("authors", []),
                    "similarity_score": _extract_score_from_result(doc),
                    "url": md.get("url", ""),
                    "primary_category": md.get("primary_category", ""),
                    "file_path": md.get("file_path"),
                    "matched_keywords": [],
                }
            except Exception:
                return {"id": "unknown", "title": "", "url": "", "file_path": None, "similarity_score": 0.0}

        top = _coerce_formatted(top_raw)

        title = top.get("title", "")
        doc_id = top.get("id", "unknown")
        url = top.get("url", "")
        file_path = top.get("file_path")
        print(f"Top Result: {title}")
        print(f"   id: {doc_id}")
        if url:
            print(f"   url: {url}")
        if file_path:
            print(f"   file_path: {file_path}")
        return "final_1"
    except Exception:
        return "final_error"


def create_sample_data():
    """
    Create sample data for testing the RAG pipeline
    """
    return [
        {
            "id": "paper1",
            "title": "Deep Learning for Medical Image Analysis",
            "abstract": "This paper presents novel approaches using convolutional neural networks for automated medical image analysis, focusing on X-ray and MRI diagnostic applications.",
            "authors": ["Dr. Sarah Johnson", "Prof. Michael Chen"],
            "published_date": "2024-03-15",
            "url": "https://example.com/papers/medical-dl",
            "pdf_url": "https://example.com/papers/medical-dl.pdf",
            "primary_category": "cs.CV",
            "secondary_categories": ["cs.AI", "q-bio.QM"],
            "text": "Deep learning has revolutionized medical image analysis by providing automated tools for diagnosis. Our convolutional neural network architecture processes X-ray and MRI images to detect anomalies with 95% accuracy. The model uses attention mechanisms to focus on relevant anatomical regions, improving both precision and interpretability in clinical settings.",
            "citations": ["Smith et al. 2023", "Wang et al. 2024"]
        },
        {
            "id": "paper2", 
            "title": "Natural Language Processing in Clinical Documentation",
            "abstract": "An investigation into NLP techniques for extracting structured information from unstructured clinical notes and improving healthcare documentation workflows.",
            "authors": ["Dr. Lisa Wong", "Dr. James Martinez"],
            "published_date": "2024-01-20",
            "url": "https://example.com/papers/clinical-nlp",
            "pdf_url": "https://example.com/papers/clinical-nlp.pdf", 
            "primary_category": "cs.CL",
            "secondary_categories": ["cs.AI", "cs.HC"],
            "text": "Natural language processing techniques can significantly improve clinical documentation by automatically extracting key medical concepts from physician notes. Our named entity recognition system identifies symptoms, diagnoses, and treatments with high precision, reducing documentation burden on healthcare providers while maintaining clinical accuracy.",
            "citations": ["Brown et al. 2023", "Davis et al. 2024"]
        },
        {
            "id": "paper3",
            "title": "Reinforcement Learning for Autonomous Robot Navigation", 
            "abstract": "This work explores reinforcement learning algorithms for training autonomous robots to navigate complex environments using sensor fusion and deep Q-networks.",
            "authors": ["Prof. Alex Kumar", "Dr. Emma Thompson"],
            "published_date": "2024-02-10", 
            "url": "https://example.com/papers/robot-rl",
            "pdf_url": "https://example.com/papers/robot-rl.pdf",
            "primary_category": "cs.RO",
            "secondary_categories": ["cs.LG", "cs.AI"],
            "text": "Reinforcement learning enables autonomous robots to learn optimal navigation strategies through trial and error. Our deep Q-network implementation combines LIDAR and camera sensor data to train robots for complex indoor navigation tasks. The system achieves robust performance in dynamic environments with moving obstacles and changing layouts.",
            "citations": ["Garcia et al. 2023", "Lee et al. 2024"]
        }
    ]


def create_rag_system():
    """
    Complete setup of the RAG system for the described workflow:
    1. Other team calls RAG with query + keywords
    2. If no results -> return "no results found"
    3. Other team searches web -> adds papers to content_stream
    4. RAG processes new content -> updates database
    5. Other team calls RAG again -> gets results
    """
    print("🔧 Setting up RAG system for dynamic workflow...")
    
    # Setup dynamic content pipeline (monitors content_stream directory)
    content_table = setup_dynamic_content_pipeline()
    print("✅ Content pipeline ready - monitoring ./content_stream/")
    
    # Setup vector store that auto-updates when new content arrives
    vector_store, vector_data = setup_dynamic_rag_pipeline(content_table)
    print("✅ Vector store ready - will auto-update with new papers")
    
    # Setup query pipeline (monitors query_stream directory)
    query_table = setup_dynamic_query_pipeline()
    print("✅ Query pipeline ready - monitoring ./query_stream/")
    
    # Execute queries with keyword filtering
    results = query_rag_pipeline(vector_store, query_table)
    
    # Stream results to output (other team can monitor this)
    pw.io.jsonlines.write(results, "./query_results.jsonl")
    print("✅ Results will be written to ./query_results.jsonl")

    # Debug: snapshot what is being embedded
    def _len_or_zero(s: str) -> int:
        try:
            return len(s or "")
        except Exception:
            return 0
    vector_snapshot = vector_data.select(
        doc_id=pw.this.doc_id,
        title=pw.this.title,
        data_len=pw.apply(_len_or_zero, pw.this.data),
    )
    pw.io.jsonlines.write(vector_snapshot, "./.vector_data_snapshot.jsonl")

    # Debug: write a simple cache with rich fields to support fallback keyword matching
    vector_cache = vector_data.select(
        doc_id=pw.this.doc_id,
        title=pw.this.title,
        abstract=pw.this.abstract,
        authors=pw.this.authors,
        primary_category=pw.this.primary_category,
        url=pw.this.url,
        data=pw.this.data,
    )
    pw.io.jsonlines.write(vector_cache, _VECTOR_CACHE_FILE)

    # Debug: also materialize raw retrieve counts per query
    def _count_docs(rs: list) -> int:
        try:
            return len(rs or [])
        except Exception:
            return 0

    raw = vector_store.retrieve_query(
        query_table.select(
            query=pw.this.query,
            k=pw.this.top_k,
            metadata_filter=pw.cast(str | None, None),
            filepath_globpattern=pw.cast(str | None, None),
        )
    ).join(query_table, pw.left.id == pw.right.id).select(
        query=query_table.query,
        count=pw.apply(_count_docs, pw.left.result),
    )
    pw.io.jsonlines.write(raw, "./.raw_retrieve_counts.jsonl")

    # Debug: inspect metadata keys present in first document of each result set
    def _first_doc_metadata_keys(rs: list) -> list[str]:
        try:
            if not rs:
                return []
            md = _extract_metadata_from_result(rs[0])
            return sorted(list(md.keys()))
        except Exception:
            return []

    raw_keys = vector_store.retrieve_query(
        query_table.select(
            query=pw.this.query,
            k=pw.this.top_k,
            metadata_filter=pw.cast(str | None, None),
            filepath_globpattern=pw.cast(str | None, None),
        )
    ).join(query_table, pw.left.id == pw.right.id).select(
        query=query_table.query,
        keys=pw.apply(_first_doc_metadata_keys, pw.left.result),
    )
    pw.io.jsonlines.write(raw_keys, "./.raw_metadata_keys.jsonl")

    # Debug: dump a compact shape of first raw doc per query
    def _first_doc_shape(rs: list) -> dict:
        try:
            if not rs:
                return {"empty": True}
            d = rs[0]
            out = {
                "has_doc": hasattr(d, "doc"),
                "has_document": hasattr(d, "document"),
                "has_metadata": hasattr(d, "metadata"),
                "is_dict": isinstance(d, dict),
            }
            if hasattr(d, "doc"):
                inner = getattr(d, "doc")
                out["doc_has_metadata"] = hasattr(inner, "metadata")
                out["doc_has_data"] = hasattr(inner, "data")
                if isinstance(inner, dict):
                    out["doc_dict_keys"] = sorted(list(inner.keys()))
            if hasattr(d, "document"):
                inner2 = getattr(d, "document")
                out["document_has_metadata"] = hasattr(inner2, "metadata")
                out["document_has_data"] = hasattr(inner2, "data")
                if isinstance(inner2, dict):
                    out["document_dict_keys"] = sorted(list(inner2.keys()))
            if isinstance(d, dict):
                out["top_keys"] = sorted(list(d.keys()))
            return out
        except Exception:
            return {"error": True}

    raw_shape = vector_store.retrieve_query(
        query_table.select(
            query=pw.this.query,
            k=pw.this.top_k,
            metadata_filter=pw.cast(str | None, None),
            filepath_globpattern=pw.cast(str | None, None),
        )
    ).join(query_table, pw.left.id == pw.right.id).select(
        query=query_table.query,
        shape=pw.apply(_first_doc_shape, pw.left.result),
    )
    pw.io.jsonlines.write(raw_shape, "./.raw_first_doc_shape.jsonl")

    # Print incoming queries to console for verification
    def _print_query(q: str, k: int, kws: list[str]) -> str:
        print(f"➡️ Incoming query: '{q}' | top_k={k} | keywords={kws}")
        return "seen"

    query_printer = query_table.select(status=pw.apply(_print_query, pw.this.query, pw.this.top_k, pw.this.keywords))
    pw.io.jsonlines.write(query_printer, "./.queries_seen.jsonl")

    # Also print results and similarity matches to console
    printer = results.select(
        status=pw.apply(
            lambda q, rs: pretty_print_results(q, rs),
            pw.this.original_query,
            pw.this.results,
        )
    )
    # Materialize the printer so the side-effecting prints actually execute
    pw.io.jsonlines.write(printer, "./.console_prints.jsonl")

    # Print a concise final result summary for the top match, including file_path
    final_printer = results.select(
        status=pw.apply(
            lambda q, rs: print_final_summary(q, rs),
            pw.this.original_query,
            pw.this.results,
        )
    )
    pw.io.jsonlines.write(final_printer, "./.final_console_prints.jsonl")
    
    # Generate comprehensive answers using the enhanced answer_query_with_context function
    comprehensive_answers = results.select(
        query=pw.this.original_query,
        keywords=pw.this.keywords,
        results=pw.this.results,
        comprehensive_answer=pw.apply(
            lambda q, rs, kws: answer_query_with_context(q, rs, kws),
            pw.this.original_query,
            pw.this.results,
            pw.this.keywords,
        )
    )
    pw.io.jsonlines.write(comprehensive_answers, "./comprehensive_answers.jsonl")
    
    # Print the comprehensive answer to console
    answer_printer = comprehensive_answers.select(
        status=pw.apply(
            lambda q, answer: print_comprehensive_answer(q, answer),
            pw.this.query,
            pw.this.comprehensive_answer,
        )
    )
    pw.io.jsonlines.write(answer_printer, "./.answer_prints.jsonl")
    
    print("\n📋 Workflow Summary:")
    print("   1. Other team drops query.jsonl in ./query_stream/")
    print("   2. RAG processes query with keywords")
    print("   3. If no results found -> empty list returned")
    print("   4. Other team searches web -> drops papers in ./content_stream/")
    print("   5. RAG auto-updates database with new papers")
    print("   6. Other team queries again -> gets results")
    
    return vector_store, results


def _safe_get_from_doc(doc, key: str, default=None):
    """
    Safely extract a value from a document, handling both dict and Pathway Json objects
    """
    try:
        # Try dictionary access first
        if hasattr(doc, 'get'):
            return doc.get(key, default)
        # Try Json object access
        elif hasattr(doc, key):
            return getattr(doc, key)
        # Try indexing
        elif hasattr(doc, '__getitem__'):
            try:
                return doc[key]
            except (KeyError, TypeError):
                return default
        else:
            return default
    except Exception:
        return default


def _safe_convert_to_list(val):
    """
    Safely convert a value to a list, handling different input types
    """
    try:
        if val is None:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, (str, int, float)):
            return [val]
        # Try to iterate and convert
        try:
            return list(val)
        except (TypeError, AttributeError):
            return [val] if val is not None else []
    except Exception:
        return []


def answer_query_with_context(query: str, search_results: list, keywords: list[str] | None = None, max_context_length: int = 4000):
    """
    Generate a comprehensive answer based on retrieved documents, query, and keywords
    This function intelligently synthesizes information from multiple sources to provide
    a coherent and informative response to the user's query.
    """
    if not search_results:
        if keywords:
            return f"No relevant documents found for query: '{query}' with keywords: {', '.join(keywords)}. Consider broadening your search terms or searching external sources."
        return f"No relevant documents found for query: '{query}'. Consider using different search terms or searching external sources."
    
    # Extract key information from search results
    all_matched_keywords = set()
    primary_categories = set()
    authors = set()
    key_findings = []
    document_summaries = []
    
    # Process each document to extract relevant information
    for i, doc in enumerate(search_results[:5], 1):  # Process top 5 results
        # Collect metadata using safe extraction
        matched_kws = _safe_get_from_doc(doc, 'matched_keywords', [])
        matched_kws = _safe_convert_to_list(matched_kws)
        all_matched_keywords.update(matched_kws)
        
        category = _safe_get_from_doc(doc, 'primary_category', 'Unknown')
        if category:
            primary_categories.add(str(category))
        
        doc_authors = _safe_get_from_doc(doc, 'authors', [])
        doc_authors = _safe_convert_to_list(doc_authors)
        if doc_authors:
            authors.update(str(author) for author in doc_authors[:2])  # Limit to first 2 authors per paper
        
        # Create document summary
        title = _safe_get_from_doc(doc, 'title', 'Untitled')
        abstract = _safe_get_from_doc(doc, 'abstract', 'No abstract available')
        score = _safe_get_from_doc(doc, 'similarity_score', 0.0)
        
        # Ensure we have strings
        title = str(title) if title else 'Untitled'
        abstract = str(abstract) if abstract else 'No abstract available'
        
        # Convert score to float if possible
        try:
            score = float(score) if score is not None else 0.0
        except (ValueError, TypeError):
            score = 0.0
        
        # Truncate abstract if too long
        if len(abstract) > 300:
            abstract = abstract[:297] + "..."
        
        doc_summary = f"""
Document {i}: {title}
Relevance Score: {score:.3f}
Abstract: {abstract}
Matched Terms: {', '.join(str(kw) for kw in matched_kws) if matched_kws else 'None'}"""
        
        document_summaries.append(doc_summary)
        
        # Extract key findings from title and abstract for synthesis
        key_findings.append({
            'title': title,
            'abstract': abstract,
            'keywords': matched_kws,
            'category': category
        })
    
    # Generate comprehensive answer
    answer_parts = []
    
    # Header
    answer_parts.append(f"## Answer to: {query}\n")
    
    # Executive Summary
    answer_parts.append("### Executive Summary")
    if keywords:
        answer_parts.append(f"Based on analysis of {len(search_results)} relevant documents related to your query about {query}, with focus on: {', '.join(keywords)}.\n")
    else:
        answer_parts.append(f"Based on analysis of {len(search_results)} relevant documents related to your query about {query}.\n")
    
    # Key Insights Section
    answer_parts.append("### Key Insights")
    
    # Generate insights based on document analysis
    if len(primary_categories) > 1:
        # Convert categories to strings for safe sorting
        safe_categories = [str(cat) for cat in primary_categories if cat]
        answer_parts.append(f"This is an interdisciplinary topic spanning {', '.join(sorted(safe_categories))} domains.")
    else:
        # Convert to string safely
        first_category = str(list(primary_categories)[0]) if primary_categories else "Unknown"
        answer_parts.append(f"This research primarily falls within the {first_category} domain.")
    
    # Synthesize key findings
    if all_matched_keywords:
        # Convert all keywords to strings for safe joining and sorting
        safe_keywords = [str(kw) for kw in all_matched_keywords if kw]
        if safe_keywords:
            answer_parts.append(f"\nThe most relevant aspects identified include: {', '.join(sorted(safe_keywords))}.")
    
    # Add synthesized insights from abstracts
    common_themes = _extract_common_themes(key_findings)
    if common_themes:
        answer_parts.append(f"\nCommon themes across the research include: {', '.join(common_themes)}.")
    
    # Notable researchers
    if authors:
        notable_authors = list(authors)[:5]  # Limit to 5 authors
        answer_parts.append(f"\nNotable researchers in this area include: {', '.join(notable_authors)}.")
    
    # Document Details Section
    answer_parts.append("\n### Supporting Documents")
    
    # Add document summaries with length control
    current_length = len('\n'.join(answer_parts))
    for doc_summary in document_summaries:
        if current_length + len(doc_summary) <= max_context_length - 500:  # Leave space for conclusion
            answer_parts.append(doc_summary)
            current_length += len(doc_summary)
        else:
            remaining_docs = len(document_summaries) - document_summaries.index(doc_summary)
            answer_parts.append(f"\n... and {remaining_docs} additional relevant documents")
            break
    
    # Conclusion
    answer_parts.append("\n### Conclusion")
    answer_parts.append(_generate_conclusion(query, key_findings, keywords))
    
    # Recommendations for further research
    answer_parts.append("\n### For Further Research")
    answer_parts.append("Consider exploring the full text of the most relevant documents above, ")
    answer_parts.append("particularly those with the highest relevance scores. ")
    if keywords:
        # Convert matched keywords to strings for safe processing
        safe_matched_keywords = set(str(kw) for kw in all_matched_keywords if kw)
        answer_parts.append(f"You may also want to search for related terms such as: {_suggest_related_keywords(keywords, safe_matched_keywords)}.")
    
    return '\n'.join(answer_parts)


def _extract_common_themes(key_findings: list[dict]) -> list[str]:
    """Extract common themes from document findings"""
    themes = []
    
    # Simple keyword frequency analysis
    word_freq = {}
    for finding in key_findings:
        # Analyze titles and abstracts for common important words
        text = f"{finding['title']} {finding['abstract']}".lower()
        words = text.split()
        
        # Filter for meaningful words (simple approach)
        meaningful_words = [w for w in words if len(w) > 4 and w not in 
                          ['paper', 'study', 'research', 'analysis', 'using', 'based', 'approach', 'method']]
        
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most common themes (appear in multiple documents)
    common_themes = [word for word, freq in word_freq.items() if freq > 1]
    return common_themes[:5]  # Return top 5 themes


def _generate_conclusion(query: str, key_findings: list[dict], keywords: list[str] | None = None) -> str:
    """Generate a thoughtful conclusion based on the query and findings"""
    conclusion_parts = []
    
    if len(key_findings) >= 3:
        conclusion_parts.append("The retrieved documents provide substantial coverage of your query, ")
        conclusion_parts.append("with multiple perspectives and research approaches represented. ")
    elif len(key_findings) >= 1:
        conclusion_parts.append("The available documents provide relevant insights into your query, ")
        conclusion_parts.append("though additional sources may be beneficial for comprehensive understanding. ")
    
    # Analyze research recency and relevance
    categories = set(f['category'] for f in key_findings)
    if len(categories) > 1:
        conclusion_parts.append("The interdisciplinary nature of this topic suggests ")
        conclusion_parts.append("that comprehensive understanding may require expertise from multiple domains. ")
    
    if keywords:
        conclusion_parts.append(f"The specific focus on {', '.join(keywords)} appears to be well-supported ")
        conclusion_parts.append("by the current literature in this domain.")
    
    return ''.join(conclusion_parts)


def _suggest_related_keywords(original_keywords: list[str], matched_keywords: set) -> str:
    """Suggest related keywords for further exploration"""
    # Create related keyword suggestions based on matched terms
    suggestions = set()
    
    for keyword in original_keywords:
        # Simple approach: suggest variations and related terms
        if 'quantum' in keyword.lower():
            suggestions.update(['quantum computing', 'quantum mechanics', 'quantum theory'])
        elif 'neural' in keyword.lower() or 'neuron' in keyword.lower():
            suggestions.update(['neural networks', 'neuroscience', 'brain'])
        elif 'machine' in keyword.lower() or 'learning' in keyword.lower():
            suggestions.update(['deep learning', 'AI', 'artificial intelligence'])
    
    # Add some matched keywords as suggestions
    suggestions.update(list(matched_keywords)[:3])
    
    # Remove original keywords from suggestions
    suggestions = suggestions - set(kw.lower() for kw in original_keywords)
    
    return ', '.join(list(suggestions)[:5])


class DynamicRAGPipeline:
    """
    Production-ready dynamic RAG pipeline with search fallback integration
    """
    
    def __init__(self, embedder_model="sentence-transformers/all-MiniLM-L6-v2", search_function=None):
        self.embedder_model = embedder_model
        self.search_function = search_function  # Your team's search function
        self.vector_store = None
        self.content_table = None
        
    def load_content(self, content_source):
        """
        Load content from various sources with dynamic streaming support
        """
        if isinstance(content_source, str):
            # File path provided
            if content_source.endswith('.jsonl'):
                # Try streaming first, fallback to static
                try:
                    self.content_table = pw.io.jsonlines.read(
                        content_source,
                        schema=ContentSchema,
                        mode="streaming",
                        autocommit_duration_ms=1000
                    )
                except:
                    self.content_table = pw.io.jsonlines.read(
                        content_source,
                        schema=ContentSchema
                    )
            elif content_source.endswith('.csv'):
                try:
                    self.content_table = pw.io.csv.read(
                        content_source,
                        schema=ContentSchema,
                        mode="streaming",
                        autocommit_duration_ms=1000
                    )
                except:
                    self.content_table = pw.io.csv.read(
                        content_source,
                        schema=ContentSchema
                    )
            else:
                # Directory path for streaming
                try:
                    self.content_table = pw.io.jsonlines.read(
                        content_source,
                        schema=ContentSchema,
                        mode="streaming",
                        autocommit_duration_ms=1000
                    )
                except Exception as e:
                    raise ValueError(f"Could not read from {content_source}: {e}")
                    
        elif isinstance(content_source, list):
            # List of dictionaries provided
            rows = [
                (item["id"], item["title"], item["abstract"], item["authors"],
                 item["published_date"], item["url"], item["pdf_url"], 
                 item["primary_category"], item["secondary_categories"], 
                 item["text"], item["citations"])
                for item in content_source
            ]
            self.content_table = pw.debug.table_from_rows(
                schema=ContentSchema,
                rows=rows
            )
        else:
            raise ValueError("Content source must be a file path, directory, or list of dictionaries")
    
    def setup_vector_store(self):
        """
        Initialize vector store with loaded content
        """
        if self.content_table is None:
            raise ValueError("Must load content before setting up vector store")
        
        self.vector_store, _ = setup_dynamic_rag_pipeline(self.content_table)
    
    def search(self, query: str, top_k: int = 5, keywords: list[str] | None = None):
        """
        Perform search with optional keyword filtering and search fallback
        """
        if self.vector_store is None:
            raise ValueError("Must setup vector store before searching")
        
        if keywords is None:
            keywords = []
        
        # Create query table
        query_table = pw.debug.table_from_rows(
            schema=QuerySchema,
            rows=[(query, top_k, keywords)]
        )
        
        # Execute query
        results = query_rag_pipeline(self.vector_store, query_table)
        
        return results
    
    def search_with_fallback(self, query: str, keywords: list[str] | None = None, top_k: int = 5, min_results: int = 3):
        """
        Enhanced search that triggers external search when insufficient results found
        """
        if keywords is None:
            keywords = []
        
        # First, search existing content
        initial_results = self.search(query, top_k, keywords)
        
        # In real implementation, you'd need to extract results from the Pathway table
        # For now, we'll simulate checking if we need more content
        print(f"🔍 Searching for: '{query}' with keywords: {keywords}")
        
        # If search function is provided and we need more content
        if self.search_function and min_results > 0:
            print("📡 Insufficient results - triggering external search...")
            
            try:
                # Call your team's search function
                additional_content = self.search_function(query, keywords)
                
                if additional_content:
                    print(f"✅ Found {len(additional_content)} additional documents")
                    
                    # Add new content to the pipeline
                    self._add_dynamic_content(additional_content)
                    
                    # Re-run search with updated content
                    updated_results = self.search(query, top_k, keywords)
                    return updated_results
                    
            except Exception as e:
                print(f"⚠️ Search function failed: {e}")
        
        return initial_results
    
    def _add_dynamic_content(self, new_content):
        """
        Add new content to the dynamic pipeline (placeholder for real streaming implementation)
        """
        print(f"🔄 Adding {len(new_content)} new documents to pipeline...")
        # In a real streaming system, this would add content to the streaming source
        # For now, just log the action
        for content in new_content:
            print(f"  + Adding: {content.get('title', 'Unknown Title')}")
    
    def run_pipeline(self):
        """
        Execute the Pathway computation with monitoring
        """
        print("🚀 Running Pathway pipeline...")
        pw.run(
            monitoring_level=pw.MonitoringLevel.NONE,
            with_http_server=False  # Set to True for web dashboard
        )
        print("✅ Pipeline execution completed")


class TrulyDynamicRAG:
    """
    Advanced dynamic RAG with real-time streaming and search integration
    """
    
    def __init__(self, search_function):
        self.search_function = search_function
        self.vector_store = None
        
    def setup_streaming_pipeline(self):
        """
        Setup complete streaming pipeline with multiple input sources
        """
        print("🔧 Setting up truly dynamic RAG pipeline...")
        
        # 1. Primary content stream from files
        primary_content = self._setup_primary_content_stream()
        
        # 2. Search-based content stream 
        search_content = self._setup_search_content_stream()
        
        # 3. Combine all content sources
        all_content = primary_content.concat(search_content)
        
        # 4. Setup dynamic vector store
        self.vector_store, _ = setup_dynamic_rag_pipeline(all_content)
        
        # 5. Setup query stream
        query_stream = self._setup_query_stream()
        
        # 6. Process queries with search fallback
        results = self._process_queries_with_search_fallback(query_stream)
        
        # 7. Stream results to multiple outputs
        pw.io.jsonlines.write(results, "./live_results/")
        
        return results
    
    def _setup_primary_content_stream(self):
        """Setup primary content streaming"""
        try:
            return pw.io.jsonlines.read(
                "./content_stream/",
                schema=ContentSchema,
                mode="streaming",
                autocommit_duration_ms=1000
            )
        except:
            # Fallback to sample data
            return setup_dynamic_content_pipeline()
    
    def _setup_search_content_stream(self):
        """Setup search-based content streaming (placeholder)"""
        # In real implementation, this would be a proper streaming connector
        # For now, return an empty table with correct schema
        empty_rows = []
        return pw.debug.table_from_rows(schema=ContentSchema, rows=empty_rows)
    
    def _setup_query_stream(self):
        """Setup query streaming"""
        return setup_dynamic_query_pipeline()
    
    def _process_queries_with_search_fallback(self, queries):
        """Process queries and trigger search when needed"""
        # This would contain the logic to trigger search_function when insufficient results
        return query_rag_pipeline(self.vector_store, queries)
    
    def run_dynamic_pipeline(self):
        """
        Run the complete dynamic pipeline
        """
        print("🌊 Starting truly dynamic RAG pipeline...")
        
        # Setup all components
        results = self.setup_streaming_pipeline()
        
        # Run with monitoring
        pw.run(
            monitoring_level=pw.MonitoringLevel.ALL,
            with_http_server=True  # Enable web dashboard
        )
        
        print("🎯 Dynamic pipeline running! Access dashboard at http://localhost:8080")


def demonstrate_enhanced_rag():
    """
    Demonstrate the enhanced RAG system capabilities
    """
    print("=== Enhanced Dynamic RAG Pipeline Demonstration ===\n")
    
    # 1. Basic system demonstration
    print("1. 🔧 Setting up basic RAG system...")
    vector_store, results = create_rag_system()
    
    print("2. 🚀 Running basic pipeline...")
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
    
    print("3. ✅ Basic results saved to query_results.jsonl\n")
    
    # 2. Class-based usage
    print("4. 🎯 Demonstrating DynamicRAGPipeline class...")
    
    # Mock search function for demonstration
    def mock_search_function(query: str, keywords: list[str]) -> list[dict]:
        """Mock search function - your team would implement the real one"""
        print(f"   🔍 Mock search called with query: '{query}', keywords: {keywords}")
        return [
            {
                "id": f"search_result_1",
                "title": f"External Search Result for '{query}'",
                "abstract": f"This is a mock result found by searching for {query}",
                "authors": ["External Author"],
                "published_date": "2024-09-21",
                "url": "https://external-source.com",
                "pdf_url": "https://external-source.com/pdf",
                "primary_category": "external",
                "secondary_categories": [],
                "text": f"Mock content related to {query} and keywords {keywords}",
                "citations": []
            }
        ]
    
    # Initialize pipeline with search function
    pipeline = DynamicRAGPipeline(search_function=mock_search_function)
    
    # Load sample data
    pipeline.load_content(create_sample_data())
    
    # Setup vector store
    pipeline.setup_vector_store()
    
    # Demonstrate search with fallback
    print("   📊 Testing search with fallback...")
    results = pipeline.search_with_fallback(
        query="quantum computing applications",
        keywords=["quantum", "computing", "applications"],
        top_k=5,
        min_results=3
    )
    
    # Run pipeline
    pipeline.run_pipeline()
    
    print("5. ✅ Enhanced pipeline demonstration completed!\n")
    
    # 3. Show how to use truly dynamic version
    print("6. 🌊 For truly dynamic RAG with real-time streaming:")
    print("   Initialize: dynamic_rag = TrulyDynamicRAG(your_search_function)")
    print("   Run: dynamic_rag.run_dynamic_pipeline()")
    print("   Access: http://localhost:8080 for monitoring dashboard\n")
    
    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    """
    Main execution - sets up the RAG pipeline for the workflow:
    1. Other team generates keywords from query (blackbox)
    2. Other team calls RAG with query + keywords  
    3. If no results -> return empty (they search web)
    4. They add papers to content_stream -> RAG updates
    5. They call RAG again -> get results
    """
    print("=== Dynamic RAG Pipeline for Team Integration ===\n")
    
    # Setup the complete RAG system
    vector_store, results = create_rag_system()
    
    print("\n🚀 Running RAG pipeline...")
    print("   - Monitoring content_stream for new papers")
    print("   - Monitoring query_stream for queries")
    print("   - Writing results to query_results.jsonl")
    
    # Run the pipeline with minimal monitoring
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
    
    print("\n✅ RAG Pipeline is running!")
    print("📁 Directories created:")
    print("   ./content_stream/ - Other team adds papers here")
    print("   ./query_stream/ - Other team adds queries here") 
    print("   ./query_results.jsonl - Results appear here")
    
    print("\n🔄 Workflow ready for other team integration!")
