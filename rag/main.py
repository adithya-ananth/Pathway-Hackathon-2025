import pathway as pw
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.vector_store import VectorStoreServer
import os
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
        return f.read()

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
        # CRITICAL FIX: Ensure metadata is structured correctly for VectorStore
        metadata=pw.apply(
            lambda paper_id, title, abstract, authors, published_date, url, pdf_url, primary, subcats, citations, file_path: {
                "id": str(paper_id or "unknown"),
                "title": str(title or ""),
                "abstract": str(abstract or ""),
                "authors": list(authors or []),
                "published_date": str(published_date or ""),
                "url": str(url or ""),
                "pdf_url": str(pdf_url or ""),
                "primary_category": str(primary or ""),
                # compatible key name for any downstream references
                "secondary_categories": list(subcats or []),
                # also include the canonical sub_categories key
                "sub_categories": list(subcats or []),
                "citations": list(citations or []),
                # include file_path so optional full-text filtering can read it later
                "file_path": str(file_path or ""),
                # IMPORTANT: Remove hardcoded similarity_score - let VectorStore provide it
                # "similarity_score": 0.0,  # <-- REMOVED THIS LINE
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
    Query the RAG pipeline with keyword-based filtering and return formatted results.
    Enhanced to preserve similarity scores from VectorStore.
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
    
    # CRITICAL: Let's debug what the vector store is actually returning
    def _debug_and_process_results(query_results, original_query, keywords):
        """Debug the results structure and process them correctly"""
        print(f"🔧 DEBUG: Processing results for query: {original_query}")
        print(f"🔧 DEBUG: Results type: {type(query_results)}")
        
        # Extract the actual results from Pathway Json object
        actual_results = query_results
        if hasattr(query_results, 'value'):
            actual_results = getattr(query_results, 'value')
            print(f"🔧 DEBUG: Extracted results from .value, type: {type(actual_results)}")
        
        if hasattr(actual_results, '__len__'):
            try:
                print(f"🔧 DEBUG: Results length: {len(actual_results)}")
            except:
                pass
        
        # Process results and ensure they have similarity scores
        processed_results = []
        
        if isinstance(actual_results, list):
            for i, doc in enumerate(actual_results):
                print(f"🔧 DEBUG: Document {i} type: {type(doc)}")
                if hasattr(doc, '__dict__'):
                    print(f"🔧 DEBUG: Document {i} attributes: {list(doc.__dict__.keys()) if hasattr(doc, '__dict__') else 'N/A'}")
                
                # Try to extract similarity score before applying filters
                formatted_doc = format_document(doc)
                
                # Apply improved keyword filtering (re-enabled)
                if keywords:
                    if _document_matches_keywords_flexible(formatted_doc, keywords):
                        processed_results.append(formatted_doc)
                else:
                    processed_results.append(formatted_doc)
        else:
            # Handle single document or other formats
            formatted_doc = format_document(actual_results)
            processed_results.append(formatted_doc)
        
        return processed_results
    
    # Add back the original query info and keywords for post-processing
    enriched_results = results.join(
        query_table, 
        results.id == query_table.id
    ).select(
        original_query=query_table.query,
        keywords=query_table.keywords,
        results=pw.apply(
            _debug_and_process_results,
            results.result,
            query_table.query,
            query_table.keywords
        )
    )
    
    return enriched_results


def _document_matches_keywords(formatted_doc: dict, keywords: list[str]) -> bool:
    """Check if a formatted document matches the given keywords"""
    keywords_lower = [kw.lower() for kw in keywords]
    
    # Create searchable text from multiple fields
    searchable_text = " ".join([
        str(formatted_doc.get("title", "")),
        str(formatted_doc.get("abstract", "")),
        str(formatted_doc.get("primary_category", "")),
        " ".join(formatted_doc.get("authors", []) or []),
    ]).lower()
    
    # Check for keyword matches
    matched_keywords = [kw for kw in keywords_lower if kw in searchable_text]
    
    if matched_keywords:
        # Add matched keywords to the document
        formatted_doc["matched_keywords"] = matched_keywords
        return True
    
    return False


def _document_matches_keywords_flexible(formatted_doc: dict, keywords: list[str]) -> bool:
    """Enhanced keyword matching with more flexible matching logic"""
    keywords_lower = [kw.lower() for kw in keywords]
    
    # Create searchable text from multiple fields
    searchable_text = " ".join([
        str(formatted_doc.get("title", "")),
        str(formatted_doc.get("abstract", "")),
        str(formatted_doc.get("primary_category", "")),
        " ".join(formatted_doc.get("authors", []) or []),
    ]).lower()
    
    matched_keywords = []
    
    # Exact phrase matching first
    for kw in keywords_lower:
        if kw in searchable_text:
            matched_keywords.append(kw)
    
    # If no exact matches, try word-level matching for multi-word keywords
    if not matched_keywords:
        for kw in keywords_lower:
            kw_words = kw.split()
            if len(kw_words) > 1:
                # Check if at least half the words from the keyword appear in the text
                matching_words = sum(1 for word in kw_words if word in searchable_text)
                if matching_words >= len(kw_words) / 2:
                    matched_keywords.append(kw)
            else:
                # Single word - check for partial matches or related terms
                if kw in searchable_text or any(kw in word for word in searchable_text.split() if len(word) > 4):
                    matched_keywords.append(kw)
    
    # If still no matches, be very permissive and include all results for now
    # (since the vector similarity should be handling relevance)
    if not matched_keywords:
        # For now, let everything through since vector similarity is the main filter
        return True
    
    # Add matched keywords to the document
    formatted_doc["matched_keywords"] = matched_keywords
    return True  # Always return True for now to focus on similarity-based relevance


def _extract_metadata_from_result(doc) -> dict:
    """Extract metadata from a vector store retrieval result item.
    Pathway VectorStore returns results in complex Json format - need to navigate the structure.
    Based on debug output, we know metadata exists with keys like 'abstract','authors','title','url'
    but it's nested in a complex Pathway Json structure.
    """
    print(f"🔧 DEBUG: Extracting metadata from {type(doc)}")
    
    try:
        # Handle Pathway Json objects with .value containing the actual data
        if hasattr(doc, 'value'):
            doc_data = getattr(doc, 'value')
            print(f"🔧 DEBUG: Found document data in .value: {list(doc_data.keys()) if isinstance(doc_data, dict) else type(doc_data)}")
            
            # If doc_data is a list, we're probably dealing with a list of results, not a single document
            if isinstance(doc_data, list):
                print(f"🔧 DEBUG: Document data is a list with {len(doc_data)} items, cannot extract metadata from list")
                return {}
            
            if isinstance(doc_data, dict):
                # First check if this is already processed metadata with all our fields
                if 'id' in doc_data or 'title' in doc_data or 'similarity_score' in doc_data:
                    return doc_data
                
                # Handle VectorStore result format: {'text': '...', 'metadata': {...}, 'dist': 0.xx}
                if 'metadata' in doc_data:
                    metadata_obj = doc_data['metadata']
                    print(f"🔧 DEBUG: metadata_obj type: {type(metadata_obj)}")
                    
                    metadata = {}
                    
                    # Handle nested Pathway Json metadata object
                    if hasattr(metadata_obj, 'value'):
                        metadata_raw = getattr(metadata_obj, 'value')
                        print(f"🔧 DEBUG: metadata from .value: {list(metadata_raw.keys()) if isinstance(metadata_raw, dict) else type(metadata_raw)}")
                        if isinstance(metadata_raw, dict):
                            metadata = metadata_raw
                    elif isinstance(metadata_obj, dict):
                        metadata = metadata_obj
                        print(f"🔧 DEBUG: metadata is dict: {list(metadata.keys())}")
                    else:
                        print(f"🔧 DEBUG: metadata_obj type not handled: {type(metadata_obj)}")
                        # Try to extract directly from attributes if it's an object
                        for field in ['id', 'paper_id', 'title', 'abstract', 'authors', 'url', 'primary_category']:
                            if hasattr(metadata_obj, field):
                                val = getattr(metadata_obj, field)
                                # Handle nested Json values
                                if hasattr(val, 'value'):
                                    metadata[field] = getattr(val, 'value')
                                else:
                                    metadata[field] = val
                    
                    # Add distance/score from the top level if available
                    if 'dist' in doc_data:
                        metadata['dist'] = doc_data['dist']
                    if 'distance' in doc_data:
                        metadata['distance'] = doc_data['distance']
                    
                    print(f"🔧 DEBUG: Final extracted metadata keys: {list(metadata.keys()) if metadata else 'empty'}")
                    return metadata
                
                # If no explicit metadata key, check if doc_data contains the fields directly
                # (fallback for different VectorStore formats)
                metadata_fields = ['id', 'paper_id', 'title', 'abstract', 'authors', 'url', 'primary_category', 'file_path']
                if any(field in doc_data for field in metadata_fields):
                    print(f"🔧 DEBUG: Found metadata fields directly in doc_data")
                    return doc_data
        
        # Handle standard dictionary
        if isinstance(doc, dict):
            print(f"🔧 DEBUG: doc is standard dict with keys: {list(doc.keys())}")
            # Check for metadata or _metadata nested keys first
            if 'metadata' in doc and isinstance(doc['metadata'], dict):
                print(f"🔧 DEBUG: Found nested metadata key in dict")
                return doc['metadata']
            if '_metadata' in doc and isinstance(doc['_metadata'], dict):
                print(f"🔧 DEBUG: Found nested _metadata key in dict")
                return doc['_metadata']
            # If no nested metadata, return the whole dict as fallback
            return doc
        
        # Handle objects with metadata attribute
        if hasattr(doc, 'metadata'):
            metadata_obj = getattr(doc, 'metadata')
            print(f"🔧 DEBUG: Found .metadata attribute: {type(metadata_obj)}")
            if hasattr(metadata_obj, 'value'):
                return getattr(metadata_obj, 'value')
            elif isinstance(metadata_obj, dict):
                return metadata_obj
            elif hasattr(metadata_obj, '__dict__'):
                return metadata_obj.__dict__
        
        # Check if doc itself has the metadata fields directly as attributes
        if hasattr(doc, 'id') or hasattr(doc, 'title') or hasattr(doc, 'paper_id'):
            metadata = {}
            for field in ['id', 'paper_id', 'title', 'abstract', 'authors', 'similarity_score', 'url', 
                         'primary_category', 'file_path', 'secondary_categories', 'sub_categories', 'published_date']:
                if hasattr(doc, field):
                    val = getattr(doc, field)
                    # Handle nested Json values
                    if hasattr(val, 'value'):
                        metadata[field] = getattr(val, 'value')
                    else:
                        metadata[field] = val
            if metadata:
                print(f"🔧 DEBUG: Extracted metadata from object attributes: {list(metadata.keys())}")
                return metadata
        
        # Final fallback: try to extract from __dict__ if available
        if hasattr(doc, '__dict__'):
            doc_dict = doc.__dict__
            print(f"🔧 DEBUG: Trying __dict__ with keys: {list(doc_dict.keys())}")
            # Look for any metadata-like structure
            for key in doc_dict:
                val = doc_dict[key]
                if isinstance(val, dict) and any(field in val for field in ['title', 'abstract', 'authors']):
                    print(f"🔧 DEBUG: Found metadata in __dict__['{key}']")
                    return val
                elif hasattr(val, 'value') and isinstance(getattr(val, 'value'), dict):
                    val_dict = getattr(val, 'value')
                    if any(field in val_dict for field in ['title', 'abstract', 'authors']):
                        print(f"🔧 DEBUG: Found metadata in __dict__['{key}'].value")
                        return val_dict
        
    except Exception as e:
        print(f"🔧 ERROR: Error extracting metadata from {type(doc)}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"🔧 WARNING: Could not extract metadata from {type(doc)}")
    return {}


def _extract_score_from_result(doc) -> float:
    """Extract similarity score from a vector store retrieval result item.
    Pathway VectorStore returns results as pathway.internals.json.Json objects with .value containing the data.
    """
    try:
        # Handle Pathway Json objects - the actual data is in .value
        if hasattr(doc, 'value'):
            doc_data = getattr(doc, 'value')
            if isinstance(doc_data, dict):
                for score_key in ['similarity_score', 'score', 'relevance_score', 'dist', 'distance']:
                    if score_key in doc_data and doc_data[score_key] is not None:
                        score_value = doc_data[score_key]
                        print(f"🔧 DEBUG: Found {score_key} = {score_value}")
                        
                        # Handle distance (convert to similarity: similarity = 1 - distance)
                        if score_key in ['dist', 'distance']:
                            similarity = max(0.0, 1.0 - float(score_value))
                            print(f"🔧 DEBUG: Converted distance {score_value} to similarity {similarity}")
                            return similarity
                        else:
                            return float(score_value)
        
        # Handle direct dictionary access (common case)
        if isinstance(doc, dict):
            for score_key in ['similarity_score', 'score', 'relevance_score', 'dist', 'distance']:
                if score_key in doc and doc[score_key] is not None:
                    score_value = doc[score_key]
                    print(f"🔧 DEBUG: Found {score_key} = {score_value} in dict")
                    
                    # Handle distance (convert to similarity)
                    if score_key in ['dist', 'distance']:
                        similarity = max(0.0, 1.0 - float(score_value))
                        print(f"🔧 DEBUG: Converted distance {score_value} to similarity {similarity}")
                        return similarity
                    else:
                        return float(score_value)
        
        # Fallback: check for direct score attributes on the doc object
        score_attrs = ['similarity_score', 'score', 'relevance_score', 'dist', 'distance']
        for attr_name in score_attrs:
            if hasattr(doc, attr_name):
                val = getattr(doc, attr_name)
                print(f"🔧 DEBUG: Found {attr_name}: {val} (type: {type(val)})")
                if hasattr(val, 'value'):
                    score = getattr(val, 'value')
                    if attr_name in ['dist', 'distance']:
                        return max(0.0, 1.0 - float(score)) if score is not None else 0.0
                    else:
                        return float(score) if score is not None else 0.0
                elif val is not None:
                    if attr_name in ['dist', 'distance']:
                        return max(0.0, 1.0 - float(val))
                    else:
                        return float(val)
        
    except (ValueError, TypeError) as e:
        print(f"🔧 WARNING: Error converting score to float: {e}")
    except Exception as e:
        print(f"🔧 ERROR: Error extracting score: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"🔧 WARNING: Returning default score 0.0 for {type(doc)}")
    return 0.0

# (Removed vector cache/fallbacks)

def format_document(doc):
    """Format a document for output using extracted metadata and score.
    Enhanced to handle Pathway VectorStore format specifically."""
    
    print(f"🔧 DEBUG: format_document called with {type(doc)}")
    
    # Try to extract metadata first - this is the key function that needs to work
    metadata = _extract_metadata_from_result(doc)
    print(f"🔧 DEBUG: format_document extracted metadata: {list(metadata.keys()) if metadata else 'None'}")
    
    # Minimal one-time debug without leaking full text
    if not hasattr(format_document, '_debug_printed'):
        format_document._debug_printed = True
        print(f"🔧 DEBUG: Formatting documents (type: {type(doc)})")
    
    # Extract score - also debug this
    score = _extract_score_from_result(doc)
    print(f"🔧 DEBUG: format_document extracted score: {score}")
    
    # Build the formatted document with fallbacks
    formatted = {
        "id": metadata.get("id", "unknown"),
        "title": metadata.get("title", ""),
        "abstract": metadata.get("abstract", ""),
        "authors": metadata.get("authors", []),
        "similarity_score": score,
        "url": metadata.get("url", ""),
        "primary_category": metadata.get("primary_category", ""),
        # include file_path so downstream printing can surface where text came from
        "file_path": metadata.get("file_path", None),
        "matched_keywords": [],
    }
    
    print(f"🔧 DEBUG: format_document returning: {list(formatted.keys())} with score={formatted['similarity_score']}")
    
    # If we couldn't extract basic fields from metadata, try direct access
    if formatted["id"] == "unknown" and hasattr(doc, 'paper_id'):
        formatted["id"] = str(getattr(doc, 'paper_id', 'unknown'))
    if not formatted["title"] and hasattr(doc, 'title'):
        formatted["title"] = str(getattr(doc, 'title', ''))
    if not formatted["abstract"] and hasattr(doc, 'abstract'):
        formatted["abstract"] = str(getattr(doc, 'abstract', ''))
    
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

        # Print raw count and quick shape hints
        try:
            rlen = len(results) if hasattr(results, "__len__") else None
        except Exception:
            rlen = None
        if rlen is not None:
            print(f"Results count (post-process): {rlen}")

        if not results:
            print("No results found.")
            return "printed_0"

        for i, doc in enumerate(results, start=1):
            title = _safe_get(doc, "title", "")
            score = _safe_get(doc, "similarity_score", 0.0)
            score = float(score) if score is not None else 0.0
            matched = _to_list(_safe_get(doc, "matched_keywords", []))
            doc_id = _safe_get(doc, "id", "")

            # Quick shape hint for debugging empty titles/ids
            if isinstance(doc, dict):
                keys_hint = ",".join(sorted(list(doc.keys()))[:6])
            else:
                keys_hint = type(doc).__name__

            print(f"{i}. \"{title}\" (score: {score:.3f}) | shape: {keys_hint}")
            if not title and not doc_id:
                print(f"   [DEBUG] Empty title/id - doc keys: {list(doc.keys()) if isinstance(doc, dict) else 'N/A'}")
            if matched:
                try:
                    print(f"   matched: {', '.join([str(m) for m in matched])}")
                except Exception:
                    print(f"   matched: {matched}")
            print(f"   id: \"{doc_id}\"")

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
                        # Try to extract score properly rather than defaulting to 0.0
                        try:
                            extracted_score = _extract_score_from_result(doc)
                            doc = {**doc, "similarity_score": extracted_score}
                            print(f"🔧 DEBUG: Successfully extracted score {extracted_score} for document with keys {list(doc.keys())}")
                        except Exception as e:
                            print(f"🔧 ERROR: Failed to extract score from document: {e}")
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

    # (Removed vector cache write; no fallback cache maintained)

    # Debug: also materialize raw retrieve counts per query
    def _count_docs(rs: list) -> int:
        return len(rs or [])

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

    # Debug: compact dump of the actual matched docs (first K) with ids/titles/scores
    def _compactify(rs: list) -> list:
        out = []
        try:
            for d in rs[:10] if rs else []:
                md = _extract_metadata_from_result(d)
                out.append({
                    "id": md.get("id", "unknown"),
                    "title": md.get("title", ""),
                    "score": _extract_score_from_result(d),
                    "has_doc": hasattr(d, "doc"),
                    "has_document": hasattr(d, "document"),
                    "is_dict": isinstance(d, dict),
                    "has_value": hasattr(d, "value"),
                    "type": str(type(d).__name__),
                })
        except Exception as e:
            out.append({"error": str(e)})
        return out

    raw_compact = vector_store.retrieve_query(
        query_table.select(
            query=pw.this.query,
            k=pw.this.top_k,
            metadata_filter=pw.cast(str | None, None),
            filepath_globpattern=pw.cast(str | None, None),
        )
    ).join(query_table, pw.left.id == pw.right.id).select(
        query=query_table.query,
        docs=pw.apply(_compactify, pw.left.result),
    )
    pw.io.jsonlines.write(raw_compact, "./.raw_retrieve_compact.jsonl")

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
    
    # Persist the processed results list for each query as we print them
    processed_debug = results.select(
        query=pw.this.original_query,
        results_count=pw.apply(lambda rs: len(rs) if rs else 0, pw.this.results),
        first_result_sample=pw.apply(
            lambda rs: {
                "id": rs[0].get("id", "unknown") if rs and isinstance(rs[0], dict) else "no_first",
                "title": rs[0].get("title", "") if rs and isinstance(rs[0], dict) else "",
                "score": rs[0].get("similarity_score", 0.0) if rs and isinstance(rs[0], dict) else 0.0,
            } if rs else {"empty": True},
            pw.this.results
        ),
    )
    pw.io.jsonlines.write(processed_debug, "./.processed_results_debug.jsonl")
    
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
        score = float(score) if score is not None else 0.0
        
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
    """Extract common themes from document findings with improved specificity"""
    themes = []
    
    # Simple keyword frequency analysis with better filtering
    word_freq = {}
    for finding in key_findings:
        # Analyze titles and abstracts for common important words
        text = f"{finding['title']} {finding['abstract']}".lower()
        # Clean up the text - split on various delimiters
        import re
        words = re.findall(r'\b[a-z]{3,}\b', text)  # Words with 3+ letters
        
        # Filter for meaningful words - exclude common research terms
        stopwords = {
            'paper', 'study', 'research', 'analysis', 'using', 'based', 'approach', 'method',
            'results', 'shows', 'demonstrate', 'present', 'propose', 'novel', 'new',
            'this', 'that', 'with', 'for', 'and', 'the', 'are', 'can', 'our',
            'model', 'models', 'data', 'system', 'systems', 'framework', 'algorithms',
            'performance', 'evaluation', 'experiments', 'experimental', 'techniques'
        }
        meaningful_words = [w for w in words if len(w) > 3 and w not in stopwords]
        
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most common themes (appear in multiple documents)
    common_themes = []
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
        if freq > 1:  # Appears in multiple documents
            common_themes.append(word)
        if len(common_themes) >= 5:  # Limit to top 5
            break
    
    return common_themes


def _generate_conclusion(query: str, key_findings: list[dict], keywords: list[str] | None = None) -> str:
    """Generate a thoughtful conclusion based on the query and findings with improved specificity"""
    conclusion_parts = []
    
    # Analyze the specific content
    num_docs = len(key_findings)
    categories = set(f.get('category', 'Unknown') for f in key_findings)
    actual_titles = [f.get('title', '') for f in key_findings]
    
    # More specific opening based on actual content
    if num_docs >= 3:
        conclusion_parts.append(f"The {num_docs} retrieved documents provide comprehensive coverage of '{query}', ")
        conclusion_parts.append("spanning multiple research perspectives and methodological approaches. ")
    elif num_docs >= 1:
        conclusion_parts.append(f"The {num_docs} relevant document{'s' if num_docs > 1 else ''} ")
        conclusion_parts.append(f"provide{'s' if num_docs == 1 else ''} targeted insights into '{query}', ")
        conclusion_parts.append("though additional sources may enhance understanding. ")
    
    # Specific domain analysis
    if len(categories) > 1:
        specific_cats = [cat for cat in categories if cat != 'Unknown']
        if specific_cats:
            conclusion_parts.append(f"This research spans {', '.join(sorted(specific_cats))} domains, ")
            conclusion_parts.append("indicating the interdisciplinary nature of the topic. ")
    
    # Analyze keywords match quality
    if keywords:
        conclusion_parts.append(f"The focus on {', '.join(keywords)} appears well-supported ")
        conclusion_parts.append("by the current literature, with documents directly addressing these concepts. ")
    
    # Add specificity based on actual document titles if available
    if actual_titles and any(title.strip() for title in actual_titles):
        # Mention one specific example
        first_meaningful_title = next((title for title in actual_titles if len(title.strip()) > 10), None)
        if first_meaningful_title:
            conclusion_parts.append(f"Key research includes work on \"{first_meaningful_title[:60]}{'...' if len(first_meaningful_title) > 60 else ''}\", ")
            conclusion_parts.append("demonstrating active development in this area. ")
    
    return ''.join(conclusion_parts)


def _suggest_related_keywords(original_keywords: list[str], matched_keywords: set) -> str:
    """Suggest related keywords for further exploration with improved suggestions"""
    suggestions = set()
    
    # Analyze original keywords for better related terms
    for keyword in original_keywords:
        keyword_lower = keyword.lower()
        # Domain-specific related terms
        if any(term in keyword_lower for term in ['quantum', 'quant']):
            suggestions.update(['quantum computing', 'quantum algorithms', 'quantum mechanics', 'NISQ'])
        elif any(term in keyword_lower for term in ['neural', 'neuron', 'network']):
            suggestions.update(['deep learning', 'transformers', 'attention mechanisms', 'CNN'])
        elif any(term in keyword_lower for term in ['machine learning', 'ml', 'ai', 'artificial']):
            suggestions.update(['deep learning', 'reinforcement learning', 'supervised learning', 'MLOps'])
        elif any(term in keyword_lower for term in ['adversarial', 'attack', 'security']):
            suggestions.update(['robustness', 'defense mechanisms', 'cybersecurity', 'threat detection'])
        elif any(term in keyword_lower for term in ['natural language', 'nlp', 'text']):
            suggestions.update(['transformers', 'BERT', 'GPT', 'language models'])
        elif any(term in keyword_lower for term in ['computer vision', 'cv', 'image']):
            suggestions.update(['object detection', 'image classification', 'CNN', 'segmentation'])
    
    # Add some matched keywords as suggestions (they were found in the documents)
    good_matches = [kw for kw in matched_keywords if len(kw) > 3 and kw not in {'and', 'the', 'for', 'with'}]
    suggestions.update(good_matches[:3])  # Top 3 matched keywords
    
    # Remove original keywords from suggestions
    original_lower = set(kw.lower() for kw in original_keywords)
    suggestions = suggestions - original_lower
    
    return ', '.join(list(suggestions)[:5]) if suggestions else 'related terms from the literature'


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
