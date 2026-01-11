def extract_metadata_from_result(doc) -> dict:
    """Extract metadata dict from a vector store retrieval result item."""
    try:
        if isinstance(doc, dict):
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                return doc["metadata"]
            for k in ("doc", "document"):
                inner = doc.get(k)
                if isinstance(inner, dict):
                    if "metadata" in inner and isinstance(inner["metadata"], dict):
                        return inner["metadata"]
            for key in ("id", "title", "abstract", "authors", "primary_category", "file_path"):
                if key in doc:
                    return doc

        if hasattr(doc, "value"):
            v = getattr(doc, "value")
            if isinstance(v, dict):
                if "metadata" in v and isinstance(v["metadata"], dict):
                    return v["metadata"]
                for k in ("doc", "document"):
                    inner = v.get(k)
                    if isinstance(inner, dict):
                        if "metadata" in inner and isinstance(inner["metadata"], dict):
                            return inner["metadata"]
                for key in ("id", "title", "abstract", "authors", "primary_category", "file_path"):
                    if key in v:
                        return v

        for attr in ("doc", "document"):
            if hasattr(doc, attr):
                inner = getattr(doc, attr)
                if hasattr(inner, "metadata") and isinstance(getattr(inner, "metadata"), dict):
                    return getattr(inner, "metadata")
                if isinstance(inner, dict):
                    if "metadata" in inner and isinstance(inner["metadata"], dict):
                        return inner["metadata"]

        if hasattr(doc, "metadata") and isinstance(getattr(doc, "metadata"), dict):
            return getattr(doc, "metadata")

    except Exception as e:
        print(f"Error extracting metadata from {type(doc)}: {e}")

    return {}


def extract_score_from_result(doc) -> float:
    """Extract similarity score from a vector store retrieval result item."""
    def _as_similarity(val, key_name):
        try:
            x = float(val)
            if key_name in ("dist", "distance"):
                return max(0.0, 1.0 - x)
            return x
        except Exception:
            return 0.0

    def _scan_dict(d: dict) -> float:
        for k in ("similarity", "similarity_score", "score", "relevance", "relevance_score", "dist", "distance"):
            if k in d and d[k] is not None:
                return _as_similarity(d[k], k)
        for kk in ("doc", "document"):
            inner = d.get(kk)
            if isinstance(inner, dict):
                for k in ("similarity", "similarity_score", "score", "relevance", "relevance_score", "dist", "distance"):
                    if k in inner and inner[k] is not None:
                        return _as_similarity(inner[k], k)
        return 0.0

    try:
        if isinstance(doc, dict):
            return _scan_dict(doc)

        if hasattr(doc, "value") and isinstance(getattr(doc, "value"), dict):
            val = _scan_dict(getattr(doc, "value"))
            if val:
                return val

        for k in ("similarity", "similarity_score", "score", "relevance", "relevance_score", "dist", "distance"):
            if hasattr(doc, k):
                v = getattr(doc, k)
                if v is not None:
                    return _as_similarity(v, k)

        for attr in ("doc", "document"):
            if hasattr(doc, attr):
                inner = getattr(doc, attr)
                if isinstance(inner, dict):
                    val = _scan_dict(inner)
                    if val:
                        return val
                else:
                    for k in ("similarity", "similarity_score", "score", "relevance", "relevance_score", "dist", "distance"):
                        if hasattr(inner, k):
                            v = getattr(inner, k)
                            if v is not None:
                                return _as_similarity(v, k)

    except Exception as e:
        print(f"Error extracting score: {e}")

    return 0.0


def format_document(doc):
    """Format a document for output using extracted metadata and score."""
    metadata = extract_metadata_from_result(doc)
    
    if not hasattr(format_document, '_debug_printed'):
        format_document._debug_printed = True
        print(f"Formatting documents (type: {type(doc)})")
    
    score = extract_score_from_result(doc)
    
    formatted = {
        "id": metadata.get("id", "unknown"),
        "title": metadata.get("title", ""),
        "abstract": metadata.get("abstract", ""),
        "authors": metadata.get("authors", []),
        "similarity_score": score,
        "url": metadata.get("url", ""),
        "primary_category": metadata.get("primary_category", ""),
        "file_path": metadata.get("file_path", None),
        "matched_keywords": [],
    }
    
    return formatted


def document_matches_keywords(formatted_doc: dict, keywords: list[str]) -> bool:
    """Check if a formatted document matches the given keywords."""
    keywords_lower = [kw.lower() for kw in keywords]
    
    searchable_text = " ".join([
        str(formatted_doc.get("title", "")),
        str(formatted_doc.get("abstract", "")),
        str(formatted_doc.get("primary_category", "")),
        " ".join(formatted_doc.get("authors", []) or []),
    ]).lower()
    
    matched_keywords = [kw for kw in keywords_lower if kw in searchable_text]
    
    if matched_keywords:
        formatted_doc["matched_keywords"] = matched_keywords
        return True
    
    return False
