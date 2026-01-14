from .utils import safe_get_from_doc, safe_convert_to_list
from .formatting import extract_metadata_from_result, extract_score_from_result


def pretty_print_results(original_query: str, results) -> str:
    """Nicely print results with similarity scores and matched keywords."""
    print("\n=== Query Results ===")
    print(f"Query: {original_query}")

    if not results:
        print("No results found.")
        return "printed_0"

    for i, doc in enumerate(results, start=1):
        title = safe_get_from_doc(doc, "title", "")
        score = safe_get_from_doc(doc, "similarity_score", 0.0)
        score = float(score) if score is not None else 0.0
        matched = safe_convert_to_list(safe_get_from_doc(doc, "matched_keywords", []))
        doc_id = safe_get_from_doc(doc, "id", "")

        print(f"{i}. {title} (score: {score:.3f})")
        if matched:
            try:
                print(f"   matched: {', '.join([str(m) for m in matched])}")
            except Exception:
                print(f"   matched: {matched}")
        print(f"   id: {doc_id}")

    return f"printed_{len(results)}"


def print_comprehensive_answer(query: str, answer: str) -> str:
    """Print the comprehensive answer to the console."""
    print("\n" + "="*80)
    print("COMPREHENSIVE RAG ANSWER")
    print("="*80)
    print(answer)
    print("="*80 + "\n")
    return "comprehensive_answer_printed"


def print_final_summary(original_query: str, results) -> str:
    """Print a concise final summary for the top match."""
    try:
        print("\n=== FINAL RAG RESULT ===")
        print(f"Query: {original_query}")
        if not results:
            print("No results found.")
            return "final_0"

        top_raw = None
        try:
            top_raw = results[0]
        except Exception:
            try:
                seq = list(results)
                top_raw = seq[0] if seq else None
            except Exception:
                top_raw = None

        if top_raw is None:
            print("No results found.")
            return "final_0"

        def _coerce_formatted(doc) -> dict:
            if isinstance(doc, dict) and any(k in doc for k in ("id", "title", "abstract", "file_path")):
                if "similarity_score" not in doc:
                    doc = {**doc, "similarity_score": doc.get("score", 0.0)}
                return doc
            md = extract_metadata_from_result(doc)
            return {
                "id": md.get("id", "unknown"),
                "title": md.get("title", ""),
                "abstract": md.get("abstract", ""),
                "authors": md.get("authors", []),
                "similarity_score": extract_score_from_result(doc),
                "url": md.get("url", ""),
                "primary_category": md.get("primary_category", ""),
                "file_path": md.get("file_path"),
                "matched_keywords": [],
            }

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


def print_query_info(q: str, k: int, kws: list[str]) -> str:
    """Print incoming query information."""
    print(f"➡️ Incoming query: '{q}' | top_k={k} | keywords={kws}")
    return "seen"
