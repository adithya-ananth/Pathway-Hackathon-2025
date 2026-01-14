import pathway as pw

from . import config
from .vector_store import setup_content_pipeline, setup_vector_store
from .query import setup_query_pipeline, process_query_results
from .answer_generation import generate_answer_with_context
from .formatting import extract_metadata_from_result
from .output import (
    pretty_print_results,
    print_comprehensive_answer,
    print_final_summary,
    print_query_info
)


def create_rag_system():
    """Complete setup of the RAG system."""
    print("Setting up RAG system...")
    
    content_table = setup_content_pipeline()
    print("✅ Content pipeline ready")
    
    vector_store, vector_data = setup_vector_store(content_table)
    print("✅ Vector store ready")
    
    query_table = setup_query_pipeline()
    print("✅ Query pipeline ready")
    
    results = process_query_results(vector_store, query_table)
    
    pw.io.jsonlines.write(results, "./query_results.jsonl")
    print("✅ Results will be written to ./query_results.jsonl")

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

    def _first_doc_metadata_keys(rs: list) -> list[str]:
        try:
            if not rs:
                return []
            md = extract_metadata_from_result(rs[0])
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

    query_printer = query_table.select(
        status=pw.apply(print_query_info, pw.this.query, pw.this.top_k, pw.this.keywords)
    )
    pw.io.jsonlines.write(query_printer, "./.queries_seen.jsonl")

    printer = results.select(
        status=pw.apply(
            lambda q, rs: pretty_print_results(q, rs),
            pw.this.original_query,
            pw.this.results,
        )
    )
    pw.io.jsonlines.write(printer, "./.console_prints.jsonl")

    final_printer = results.select(
        status=pw.apply(
            lambda q, rs: print_final_summary(q, rs),
            pw.this.original_query,
            pw.this.results,
        )
    )
    pw.io.jsonlines.write(final_printer, "./.final_console_prints.jsonl")
    
    comprehensive_answers = results.select(
        query=pw.this.original_query,
        keywords=pw.this.keywords,
        results=pw.this.results,
        comprehensive_answer=pw.apply(
            lambda q, rs, kws: generate_answer_with_context(q, rs, kws),
            pw.this.original_query,
            pw.this.results,
            pw.this.keywords,
        )
    )
    pw.io.jsonlines.write(comprehensive_answers, "./comprehensive_answers.jsonl")
    
    def _top5(rs: list[dict]) -> list[dict]:
        try:
            return list(rs[:5]) if rs else []
        except Exception:
            return []

    top5_docs = results.select(
        query=pw.this.original_query,
        top5=pw.apply(_top5, pw.this.results),
    )

    def _capture(answer: str, docs: list[dict]) -> str:
        config.LAST_COMPREHENSIVE_ANSWER = answer
        config.LAST_TOP5_DOCS = docs
        return "captured"

    capture = comprehensive_answers.join(top5_docs, pw.left.query == pw.right.query).select(
        status=pw.apply(_capture, pw.left.comprehensive_answer, pw.right.top5)
    )
    pw.io.jsonlines.write(capture, "./.capture.jsonl")

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
    
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
    return vector_store, config.LAST_COMPREHENSIVE_ANSWER, config.LAST_TOP5_DOCS


def main():
    """Main execution."""
    print("=== Dynamic RAG Pipeline ===\n")
    
    vector_store, answer, top5_docs = create_rag_system()
    
    print("\nOne-shot RAG pipeline completed!")
    print("Directories used:")
    print("   ./content_stream/ - Other team adds papers here")
    print("   ./query_stream/ - Other team adds queries here") 
    print("   ./query_results.jsonl - Results appear here")
    
    if answer is not None:
        print("\nCaptured comprehensive answer string in-memory.")
    if top5_docs is not None:
        print("Captured top-5 docs list in-memory.")
    
    print("\nWorkflow ready!")

    return vector_store, answer, top5_docs


if __name__ == "__main__":
    main()
