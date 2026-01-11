import pathway as pw

from .schemas import QuerySchema
from .formatting import format_document, document_matches_keywords


def setup_query_pipeline():
    """Setup query processing from JSONL files."""
    query_stream = pw.io.jsonlines.read(
        "./query_stream/",
        schema=QuerySchema,
        mode="static",
    )
    print("📨 Queries: reading from ./query_stream/")
    
    return query_stream


def process_query_results(vector_store, query_table: pw.Table[QuerySchema]):
    """Query the RAG pipeline with keyword-based filtering."""
    
    simple_query_table = query_table.select(
        query=pw.this.query,
        k=pw.this.top_k,
        metadata_filter=pw.cast(str | None, None),
        filepath_globpattern=pw.cast(str | None, None)
    )
    
    results = vector_store.retrieve_query(simple_query_table)
    
    def _debug_and_process_results(query_results, original_query, keywords):
        """Debug the results structure and process them correctly."""
        print(f"Processing results for query: {original_query}")
        
        processed_results = []
        
        def _as_list(obj):
            if isinstance(obj, list):
                return obj
            if hasattr(obj, 'value'):
                try:
                    v = obj.value
                    if isinstance(v, list):
                        return v
                except Exception:
                    pass
            try:
                return list(obj)
            except Exception:
                return None

        qr_list = _as_list(query_results)

        if isinstance(qr_list, list):
            for doc in qr_list:
                formatted_doc = format_document(doc)
                
                if keywords:
                    if document_matches_keywords(formatted_doc, keywords):
                        processed_results.append(formatted_doc)
                else:
                    processed_results.append(formatted_doc)
        else:
            formatted_doc = format_document(query_results)
            processed_results.append(formatted_doc)
        
        return processed_results
    
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
