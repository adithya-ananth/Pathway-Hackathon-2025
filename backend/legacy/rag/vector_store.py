import os
import pathway as pw
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.vector_store import VectorStoreServer

from .schemas import ContentSchema
from .utils import read_text_from_file


def setup_content_pipeline():
    """Setup content ingestion from JSONL files."""
    os.makedirs("./content_stream", exist_ok=True)
    os.makedirs("./query_stream", exist_ok=True)
    
    content_stream = pw.io.jsonlines.read(
        "./content_stream/",
        schema=ContentSchema,
        mode="static",
    )
    print("📥 Content: reading from ./content_stream/ (JSONL)")
    
    return content_stream


def setup_vector_store(content_table: pw.Table[ContentSchema]):
    """Set up vector store with embeddings from content table."""
    
    content_with_files = content_table.filter(pw.this.file_path != None)

    vector_data = content_with_files.select(
        data=pw.apply(read_text_from_file, pw.this.file_path),
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
                "secondary_categories": list(subcats or []),
                "sub_categories": list(subcats or []),
                "citations": list(citations or []),
                "file_path": str(file_path or ""),
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
        doc_id=pw.this.paper_id,
        title=pw.this.title,
        abstract=pw.this.abstract,
        authors=pw.this.authors,
        primary_category=pw.this.primary_category,
        url=pw.this.url,
    )
    
    embedder = embedders.SentenceTransformerEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = VectorStoreServer(
        vector_data,
        embedder=embedder,
        splitter=None,
        parser=None
    )
    
    return vector_store, vector_data
