import pathway as pw
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.vector_store import VectorStoreServer
import logging
import json

class ContentSchema(pw.Schema):
    id: str
    title: str
    abstract: str
    authors: list[str]
    published_date: str
    url: str
    pdf_url: str
    primary_category: str
    secondary_categories: list[str]
    text: str
    citations: list[str]
    
class QuerySchema(pw.Schema):
    query: str
    top_k: int = 5

class MainVectorSchema(pw.Schema):
    id: str
    title: str
    hashed_title: str
    primary_category: str
    secondary_categories: list[str]
    text: str
    metadata: dict

class IndexVectorSchema(pw.Schema):
    id: str
    hashed_title : str
    primary_category: str
    secondary_categories: list[str]

def setup_dynamic_rag_pipeline(content_table: pw.Table[ContentSchema]):
    """
    Set up a dynamic RAG pipeline using Pathway vector store
    """
    
    # Transform the content table to prepare data for vector store
    vector_data = content_table.select(
        id=pw.this.id,
        title=pw.this.title,
        primary_category=pw.this.primary_category,
        secondary_categories=pw.this.secondary_categories,
        # Combine text fields for better embedding
        combined_text=pw.this.title + " | Category: " + pw.this.primary_category + " | " + 
                     pw.apply(lambda cats: "Secondary: " + ", ".join(cats) if cats else "", pw.this.secondary_categories),
        # Store metadata for retrieval
        metadata=pw.apply(
            lambda id, title, primary, secondary, url, pdf_url: {
                "id": id,
                "title": title,
                "primary_category": primary,
                "secondary_categories": secondary,
                "url": url,
                "pdf_url": pdf_url
            },
            pw.this.id,
            pw.this.title,
            pw.this.primary_category,
            pw.this.secondary_categories,
            pw.this.url,
            pw.this.pdf_url
        )
    )
    
    # Set up embedder
    embedder = embedders.SentenceTransformerEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vector_store = VectorStoreServer(
        vector_data,
        embedder=embedder,
        splitter=None,  # No splitting needed for structured data
        parser=None     # No parsing needed
    )
    
    return vector_store, vector_data


def query_rag_pipeline(vector_store, query_table: pw.Table[QuerySchema]):
    """
    Query the RAG pipeline and return formatted results
    """
    
    # Perform similarity search
    results = vector_store.similarity_search(
        query_table.select(
            query=pw.this.query,
            k=pw.this.top_k
        )
    )
    
    # Format results with metadata
    formatted_results = results.select(
        query=pw.this.query,
        results=pw.apply(
            lambda docs: [
                {
                    "id": doc.metadata["id"],
                    "title": doc.metadata["title"],
                    "primary_category": doc.metadata["primary_category"],
                    "secondary_categories": doc.metadata["secondary_categories"],
                    "url": doc.metadata["url"],
                    "pdf_url": doc.metadata["pdf_url"],
                    "similarity_score": float(doc.dist)
                }
                for doc in docs
            ],
            pw.this.documents
        )
    )
    
    return formatted_results


def create_sample_data():
    """
    Create sample data for testing the RAG pipeline
    """
    sample_content = [
        {
            "id": "paper1",
            "title": "Deep Learning for Medical Image Analysis",
            "abstract": "This paper presents novel approaches to medical image analysis using deep learning techniques.",
            "authors": ["Alice Smith", "Bob Johnson"],
            "published_date": "2024-01-15",
            "url": "https://arxiv.org/abs/2401.12345",
            "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
            "primary_category": "cs.CV",
            "secondary_categories": ["cs.LG", "cs.AI"]
        },
        {
            "id": "paper2", 
            "title": "Natural Language Processing in Healthcare",
            "abstract": "Exploring the applications of NLP techniques in healthcare data analysis.",
            "authors": ["Carol Davis", "David Wilson"],
            "published_date": "2024-02-10",
            "url": "https://arxiv.org/abs/2402.67890",
            "pdf_url": "https://arxiv.org/pdf/2402.67890.pdf",
            "primary_category": "cs.CL",
            "secondary_categories": ["cs.AI", "cs.LG"]
        },
        {
            "id": "paper3",
            "title": "Reinforcement Learning for Robotics",
            "abstract": "Application of reinforcement learning algorithms in robotic control systems.",
            "authors": ["Eve Brown", "Frank Miller"],
            "published_date": "2024-03-05",
            "url": "https://arxiv.org/abs/2403.11111",
            "pdf_url": "https://arxiv.org/pdf/2403.11111.pdf", 
            "primary_category": "cs.RO",
            "secondary_categories": ["cs.LG", "cs.AI"]
        }
    ]
    
    return sample_content


def create_rag_system():
    """
    Complete setup and demonstration of the RAG system
    """
    
    # Create sample content data
    sample_data = create_sample_data()
    
    # Create content table
    content_table = pw.debug.table_from_rows(
        schema=ContentSchema,
        rows=[
            (item["id"], item["title"], item["abstract"], item["authors"],
             item["published_date"], item["url"], item["pdf_url"], 
             item["primary_category"], item["secondary_categories"])
            for item in sample_data
        ]
    )
    
    # Set up RAG pipeline
    vector_store, vector_data = setup_dynamic_rag_pipeline(content_table)
    
    # Create sample queries
    query_rows = [
        {"query": "machine learning healthcare", "top_k": 3},
        {"query": "deep learning medical images", "top_k": 2},
        {"query": "robotics reinforcement learning", "top_k": 2}
    ]
    
    query_table = pw.debug.table_from_rows(
        schema=QuerySchema,
        rows=[(row["query"], row["top_k"]) for row in query_rows]
    )
    
    # Query the system
    results = query_rag_pipeline(vector_store, query_table)
    
    # Add output for debugging
    pw.io.jsonlines.write(results, "./query_results.jsonl")
    
    return vector_store, results


class DynamicRAGPipeline:
    """
    Production-ready RAG pipeline class
    """
    
    def __init__(self, embedder_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder_model = embedder_model
        self.vector_store = None
        self.content_table = None
        
    def load_content(self, content_source):
        """
        Load content from various sources (CSV, JSON, etc.)
        """
        if isinstance(content_source, str):
            # Assume it's a file path
            if content_source.endswith('.csv'):
                self.content_table = pw.io.csv.read(
                    content_source,
                    schema=ContentSchema
                )
            elif content_source.endswith('.jsonl'):
                self.content_table = pw.io.jsonlines.read(
                    content_source,
                    schema=ContentSchema
                )
        elif isinstance(content_source, list):
            # Assume it's a list of dictionaries
            self.content_table = pw.debug.table_from_rows(
                schema=ContentSchema,
                rows=[
                    (item["id"], item["title"], item["abstract"], item["authors"],
                     item["published_date"], item["url"], item["pdf_url"], 
                     item["primary_category"], item["secondary_categories"])
                    for item in content_source
                ]
            )
        else:
            raise ValueError("Unsupported content source type")
    
    def setup_vector_store(self):
        """
        Initialize the vector store with loaded content
        """
        if self.content_table is None:
            raise ValueError("Content must be loaded before setting up vector store")
            
        self.vector_store, _ = setup_dynamic_rag_pipeline(self.content_table)
    
    def search(self, query: str, top_k: int = 5):
        """
        Search for similar content
        """
        if self.vector_store is None:
            raise ValueError("Vector store must be set up before searching")
            
        query_table = pw.debug.table_from_rows(
            schema=QuerySchema,
            rows=[(query, top_k)]
        )
        
        return query_rag_pipeline(self.vector_store, query_table)
    
    def run_pipeline(self):
        """
        Execute the pipeline computation
        """
        if self.vector_store is None:
            raise ValueError("Vector store must be set up before running pipeline")
            
        pw.run(monitoring_level=pw.MonitoringLevel.NONE)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Setting up Dynamic RAG Pipeline...")
    
    # Method 1: Simple demonstration
    vector_store, results = create_rag_system()
    
    print("Running pipeline...")
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)
    
    print("RAG Pipeline setup and execution complete!")
    print("Check 'query_results.jsonl' for search results.")
    
    # Method 2: Using the class-based approach (commented out for demo)
    """
    pipeline = DynamicRAGPipeline()
    
    # Load sample data
    sample_data = create_sample_data()
    pipeline.load_content(sample_data)
    
    # Setup vector store
    pipeline.setup_vector_store()
    
    # Search
    results = pipeline.search("machine learning healthcare", top_k=3)
    
    # Run
    pipeline.run_pipeline()
    """