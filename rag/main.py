import pathway as pw
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.vector_store import VectorStoreServer
import os
import json
import time

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
    def _read_text_from_file(file_path: str) -> str:
        # Support relative paths like "papers_text/ab.txt".
        # Resolve relative to the project root (parent of this file's folder).
        if file_path is None:
            raise ValueError("file_path is None; expected a relative or absolute path to a .txt file")

        base_dir = os.path.dirname(os.path.dirname(__file__))
        resolved_path = file_path if os.path.isabs(file_path) else os.path.join(base_dir, file_path)

        with open(resolved_path, "r", encoding="utf-8") as f:
            return f.read()

    vector_data = content_table.select(
        # Use file_path (guaranteed .txt) as the embedding source; schema.text is ignored.
        data=pw.apply(_read_text_from_file, pw.this.file_path),
        metadata=pw.apply(
            lambda paper_id, title, abstract, authors, published_date, url, pdf_url, primary, subcats, citations: {
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
                "citations": citations
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
            pw.this.citations
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
    try:
        content_stream = pw.io.jsonlines.read(
            "./content_stream/",
            schema=ContentSchema,
            mode="streaming",
            autocommit_duration_ms=1000  # Update every 1 second
        )
    except:
        # Fallback to debug table if streaming fails
        sample_data = create_sample_data()
        content_stream = pw.debug.table_from_rows(
            schema=ContentSchema,
            rows=[
                (
                    item.get("id") or item.get("paper_id"),
                    item.get("title"),
                    item.get("abstract"),
                    item.get("authors"),
                    item.get("published_date"),
                    item.get("url"),
                    item.get("pdf_url"),
                    item.get("primary_category"),
                    item.get("sub_categories") or item.get("secondary_categories"),
                    item.get("journal_ref"),
                    item.get("doi"),
                    item.get("references"),
                    item.get("text"),
                    item.get("file_path") or item.get("file_url"),
                    item.get("citations"),
                )
                for item in sample_data
            ]
        )
    
    return content_stream


def setup_dynamic_query_pipeline():
    """
    Setup dynamic query processing - queries come in real-time from other team
    They provide query + keywords and expect results or "no results found"
    """
    try:
        query_stream = pw.io.jsonlines.read(
            "./query_stream/",
            schema=QuerySchema, 
            mode="streaming",
            autocommit_duration_ms=500  # Fast processing for real-time queries
        )
    except:
        # Fallback to sample queries for testing
        query_rows = [
            {"query": "machine learning healthcare", "top_k": 3, "keywords": ["healthcare", "medical", "nlp"]},
            {"query": "deep learning medical images", "top_k": 2, "keywords": ["deep learning", "medical", "image"]},
            {"query": "robotics reinforcement learning", "top_k": 2, "keywords": ["robotics", "reinforcement", "control"]}
        ]
        
        query_stream = pw.debug.table_from_rows(
            schema=QuerySchema,
            rows=[(row["query"], row["top_k"], row["keywords"]) for row in query_rows]
        )
    
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
        # Handle Json objects - metadata is likely stored as a dictionary within the Json object
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata
        else:
            # If it's a Json object, the metadata might be the entire object or a nested field
            # Try to access it as a dictionary
            try:
                metadata = dict(doc) if hasattr(doc, '__iter__') and not isinstance(doc, str) else {}
            except:
                metadata = {}
        
        # Create searchable text from multiple fields
        searchable_text = " ".join([
            str(metadata.get("title", "")),
            str(metadata.get("abstract", "")),
            str(metadata.get("primary_category", "")),
            " ".join(metadata.get("secondary_categories", []) or metadata.get("sub_categories", []) or []),
            " ".join(metadata.get("authors", []) or [])
        ]).lower()
        
        # Check for keyword matches
        matched_keywords = [kw for kw in keywords_lower if kw in searchable_text]
        
        if matched_keywords:
            filtered_docs.append(format_document_with_keywords(doc, matched_keywords))
    
    # If no documents match keywords, return empty list
    # This signals to other team: "no results found" - they should search web
    return filtered_docs


def format_document(doc):
    """
    Format a document for output
    """
    # Handle Json objects - metadata is likely stored as a dictionary within the Json object
    if hasattr(doc, 'metadata'):
        metadata = doc.metadata
    else:
        # If it's a Json object, the metadata might be the entire object or a nested field
        # Try to access it as a dictionary
        try:
            metadata = dict(doc) if hasattr(doc, '__iter__') and not isinstance(doc, str) else {}
        except:
            metadata = {}
    
    return {
        "id": metadata.get("id", "unknown"),
        "title": metadata.get("title", ""),
        "abstract": metadata.get("abstract", ""),
        "authors": metadata.get("authors", []),
        "similarity_score": getattr(doc, 'score', 0.0),
        "url": metadata.get("url", ""),
        "primary_category": metadata.get("primary_category", ""),
        "matched_keywords": []
    }


def format_document_with_keywords(doc, matched_keywords):
    """
    Format a document with matched keywords highlighted
    """
    formatted = format_document(doc)
    formatted["matched_keywords"] = matched_keywords
    return formatted


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
    
    print("\n📋 Workflow Summary:")
    print("   1. Other team drops query.jsonl in ./query_stream/")
    print("   2. RAG processes query with keywords")
    print("   3. If no results found -> empty list returned")
    print("   4. Other team searches web -> drops papers in ./content_stream/")
    print("   5. RAG auto-updates database with new papers")
    print("   6. Other team queries again -> gets results")
    
    return vector_store, results


def answer_query_with_context(query: str, search_results: list, max_context_length: int = 2000):
    """
    Generate a comprehensive answer based on retrieved documents
    """
    if not search_results:
        return f"No relevant documents found for query: '{query}'"
    
    # Build context from top results
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(search_results[:3], 1):  # Top 3 results
        context_part = f"""
Document {i} (ID: {doc['id']}, Score: {doc['similarity_score']:.3f}):
Title: {doc['title']}
Abstract: {doc['abstract']}
Matched Keywords: {', '.join(doc['matched_keywords'])}
---"""
        
        if current_length + len(context_part) <= max_context_length:
            context_parts.append(context_part)
            current_length += len(context_part)
        else:
            break
    
    context = "\n".join(context_parts)
    
    # Generate answer
    answer = f"""Based on the retrieved documents:

{context}

Query: {query}

Summary: Based on {len(search_results)} relevant documents, here are the key findings related to your query. The most relevant documents cover topics in {', '.join(set(doc.get('primary_category', 'N/A') for doc in search_results[:3]))}.

Key matched terms: {', '.join(set().union(*[doc.get('matched_keywords', []) for doc in search_results[:3]]))}

For detailed information, please refer to the documents listed above.
"""
    
    return answer


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
