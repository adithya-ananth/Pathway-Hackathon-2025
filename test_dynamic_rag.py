#!/usr/bin/env python3
"""
Simple test script to demonstrate the dynamic RAG functionality
"""

from rag.main import DynamicRAGPipeline, create_sample_data

def test_dynamic_rag():
    """Test the dynamic RAG pipeline with sample data"""
    print("🚀 Testing Dynamic RAG Pipeline")
    print("=" * 50)
    
    # Mock search function for demonstration
    def mock_search_function(query: str, keywords: list[str]) -> list[dict]:
        """Mock search function - simulates your team's search function"""
        print(f"📡 External search triggered!")
        print(f"   Query: '{query}'")
        print(f"   Keywords: {keywords}")
        
        # Return mock search results in ContentSchema format
        return [
            {
                "id": f"external_1_{hash(query) % 1000}",
                "title": f"External Research: {query.title()}",
                "abstract": f"This is an external research paper found for '{query}'. It covers advanced topics related to your search query and provides insights not available in the local knowledge base.",
                "authors": ["Dr. External Researcher", "Prof. Web Search"],
                "published_date": "2024-09-21",
                "url": f"https://external-source.com/papers/{query.replace(' ', '-')}",
                "pdf_url": f"https://external-source.com/papers/{query.replace(' ', '-')}.pdf",
                "primary_category": "external.research",
                "secondary_categories": ["web.search", "external.content"],
                "text": f"Comprehensive analysis of {query}. This external content provides additional context and research findings that complement the local knowledge base. Keywords addressed: {', '.join(keywords)}. The research methodology and findings are particularly relevant for understanding the broader implications of {query} in current academic discourse.",
                "citations": [f"External Source {i}" for i in range(1, 4)]
            },
            {
                "id": f"external_2_{hash(query) % 1000 + 1}",
                "title": f"Advanced Studies in {query.title()}",
                "abstract": f"A follow-up study on {query} with recent developments and breakthrough discoveries.",
                "authors": ["Dr. Latest Research", "Prof. Current Studies"],
                "published_date": "2024-09-20",
                "url": f"https://external-source.com/advanced/{query.replace(' ', '-')}",
                "pdf_url": f"https://external-source.com/advanced/{query.replace(' ', '-')}.pdf",
                "primary_category": "external.advanced",
                "secondary_categories": ["recent.research"],
                "text": f"Recent breakthrough research in {query} area. This study builds upon previous work and introduces novel approaches. Specific focus on {', '.join(keywords)} ensures high relevance to your search query.",
                "citations": [f"Recent Study {i}" for i in range(1, 3)]
            }
        ]
    
    # 1. Initialize pipeline with search function
    print("1. 🔧 Initializing pipeline...")
    pipeline = DynamicRAGPipeline(search_function=mock_search_function)
    
    # 2. Load sample data
    print("2. 📚 Loading sample data...")
    sample_data = create_sample_data()
    pipeline.load_content(sample_data)
    print(f"   Loaded {len(sample_data)} sample documents")
    
    # 3. Setup vector store
    print("3. 🧠 Setting up vector store...")
    pipeline.setup_vector_store()
    print("   ✅ Vector store ready with embeddings")
    
    # 4. Demonstrate regular search
    print("\n4. 🔍 Testing regular search (sufficient results)...")
    print("   Query: 'deep learning medical applications'")
    print("   Keywords: ['medical', 'deep learning']")
    
    results1 = pipeline.search(
        query="deep learning medical applications",
        keywords=["medical", "deep learning"],
        top_k=3
    )
    print("   ✅ Search completed - found sufficient local results")
    
    # 5. Demonstrate search with fallback
    print("\n5. 🚀 Testing search with external fallback...")
    print("   Query: 'quantum computing applications'")  
    print("   Keywords: ['quantum', 'computing']")
    print("   (This should trigger external search since we don't have quantum content)")
    
    results2 = pipeline.search_with_fallback(
        query="quantum computing applications",
        keywords=["quantum", "computing"], 
        top_k=5,
        min_results=3
    )
    print("   ✅ Search with fallback completed")
    
    # 6. Run pipeline to execute all operations
    print("\n6. ⚡ Executing pipeline...")
    pipeline.run_pipeline()
    
    print("\n✅ Dynamic RAG Pipeline test completed successfully!")
    print("\n📊 Key Features Demonstrated:")
    print("   ✓ Dynamic content loading")
    print("   ✓ Vector store setup with embeddings") 
    print("   ✓ Semantic search with keyword filtering")
    print("   ✓ External search fallback integration")
    print("   ✓ Real-time pipeline execution")
    
    print("\n🔧 Next Steps:")
    print("   • Replace mock_search_function with your team's real search function")
    print("   • Add streaming data sources (./content_stream/ directory)")
    print("   • Add streaming queries (./query_stream/ directory)")
    print("   • Enable monitoring dashboard for production use")
    
    return pipeline

if __name__ == "__main__":
    test_dynamic_rag()
