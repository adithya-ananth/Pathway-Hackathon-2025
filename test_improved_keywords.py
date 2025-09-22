#!/usr/bin/env python3
"""
Test the improved keyword extraction and RAG system
"""

import sys
import os
sys.path.append('.')

from external_team_example import generate_keywords_from_query, call_rag_pipeline

def test_improved_system():
    """Test the system with improved keyword extraction"""
    
    # Test query about wave-particle duality
    query = "Find me examples of particles having wave nature"
    print(f"\n🔍 Testing query: '{query}'")
    
    # Generate improved keywords
    keywords = generate_keywords_from_query(query)
    print(f"📝 Generated keywords: {keywords}")
    
    # Test the RAG system
    results = call_rag_pipeline(query, keywords, top_k=3)
    
    print(f"\n✅ Results received: {len(results.get('results', []))} documents")
    for i, doc in enumerate(results.get('results', [])):
        print(f"  Document {i+1}:")
        print(f"    Title: {doc.get('title', 'Unknown')}")
        print(f"    Score: {doc.get('similarity_score', 0):.3f}")
        print(f"    Keywords: {doc.get('matched_keywords', [])}")

if __name__ == "__main__":
    test_improved_system()