#!/usr/bin/env python3
"""
Test script to verify RAG pipeline integration works correctly.
This simulates the exact workflow the other team will use.
"""

import json
import os
import time
from pathlib import Path

def test_rag_integration():
    """Test the complete RAG integration workflow"""
    
    print("=== RAG Integration Test ===\n")
    
    # Clean up any existing test files
    cleanup_test_files()
    
    # Test 1: Query with no results (empty database)
    print("📝 Test 1: Query empty database")
    query_empty_database()
    
    print("\n" + "="*50)
    
    # Test 2: Add content and query again  
    print("\n� Test 2: Add content and query again")
    add_test_content()
    time.sleep(2)  # Give RAG time to process
    query_with_content()
    
    print("\n✅ All tests completed!")
    print("🔍 Check query_results.jsonl for actual results")

def cleanup_test_files():
    """Remove any existing test files"""
    dirs_to_clean = ["./query_stream", "./content_stream"]
    files_to_clean = ["./query_results.jsonl"]
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                os.remove(os.path.join(directory, file))
    
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)

def query_empty_database():
    """Test querying when database is empty"""
    
    query_data = {
        "query": "quantum computing applications",
        "keywords": ["quantum", "computing", "applications", "algorithms"],
        "top_k": 5
    }
    
    os.makedirs("./query_stream", exist_ok=True)
    query_file = f"./query_stream/test_query_empty_{int(time.time())}.jsonl"
    
    with open(query_file, 'w') as f:
        json.dump(query_data, f)
        f.write('\n')
    
    print(f"🔍 Query sent: {query_data['query']}")
    print(f"📁 Query file: {query_file}")
    print("⏳ Expected result: Empty/no results (database is empty)")

def add_test_content():
    """Add test papers to content stream"""
    
    test_papers = [
        {
            "id": "test_paper_1",
            "title": "Quantum Computing Algorithms for Machine Learning",
            "abstract": "This paper explores the application of quantum computing algorithms to accelerate machine learning tasks, focusing on quantum neural networks and optimization.",
            "authors": ["Dr. Quantum Researcher", "Prof. ML Expert"],
            "published_date": "2024-09-21",
            "url": "https://test-research.com/quantum-ml",
            "pdf_url": "https://test-research.com/quantum-ml.pdf",
            "primary_category": "quant-ph",
            "secondary_categories": ["cs.LG", "cs.AI"],
            "text": [
                "Quantum computing represents a paradigm shift in computational power, offering exponential speedups for certain classes of problems.",
                "In this work, we investigate the application of quantum algorithms to machine learning, specifically focusing on quantum neural networks, quantum support vector machines, and quantum optimization algorithms.",
                "Our experimental results demonstrate significant improvements in training time and model accuracy for specific machine learning tasks."
            ],
            "citations": ["Nielsen & Chuang 2010", "Biamonte et al. 2017"]
        },
        {
            "id": "test_paper_2",
            "title": "Applications of Quantum Algorithms in Optimization",
            "abstract": "A comprehensive study of quantum algorithms for solving complex optimization problems, with applications to logistics, finance, and resource allocation.",
            "authors": ["Dr. Optimization Expert", "Prof. Quantum Scientist"],  
            "published_date": "2024-08-15",
            "url": "https://test-research.com/quantum-opt",
            "pdf_url": "https://test-research.com/quantum-opt.pdf",
            "primary_category": "quant-ph",
            "secondary_categories": ["math.OC", "cs.AI"],
            "text": [
                "Quantum algorithms offer unique advantages for solving NP-hard optimization problems.",
                "This paper presents a comprehensive analysis of quantum approximate optimization algorithms (QAOA), quantum annealing approaches, and hybrid quantum-classical methods.",
                "We demonstrate practical applications in portfolio optimization, vehicle routing problems, and resource scheduling.",
                "The results show promising speedups compared to classical methods, particularly for large-scale optimization instances."
            ],
            "citations": ["Farhi et al. 2014", "Kadowaki & Nishimori 1998"]
        }
    ]
    
    os.makedirs("./content_stream", exist_ok=True)
    
    for i, paper in enumerate(test_papers):
        content_file = f"./content_stream/test_papers_{int(time.time())}_{i}.jsonl"
        with open(content_file, 'w') as f:
            json.dump(paper, f)
            f.write('\n')
        print(f"📄 Added paper: {paper['title']}")
    
    print("✅ Test content added to ./content_stream/")

def query_with_content():
    """Test querying after content has been added"""
    
    query_data = {
        "query": "quantum computing applications", 
        "keywords": ["quantum", "computing", "applications", "algorithms"],
        "top_k": 5
    }
    
    query_file = f"./query_stream/test_query_with_content_{int(time.time())}.jsonl"
    
    with open(query_file, 'w') as f:
        json.dump(query_data, f)
        f.write('\n')
    
    print(f"🔍 Query sent: {query_data['query']}")
    print(f"📁 Query file: {query_file}")
    print("⏳ Expected result: Should find the quantum computing papers")

if __name__ == "__main__":
    test_rag_integration()
