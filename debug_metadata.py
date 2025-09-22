#!/usr/bin/env python3
"""
Debug script to investigate metadata extraction issues
"""

import json
from pathlib import Path

def analyze_enriched_papers():
    """Analyze the structure of enriched papers to understand metadata format"""
    file_path = Path("content_stream/enriched_papers.jsonl")
    
    if not file_path.exists():
        print("❌ enriched_papers.jsonl not found")
        return
    
    print("📄 Analyzing enriched_papers.jsonl structure...")
    
    with open(file_path, 'r') as f:
        lines = list(f)
        
    print(f"📊 Total papers: {len(lines)}")
    
    # Analyze first few papers
    for i, line in enumerate(lines[:3]):
        paper = json.loads(line)
        print(f"\n=== Paper {i+1} ===")
        print(f"Keys: {list(paper.keys())}")
        
        for key, value in paper.items():
            if key == 'text':
                print(f"{key}: {type(value)} - {len(value)} characters")
                print(f"  Text preview: {str(value)[:100]}...")
            else:
                print(f"{key}: {type(value)} - {value}")

def analyze_vector_store_results():
    """Check if there are any vector store result files"""
    
    # Look for any result files
    for pattern in ["query_results.jsonl", "*.jsonl"]:
        files = list(Path(".").glob(pattern))
        for file in files:
            if "query" in file.name or "result" in file.name:
                print(f"\n📄 Found result file: {file}")
                try:
                    with open(file, 'r') as f:
                        lines = list(f)
                    print(f"📊 Lines: {len(lines)}")
                    
                    if lines:
                        result = json.loads(lines[-1])  # Latest result
                        print(f"Latest result keys: {list(result.keys())}")
                        
                        if 'results' in result:
                            results = result['results']
                            print(f"Results count: {len(results)}")
                            if results:
                                first = results[0]
                                print(f"First result type: {type(first)}")
                                print(f"First result: {first}")
                                
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")

if __name__ == "__main__":
    print("🔧 Debug: Metadata Extraction Analysis")
    analyze_enriched_papers()
    analyze_vector_store_results()