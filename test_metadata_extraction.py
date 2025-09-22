#!/usr/bin/env python3
"""
Test the metadata extraction directly using the vector cache data
"""

import json
import sys
from pathlib import Path

# Add the current directory to Python path to import rag modules
sys.path.insert(0, str(Path.cwd()))

def test_metadata_extraction():
    """Test metadata extraction using cached vector data"""
    
    # First, read some vector data from the cache
    cache_file = Path(".vector_data_cache.jsonl")
    if not cache_file.exists():
        print("❌ Vector cache file not found")
        return
    
    print("📄 Reading vector cache data...")
    with open(cache_file, 'r') as f:
        lines = list(f)
    
    if not lines:
        print("❌ No data in vector cache")
        return
        
    print(f"📊 Found {len(lines)} cached documents")
    
    # Test with first document
    first_doc = json.loads(lines[0])
    print(f"\n=== Testing with first document ===")
    print(f"Document keys: {list(first_doc.keys())}")
    print(f"Title: {first_doc.get('title', 'N/A')}")
    print(f"Abstract preview: {str(first_doc.get('abstract', 'N/A'))[:100]}...")
    
    # Simulate what the VectorStore would return
    # Based on debug output, VectorStore returns {'text': '...', 'metadata': {...}, 'dist': 0.xx}
    simulated_result = {
        'text': first_doc.get('data', ''),
        'metadata': {
            'id': first_doc.get('doc_id'),
            'title': first_doc.get('title'),
            'abstract': first_doc.get('abstract'),
            'authors': first_doc.get('authors'),
            'url': first_doc.get('url'),
            'primary_category': first_doc.get('primary_category'),
            'file_path': 'papers_text/' + first_doc.get('doc_id', '') + '.txt'
        },
        'dist': 0.15  # Sample distance
    }
    
    print(f"\n=== Simulated VectorStore result ===")
    print(f"Keys: {list(simulated_result.keys())}")
    print(f"Metadata keys: {list(simulated_result['metadata'].keys())}")
    
    # Now test our metadata extraction function
    try:
        from rag.main import _extract_metadata_from_result, format_document
        
        print(f"\n=== Testing metadata extraction ===")
        
        # Create a mock Pathway Json object
        class MockPathwayJson:
            def __init__(self, value):
                self.value = value
        
        mock_doc = MockPathwayJson(simulated_result)
        
        extracted_metadata = _extract_metadata_from_result(mock_doc)
        print(f"Extracted metadata keys: {list(extracted_metadata.keys())}")
        print(f"Title: {extracted_metadata.get('title', 'NOT FOUND')}")
        print(f"Abstract: {str(extracted_metadata.get('abstract', 'NOT FOUND'))[:100]}...")
        
        # Test full document formatting
        print(f"\n=== Testing document formatting ===")
        formatted_doc = format_document(mock_doc)
        print(f"Formatted document keys: {list(formatted_doc.keys())}")
        print(f"Formatted title: {formatted_doc.get('title', 'NOT FOUND')}")
        print(f"Similarity score: {formatted_doc.get('similarity_score', 'NOT FOUND')}")
        
        if formatted_doc.get('title') and formatted_doc.get('title') != '':
            print("✅ SUCCESS: Metadata extraction working!")
        else:
            print("❌ FAILED: Title still empty after extraction")
            
    except Exception as e:
        print(f"❌ Error testing extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metadata_extraction()