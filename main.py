import keyword_extractor
import scraper
import parser
import enrich_papers
import rag.main

keyword_extractor.run_keyword_extraction("RAG")
scraper.fetch_and_save_arxiv_papers()
parser.parse_and_save_papers()
enrich_papers.main()
(vector_store, answer, documents) = rag.main.main()


print("=== RAG System Output ===\n"*10)
print(f"Vector Store: {vector_store}")
print(f"Answer: {answer}")
print(f"Documents: {documents}")