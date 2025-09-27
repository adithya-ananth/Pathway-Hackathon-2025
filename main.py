import keyword_extractor
import scraper
import parser
import enrich_papers
import rag.main


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# CORS: permissive configuration (allow all origins, methods, and headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

class PromptRequest(BaseModel):
    prompt: str


@app.post("/prompt")
def query_rag_system(request: PromptRequest):
    keyword_extractor.run_keyword_extraction(request.prompt)
    scraper.fetch_and_save_arxiv_papers()
    parser.parse_and_save_papers()
    enrich_papers.main()
    (vector_store, answer, documents) = rag.main.main()
    return {
        "message": answer,
        "papers": documents
    }
