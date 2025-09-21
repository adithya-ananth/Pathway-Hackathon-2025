import os
import logging
from python_scraper import scrape_arxiv_papers, classify_with_gemini
import pathway as pw
from pathway.io.python import ConnectorSubject

logging.basicConfig(level=logging.INFO)

# A custom Pathway connector to ingest data from our scraper generator
class ArxivScraperSubject(ConnectorSubject):
    def __init__(self, paper_ids: list[str], refresh_interval: int):
        super().__init__()
        self._paper_ids = paper_ids
        self._refresh_interval = refresh_interval

    def run(self):
        for paper in scrape_arxiv_papers(self._paper_ids, self._refresh_interval):
            self.next(**paper) # Send each yielded paper as a row to the pipeline

# Define the schema for the data Pathway will ingest
class ArxivPaperSchema(pw.Schema):
    paper_id: str = pw.column_definition(primary_key=True)
    title: str
    authors: list[str]
    abstract: str
    submitted_date: str

# Wrap the classification function as a Pathway User-Defined Function (UDF)
@pw.udf
def classify_udf(abstract: str) -> str:
    subdomains = ["Computer Vision", "Natural Language Processing", "Machine Learning", "Robotics", "Quantum Computing", "Other"]
    return classify_with_gemini({"abstract": abstract}, subdomains)

if __name__ == "__main__":
    # Simulate a stream of papers
    papers_to_scrape = ["2509.15225v1", "2509.15226v1", "2509.15227v1", "2509.15228v1"]

    # Initialize the custom data connector
    subject = ArxivScraperSubject(
        paper_ids=papers_to_scrape,
        refresh_interval=3600 # Refresh every 30 seconds
    )

    # Ingest the live data stream into a Pathway table
    live_papers = pw.io.python.read(subject, schema=ArxivPaperSchema)

    # Apply the UDF to classify each paper and add a new column
    classified_papers = live_papers.with_columns(
        classification=classify_udf(live_papers.abstract)
    )

    # Write the results to a JSON Lines file in real-time
    pw.io.jsonlines.write(classified_papers, "classified_arxiv_papers.jsonl")

    # Run the Pathway pipeline
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)