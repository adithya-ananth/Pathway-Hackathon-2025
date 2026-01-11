import pathway as pw


class ContentSchema(pw.Schema):
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
