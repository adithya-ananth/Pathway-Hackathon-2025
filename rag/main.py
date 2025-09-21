import pathway as pw

class InputSchema(pw.Schema):
    id: str
    title: str
    abstract: str
    authors: list[str]
    published_date: str
    url: str
    pdf_url: str
    primary_category: str
    secondary_categories: list[str]