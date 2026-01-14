import os


def resolve_project_path(relative_or_abs_path: str) -> str:
    """Resolve relative file paths like 'papers_text/xyz.txt' relative to repo root."""
    if relative_or_abs_path is None:
        raise ValueError("file_path is None; expected a path to a .txt file")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return (
        relative_or_abs_path
        if os.path.isabs(relative_or_abs_path)
        else os.path.join(base_dir, relative_or_abs_path)
    )


def read_text_from_file(file_path: str) -> str:
    """Read UTF-8 text from a local .txt file path (absolute or project-relative)."""
    resolved_path = resolve_project_path(file_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read()


def safe_get_from_doc(doc, key: str, default=None):
    """Safely extract a value from a document, handling both dict and Pathway Json objects."""
    try:
        if hasattr(doc, 'get'):
            return doc.get(key, default)
        elif hasattr(doc, key):
            return getattr(doc, key)
        elif hasattr(doc, '__getitem__'):
            try:
                return doc[key]
            except (KeyError, TypeError):
                return default
        else:
            return default
    except Exception:
        return default


def safe_convert_to_list(val):
    """Safely convert a value to a list, handling different input types."""
    try:
        if val is None:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, (str, int, float)):
            return [val]
        try:
            return list(val)
        except (TypeError, AttributeError):
            return [val] if val is not None else []
    except Exception:
        return []
