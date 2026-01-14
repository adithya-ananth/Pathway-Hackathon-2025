import re
import google.generativeai as genai

from .utils import safe_get_from_doc, safe_convert_to_list


def extract_common_themes(key_findings: list[dict]) -> list[str]:
    """Extract common themes from document findings."""
    word_freq = {}
    for finding in key_findings:
        text = f"{finding['title']} {finding['abstract']}".lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        stopwords = {
            'paper', 'study', 'research', 'analysis', 'using', 'based', 'approach', 'method',
            'results', 'shows', 'demonstrate', 'present', 'propose', 'novel', 'new',
            'this', 'that', 'with', 'for', 'and', 'the', 'are', 'can', 'our',
            'model', 'models', 'data', 'system', 'systems', 'framework', 'algorithms',
            'performance', 'evaluation', 'experiments', 'experimental', 'techniques'
        }
        meaningful_words = [w for w in words if len(w) > 3 and w not in stopwords]
        
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    common_themes = []
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
        if freq > 1:
            common_themes.append(word)
        if len(common_themes) >= 5:
            break
    
    return common_themes


def generate_conclusion(query: str, key_findings: list[dict], keywords: list[str] | None = None) -> str:
    """Generate a conclusion based on the query and findings."""
    conclusion_parts = []
    
    num_docs = len(key_findings)
    categories = set(str(f.get('category', 'Unknown')) for f in key_findings)
    actual_titles = [f.get('title', '') for f in key_findings]
    
    if num_docs >= 3:
        conclusion_parts.append(f"The {num_docs} retrieved documents provide comprehensive coverage of '{query}', ")
        conclusion_parts.append("spanning multiple research perspectives and methodological approaches. ")
    elif num_docs >= 1:
        conclusion_parts.append(f"The {num_docs} relevant document{'s' if num_docs > 1 else ''} ")
        conclusion_parts.append(f"provide{'s' if num_docs == 1 else ''} targeted insights into '{query}', ")
        conclusion_parts.append("though additional sources may enhance understanding. ")
    
    if len(categories) > 1:
        specific_cats = [str(cat) for cat in categories if str(cat) != 'Unknown']
        if specific_cats:
            conclusion_parts.append(f"This research spans {', '.join(sorted(specific_cats))} domains, ")
            conclusion_parts.append("indicating the interdisciplinary nature of the topic. ")
    
    if keywords:
        conclusion_parts.append(f"The focus on {', '.join(keywords)} appears well-supported ")
        conclusion_parts.append("by the current literature, with documents directly addressing these concepts. ")
    
    if actual_titles and any(title.strip() for title in actual_titles):
        first_meaningful_title = next((title for title in actual_titles if len(title.strip()) > 10), None)
        if first_meaningful_title:
            conclusion_parts.append(f"Key research includes work on \"{first_meaningful_title[:60]}{'...' if len(first_meaningful_title) > 60 else ''}\", ")
            conclusion_parts.append("demonstrating active development in this area. ")
    
    return ''.join(conclusion_parts)


def suggest_related_keywords(original_keywords: list[str], matched_keywords: set) -> str:
    """Suggest related keywords for further exploration."""
    suggestions = set()
    
    for keyword in original_keywords:
        keyword_lower = keyword.lower()
        if any(term in keyword_lower for term in ['quantum', 'quant']):
            suggestions.update(['quantum computing', 'quantum algorithms', 'quantum mechanics', 'NISQ'])
        elif any(term in keyword_lower for term in ['neural', 'neuron', 'network']):
            suggestions.update(['deep learning', 'transformers', 'attention mechanisms', 'CNN'])
        elif any(term in keyword_lower for term in ['machine learning', 'ml', 'ai', 'artificial']):
            suggestions.update(['deep learning', 'reinforcement learning', 'supervised learning', 'MLOps'])
        elif any(term in keyword_lower for term in ['adversarial', 'attack', 'security']):
            suggestions.update(['robustness', 'defense mechanisms', 'cybersecurity', 'threat detection'])
        elif any(term in keyword_lower for term in ['natural language', 'nlp', 'text']):
            suggestions.update(['transformers', 'BERT', 'GPT', 'language models'])
        elif any(term in keyword_lower for term in ['computer vision', 'cv', 'image']):
            suggestions.update(['object detection', 'image classification', 'CNN', 'segmentation'])
    
    good_matches = [kw for kw in matched_keywords if len(kw) > 3 and kw not in {'and', 'the', 'for', 'with'}]
    suggestions.update(good_matches[:3])
    
    original_lower = set(kw.lower() for kw in original_keywords)
    suggestions = suggestions - original_lower
    
    return ', '.join(list(suggestions)[:5]) if suggestions else 'related terms from the literature'


def generate_answer_with_context(query: str, search_results: list, keywords: list[str] | None = None, max_context_length: int = 4000):
    """Generate a comprehensive answer based on retrieved documents using Gemini LLM."""
    if not search_results:
        if keywords:
            return f"No relevant documents found for query: '{query}' with keywords: {', '.join(keywords)}. Consider broadening your search terms or searching external sources."
        return f"No relevant documents found for query: '{query}'. Consider using different search terms or searching external sources."
    
    all_matched_keywords = set()
    primary_categories = set()
    authors = set()
    key_findings = []
    document_summaries = []
    
    for i, doc in enumerate(search_results[:5], 1):
        matched_kws = safe_get_from_doc(doc, 'matched_keywords', [])
        matched_kws = safe_convert_to_list(matched_kws)
        all_matched_keywords.update(matched_kws)
        
        category = safe_get_from_doc(doc, 'primary_category', 'Unknown')
        if category:
            primary_categories.add(str(category))
        
        doc_authors = safe_get_from_doc(doc, 'authors', [])
        doc_authors = safe_convert_to_list(doc_authors)
        if doc_authors:
            authors.update(str(author) for author in doc_authors[:2])
        
        title = safe_get_from_doc(doc, 'title', 'Untitled')
        abstract = safe_get_from_doc(doc, 'abstract', 'No abstract available')
        score = safe_get_from_doc(doc, 'similarity_score', 0.0)
        
        title = str(title) if title else 'Untitled'
        abstract = str(abstract) if abstract else 'No abstract available'
        score = float(score) if score is not None else 0.0
        
        if len(abstract) > 300:
            abstract = abstract[:297] + "..."
        
        doc_summary = f"""
Document {i}: {title}
Relevance Score: {score:.3f}
Abstract: {abstract}
Matched Terms: {', '.join(str(kw) for kw in matched_kws) if matched_kws else 'None'}"""
        
        document_summaries.append(doc_summary)
        
        key_findings.append({
            'title': title,
            'abstract': abstract,
            'keywords': matched_kws,
            'category': category
        })
    
    answer_parts = []
    
    answer_parts.append(f"## Answer to: {query}\n")
    
    answer_parts.append("### Executive Summary")
    if keywords:
        answer_parts.append(f"Based on analysis of {len(search_results)} relevant documents related to your query about {query}, with focus on: {', '.join(keywords)}.\n")
    else:
        answer_parts.append(f"Based on analysis of {len(search_results)} relevant documents related to your query about {query}.\n")
    
    answer_parts.append("### Key Insights")
    
    if len(primary_categories) > 1:
        safe_categories = [str(cat) for cat in primary_categories if cat]
        answer_parts.append(f"This is an interdisciplinary topic spanning {', '.join(sorted(safe_categories))} domains.")
    else:
        first_category = str(list(primary_categories)[0]) if primary_categories else "Unknown"
        answer_parts.append(f"This research primarily falls within the {first_category} domain.")
    
    if all_matched_keywords:
        safe_keywords = [str(kw) for kw in all_matched_keywords if kw]
        if safe_keywords:
            answer_parts.append(f"\nThe most relevant aspects identified include: {', '.join(sorted(safe_keywords))}.")
    
    common_themes = extract_common_themes(key_findings)
    if common_themes:
        answer_parts.append(f"\nCommon themes across the research include: {', '.join(common_themes)}.")
    
    if authors:
        notable_authors = list(authors)[:5]
        answer_parts.append(f"\nNotable researchers in this area include: {', '.join(notable_authors)}.")
    
    answer_parts.append("\n### Supporting Documents")
    
    current_length = len('\n'.join(answer_parts))
    for doc_summary in document_summaries:
        if current_length + len(doc_summary) <= max_context_length - 500:
            answer_parts.append(doc_summary)
            current_length += len(doc_summary)
        else:
            remaining_docs = len(document_summaries) - document_summaries.index(doc_summary)
            answer_parts.append(f"\n... and {remaining_docs} additional relevant documents")
            break
    
    answer_parts.append("\n### Conclusion")
    answer_parts.append(generate_conclusion(query, key_findings, keywords))
    
    answer_parts.append("\n### For Further Research")
    answer_parts.append("Consider exploring the full text of the most relevant documents above, ")
    answer_parts.append("particularly those with the highest relevance scores. ")
    if keywords:
        safe_matched_keywords = set(str(kw) for kw in all_matched_keywords if kw)
        answer_parts.append(f"You may also want to search for related terms such as: {suggest_related_keywords(keywords, safe_matched_keywords)}.")
    
    full_answer = '\n'.join(answer_parts)

    if key_findings:
        docs_text = "\n\n".join([
            f"Title: {doc['title']}\nAbstract: {doc['abstract']}"
            for doc in key_findings
        ])

        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
You are an expert research assistant.
The user query is: "{query}"

I will give you several research papers with their titles, and abstracts.

Use this information as reference to answer the query. 
Write a clear, factual answer.
Then list the titles of the papers you used under '### Sources'.

PAPERS:
{docs_text}
"""
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print("Gemini summarization failed:", e)
            return full_answer

    return full_answer
