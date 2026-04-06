"""
Literature monitoring agent
Sources: PubMed + arXiv
Topic: Radiology report generation + LLMs
"""

import os
import json
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


# ── Configuration ──────────────────────────────────────────────────────────────

PUBMED_QUERIES = [
    # Report generation — multimodal LLM (core)
    "radiology report generation large language model",
    "radiology LLM report automation",
    "automated radiology reporting transformer",

    # Vision perception layer
    "vision transformer radiology image encoding",
    "medical image feature extraction vision encoder",

    # Multimodal integration & alignment
    "multimodal alignment radiology vision language model",
    "cross-modal fusion medical imaging report",

    # Clinical context / EMR / RAG
    "retrieval augmented generation radiology clinical",
    "electronic medical record radiology NLP context",

    # Reconciliation — factual grounding / hallucination
    "hallucination radiology report large language model",
    "factual grounding medical report generation",

    # RAG retrieval (guidelines, similar cases, templates)
    "guideline retrieval radiology report generation",
    "similar case retrieval radiology AI",

    # Instruction fine-tuning
    "instruction fine-tuning radiology language model",
    "prompt engineering radiology report LLM",

    # Reinforcement learning feedback loop
    "reinforcement learning radiology report generation",
    "RLHF medical report generation",

    # Human supervision & validation
    "radiologist-in-the-loop AI report validation",
    "human evaluation automated radiology report",
]

ARXIV_QUERIES = [
    # Report generation — multimodal LLM (core)
    "radiology report generation LLM",
    "automated radiology report large language model",

    # Vision perception
    "vision transformer medical image encoder radiology",

    # Multimodal integration
    "multimodal alignment vision language radiology",

    # RAG + clinical context
    "retrieval augmented generation radiology report",
    "clinical context radiology NLP EMR",

    # Hallucination / factual grounding
    "hallucination medical report LLM factual",

    # Fine-tuning + RL
    "instruction tuning radiology report model",
    "reinforcement learning from human feedback radiology",

    # Human-in-the-loop
    "human supervision radiology AI report validation",
]

DAYS_BACK = 1          # how many days to look back
MAX_RESULTS = 20       # max papers per query


# ── PubMed ─────────────────────────────────────────────────────────────────────

def pubmed_search(query: str, days_back: int = DAYS_BACK) -> list[dict]:
    """Search PubMed and return list of paper dicts."""
    date_from = (datetime.date.today() - datetime.timedelta(days=days_back)).strftime("%Y/%m/%d")
    date_to   = datetime.date.today().strftime("%Y/%m/%d")

    # Step 1: eSearch — get PMIDs
    search_params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": f"{query} AND ({date_from}[PDAT]:{date_to}[PDAT])",
        "retmax": MAX_RESULTS,
        "retmode": "json",
        "sort": "date",
    })
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}"

    with urllib.request.urlopen(search_url, timeout=15) as resp:
        search_data = json.loads(resp.read())

    pmids = search_data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    # Step 2: eFetch — get abstracts
    fetch_params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    })
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{fetch_params}"

    with urllib.request.urlopen(fetch_url, timeout=15) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        try:
            title = article.findtext(".//ArticleTitle") or "No title"
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(p.text or "" for p in abstract_parts if p.text)
            pmid = article.findtext(".//PMID") or ""
            authors_els = article.findall(".//Author")
            authors = []
            for a in authors_els[:3]:
                last  = a.findtext("LastName") or ""
                first = a.findtext("ForeName") or ""
                if last:
                    authors.append(f"{last} {first}".strip())
            if len(authors_els) > 3:
                authors.append("et al.")

            papers.append({
                "source": "PubMed",
                "title": title,
                "authors": ", ".join(authors),
                "abstract": abstract[:1500],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "pmid": pmid,
            })
        except Exception:
            continue

    return papers


# ── arXiv ──────────────────────────────────────────────────────────────────────

def arxiv_search(query: str, days_back: int = DAYS_BACK) -> list[dict]:
    """Search arXiv and return list of paper dicts."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"

    with urllib.request.urlopen(url, timeout=15) as resp:
        xml_data = resp.read()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)
    papers = []

    for entry in root.findall("atom:entry", ns):
        try:
            published_str = entry.findtext("atom:published", namespaces=ns) or ""
            published = datetime.datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            if published < cutoff:
                continue

            title   = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip().replace("\n", " ")
            link_el = entry.find("atom:id", ns)
            url_val = link_el.text.strip() if link_el is not None and link_el.text else ""

            authors = []
            for a in entry.findall("atom:author", ns)[:3]:
                name = a.findtext("atom:name", namespaces=ns)
                if name:
                    authors.append(name)
            if len(entry.findall("atom:author", ns)) > 3:
                authors.append("et al.")

            papers.append({
                "source": "arXiv",
                "title": title,
                "authors": ", ".join(authors),
                "abstract": summary[:1500],
                "url": url_val,
            })
        except Exception:
            continue

    return papers


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by normalised title."""
    seen = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ── Main fetch ─────────────────────────────────────────────────────────────────

def fetch_all_papers() -> list[dict]:
    papers = []

    print("→ Fetching from PubMed...")
    for q in PUBMED_QUERIES:
        try:
            results = pubmed_search(q)
            print(f"  [{q[:50]}] → {len(results)} papers")
            papers.extend(results)
        except Exception as e:
            print(f"  PubMed error for '{q}': {e}")

    print("→ Fetching from arXiv...")
    for q in ARXIV_QUERIES:
        try:
            results = arxiv_search(q)
            print(f"  [{q[:50]}] → {len(results)} papers")
            papers.extend(results)
        except Exception as e:
            print(f"  arXiv error for '{q}': {e}")

    papers = deduplicate(papers)
    print(f"→ Total unique papers: {len(papers)}")
    return papers


if __name__ == "__main__":
    papers = fetch_all_papers()
    for p in papers:
        print(f"\n[{p['source']}] {p['title']}")
        print(f"  {p['authors']}")
        print(f"  {p['url']}")
