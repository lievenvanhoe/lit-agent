import json
import time
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

PUBMED_QUERIES = [
    "radiology report generation large language model",
    "automated radiology reporting transformer",
    "multimodal alignment radiology vision language model",
    "retrieval augmented generation radiology clinical",
    "hallucination radiology report large language model",
    "instruction fine-tuning radiology language model",
    "radiologist AI report validation",
]

ARXIV_QUERIES = [
    "radiology report generation LLM",
    "automated radiology report language model",
    "multimodal radiology vision language",
    "retrieval augmented radiology report",
    "hallucination medical report generation",
]

DAYS_BACK = 14
MAX_RESULTS = 15
DELAY = 1.0


def pubmed_search(query):
    date_from = (datetime.date.today() - datetime.timedelta(days=DAYS_BACK)).strftime("%Y/%m/%d")
    date_to = datetime.date.today().strftime("%Y/%m/%d")
    search_params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query + " AND (" + date_from + "[PDAT]:" + date_to + "[PDAT])",
        "retmax": MAX_RESULTS,
        "retmode": "json",
        "sort": "date",
    })
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + search_params
    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0 (research bot; mailto:research@example.com)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            search_data = json.loads(resp.read())
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        time.sleep(DELAY)
        fetch_params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        })
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + fetch_params
        req2 = urllib.request.Request(fetch_url, headers={"User-Agent": "Mozilla/5.0 (research bot; mailto:research@example.com)"})
        with urllib.request.urlopen(req2, timeout=15) as resp:
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
                    last = a.findtext("LastName") or ""
                    first = a.findtext("ForeName") or ""
                    if last:
                        authors.append(last + " " + first)
                if len(authors_els) > 3:
                    authors.append("et al.")
                papers.append({
                    "source": "PubMed",
                    "title": title,
                    "authors": ", ".join(authors),
                    "abstract": abstract[:1500],
                    "url": "https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/",
                })
            except Exception:
                continue
        return papers
    except Exception as e:
        print("PubMed error for '" + query[:40] + "': " + str(e))
        return []


def arxiv_search(query):
    params = urllib.parse.urlencode({
        "search_query": "all:" + query,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = "https://export.arxiv.org/api/query?" + params
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research bot)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_data = resp.read()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_data)
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=DAYS_BACK)
        papers = []
        for entry in root.findall("atom:entry", ns):
            try:
                published_str = entry.findtext("atom:published", namespaces=ns) or ""
                published = datetime.datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                if published < cutoff:
                    continue
                title = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
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
    except Exception as e:
        print("arXiv error for '" + query[:40] + "': " + str(e))
        return []


def deduplicate(papers):
    seen = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def fetch_all_papers():
    papers = []
    print("Fetching from PubMed...")
    for q in PUBMED_QUERIES:
        results = pubmed_search(q)
        print("  [" + q[:50] + "] -> " + str(len(results)) + " papers")
        papers.extend(results)
        time.sleep(DELAY)
    print("Fetching from arXiv...")
    for q in ARXIV_QUERIES:
        results = arxiv_search(q)
        print("  [" + q[:50] + "] -> " + str(len(results)) + " papers")
        papers.extend(results)
        time.sleep(2.0)
    papers = deduplicate(papers)
    print("Total unique papers: " + str(len(papers)))
    return papers
