import os
import json
import urllib.request

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"

RELEVANCE_SYSTEM = (
    "You are a strict relevance filter for a radiology report generation (RRG) research digest.\n\n"
    "A paper is ONLY relevant if it directly addresses one of these:\n"
    "1. Generating radiology reports from medical images using AI or LLMs\n"
    "2. LLMs or transformers applied specifically to radiology report text\n"
    "3. Multimodal models that take radiology images as input AND produce report text as output\n"
    "4. RAG systems specifically for retrieving radiology guidelines or similar cases to aid report generation\n"
    "5. Hallucination or factual grounding specifically in radiology report generation\n"
    "6. Fine-tuning or RLHF specifically for radiology report generation models\n"
    "7. Human-in-the-loop validation of AI-generated radiology reports\n\n"
    "REJECT papers about:\n"
    "- General image classification, detection, or segmentation without report generation\n"
    "- Image super-resolution or image quality\n"
    "- General NLP or LLMs not applied to radiology reports\n"
    "- General medical AI not related to report generation\n"
    "- Radiology education, scheduling, billing\n\n"
    "Be strict. When in doubt, reject.\n\n"
    "Respond ONLY with valid JSON. No preamble, no markdown fences.\n"
    'Schema: {"relevant": true/false, "score": 1-5, "reason": "one sentence", "component": "component name"}'
)

DIGEST_SYSTEM = (
    "Write a short HTML email digest of radiology report generation papers.\n\n"
    "Group by component using <h2>. For each paper: <h3> title as hyperlink, "
    "authors in <small>, one sentence summary in <p>.\n\n"
    "Keep total output under 2000 characters. No wrapper tags."
)


def call_claude(system, user):
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 1000,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def filter_relevant(papers, min_score=4):
    relevant = []
    for p in papers:
        user_msg = "Title: " + p["title"] + "\nAuthors: " + p["authors"] + "\nAbstract: " + p["abstract"]
        try:
            raw = call_claude(RELEVANCE_SYSTEM, user_msg)
            result = json.loads(raw)
            score = int(result.get("score", 0))
            print("  Score " + str(score) + "/5 - " + p["title"][:60] + "...")
            if result.get("relevant") and score >= min_score:
                p["relevance_score"] = score
                p["relevance_reason"] = result.get("reason", "")
                p["pipeline_component"] = result.get("component", "General")
                relevant.append(p)
        except Exception as e:
            print("  Filtering error for '" + p["title"][:40] + "': " + str(e))

    relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return relevant


def generate_digest(papers):
    if not papers:
        return "<p>No new relevant papers found today.</p>"

    papers_json = json.dumps([
        {
            "title": p["title"],
            "authors": p["authors"],
            "url": p["url"],
            "abstract": p["abstract"],
            "source": p["source"],
            "pipeline_component": p.get("pipeline_component", "General"),
        }
        for p in papers
    ], indent=2)

    html = call_claude(DIGEST_SYSTEM, "Papers:\n" + papers_json)
    return html
