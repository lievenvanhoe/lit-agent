import os
import json
import urllib.request

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-5"

RELEVANCE_SYSTEM = (
    "You are a relevance filter for a radiology report generation research digest.\n\n"
    "Accept a paper if it is about ANY of these topics:\n"
    "1. AI or LLM systems that generate radiology reports from images\n"
    "2. Vision-language models applied to radiology\n"
    "3. NLP applied to radiology report text\n"
    "4. RAG or retrieval systems for radiology reporting\n"
    "5. Hallucination or factual accuracy in medical report generation\n"
    "6. Fine-tuning or reinforcement learning for radiology LLMs\n"
    "7. Human-AI collaboration in radiology reporting\n\n"
    "Be generous - if the paper is related to radiology AND AI, accept it.\n"
    "Only reject papers completely unrelated to radiology reporting.\n\n"
    "You MUST respond ONLY with a JSON object. No other text.\n"
    "Example: {\"relevant\": true, \"score\": 4, \"reason\": \"directly about RRG\", \"component\": \"Report generation\"}"
)

DIGEST_SYSTEM = (
    "Write a short HTML email digest of radiology AI papers.\n"
    "Group by topic using <h2>. For each paper: <h3> title as hyperlink, <small> authors, <p> one sentence summary.\n"
    "Keep total under 2000 characters. No html/body tags."
)


def call_claude(system, user):
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 300,
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
        raw = resp.read()
        data = json.loads(raw)
    return data["content"][0]["text"]


def filter_relevant(papers, min_score=2):
    relevant = []
    for p in papers:
        user_msg = "Title: " + p["title"] + "\nAbstract: " + p["abstract"][:500]
        try:
            text = call_claude(RELEVANCE_SYSTEM, user_msg)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            score = int(result.get("score", 0))
            print("  Score " + str(score) + "/5 - " + p["title"][:60])
            if result.get("relevant") and score >= min_score:
                p["relevance_score"] = score
                p["relevance_reason"] = result.get("reason", "")
                p["pipeline_component"] = result.get("component", "General")
                relevant.append(p)
        except Exception as e:
            print("  Error: " + str(e))
    relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return relevant


def generate_digest(papers):
    if not papers:
        return "<p>No new relevant papers found today.</p>"
    papers_json = json.dumps([{
        "title": p["title"],
        "authors": p["authors"],
        "url": p["url"],
        "abstract": p["abstract"][:300],
        "component": p.get("pipeline_component", "General"),
    } for p in papers], indent=2)
    return call_claude(DIGEST_SYSTEM, "Papers:\n" + papers_json)
