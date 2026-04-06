"""
Relevance filtering and summarization using the Claude API.
Each paper is scored for relevance, then a digest is generated.
"""

import os
import json
import urllib.request


ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"

RELEVANCE_SYSTEM = """You are a strict relevance filter for a radiology report generation (RRG) research digest.

A paper is ONLY relevant if it directly addresses one of these:
1. Generating radiology reports from medical images using AI/LLMs
2. LLMs or transformers applied specifically to radiology report text
3. Multimodal models that take radiology images as input AND produce report text as output
4. RAG systems specifically for retrieving radiology guidelines or similar cases to aid report generation
5. Hallucination or factual grounding specifically in radiology report generation
6. Fine-tuning or RLHF specifically for radiology report generation models
7. Human-in-the-loop validation of AI-generated radiology reports

REJECT papers about:
- General image classification, detection, or segmentation without report generation
- Image super-resolution or image quality
- General NLP or LLMs not applied to radiology reports
- General medical AI not related to report generation
- Radiology education, scheduling, billing

Be strict. When in doubt, reject.

Respond ONLY with valid JSON. No preamble, no markdown fences.
Schema: {"relevant": true/false, "score": 1-5, "reason": "one sentence", "component": "component name"}
"""
"""

DIGEST_SYSTEM = """Write a short HTML email digest of radiology report generation papers.

Group by component using <h2>. For each paper: <h3> title as hyperlink, authors in <small>, one sentence summary.

Keep total output under 2000 characters. No wrapper tags.
"""


def call_claude(system: str, user: str) -> str:
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


def filter_relevant(papers: list[dict], min_score: int = 3) -> list[dict]:
    """Score each paper for relevance; keep those at or above min_score."""
    relevant = []
    for p in papers:
        user_msg = f"""Title: {p['title']}
Authors: {p['authors']}
Abstract: {p['abstract']}"""
        try:
            raw = call_claude(RELEVANCE_SYSTEM, user_msg)
            result = json.loads(raw)
            score = int(result.get("score", 0))
            print(f"  Score {score}/5 — {p['title'][:60]}...")
            if result.get("relevant") and score >= min_score:
                p["relevance_score"] = score
                p["relevance_reason"] = result.get("reason", "")
                p["pipeline_component"] = result.get("component", "General")
                relevant.append(p)
        except Exception as e:
            print(f"  Filtering error for '{p['title'][:40]}': {e}")

    relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return relevant


def generate_digest(papers: list[dict]) -> str:
    """Generate an HTML digest of the relevant papers."""
    if not papers:
        return "<p>No new relevant papers found today.</p>"

    papers_json = json.dumps([
        {
            "title": p["title"],
            "authors": p["authors"],
            "url": p["url"],
            "abstract": p["abstract"],
            "source": p["source"],
            "relevance_reason": p.get("relevance_reason", ""),
            "pipeline_component": p.get("pipeline_component", "General"),
        }
        for p in papers
    ], indent=2)

    html = call_claude(DIGEST_SYSTEM, f"Papers to summarize:\n{papers_json}")
    return html


if __name__ == "__main__":
    # Quick test with a dummy paper
    test_papers = [{
        "source": "arXiv",
        "title": "LLM-based Automated Radiology Report Generation from Chest CT",
        "authors": "Smith J, Jones A, et al.",
        "abstract": "We propose a large language model pipeline for automated generation of radiology reports from chest CT scans, achieving radiologist-level performance on a held-out test set.",
        "url": "https://arxiv.org/abs/2501.00000",
    }]
    print("Testing relevance filter...")
    relevant = filter_relevant(test_papers, min_score=2)
    print(f"Relevant: {len(relevant)}")
    if relevant:
        print("\nGenerating digest...")
        digest = generate_digest(relevant)
        print(digest[:500])
