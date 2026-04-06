"""
Relevance filtering and summarization using the Claude API.
Each paper is scored for relevance, then a digest is generated.
"""

import os
import json
import urllib.request


ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-20250514"

RELEVANCE_SYSTEM = """You are a radiology AI research assistant helping a radiologist 
and AI company director stay up to date on the latest research.

Your job: assess whether a paper is relevant to any component of a radiology report 
generation (RRG) pipeline, which includes:

1. Vision perception — vision transformers, image encoders, DICOM/PACS processing
2. Multimodal integration — vision-language alignment, cross-modal fusion
3. Clinical context — EMR integration, clinical NLP, patient history
4. RAG retrieval — guideline retrieval, similar case retrieval, institutional templates
5. Reconciliation — factual grounding, hallucination detection, finding prioritization
6. Report generation — multimodal LLMs, instruction fine-tuning, prompt engineering
7. Reinforcement learning — RLHF, RLAIF, reward modeling for medical text
8. Human supervision — radiologist-in-the-loop, report validation, human evaluation

Score higher (4-5) for papers directly addressing RRG or a specific component above.
Score lower (2-3) for adjacent work that is relevant but not directly about radiology reporting.
Score 1 for papers that are not relevant.

Respond ONLY with valid JSON. No preamble, no markdown fences.
Schema: {"relevant": true/false, "score": 1-5, "reason": "one sentence", "component": "component name from list above"}
"""

DIGEST_SYSTEM = """You are a radiology AI research assistant helping a busy radiologist 
and AI company director stay current with literature.

You will receive a list of relevant papers, each tagged with a pipeline_component 
corresponding to a layer in an RRG (radiology report generation) architecture:
1. Vision perception
2. Multimodal integration
3. Clinical context
4. RAG retrieval
5. Reconciliation
6. Report generation
7. Reinforcement learning
8. Human supervision

Write a concise daily digest email body (HTML).

Format:
- Brief intro sentence (today's date, number of papers)
- Group papers by pipeline_component using <h2> section headers
- For each paper: title as a hyperlink (<h3>), authors, 2-3 sentence plain-English summary 
  focusing on relevance to that pipeline layer
- Brief closing note

Keep it scannable. Use <h2> for component sections, <h3> for paper title-links, <p> for summaries.
No <html>/<body> wrapper tags needed.
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
