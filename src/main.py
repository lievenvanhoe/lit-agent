"""
Main orchestrator for the radiology literature monitoring agent.
Run this script directly or via GitHub Actions.
"""

import sys
from fetch_papers import fetch_all_papers
from summarize import filter_relevant, generate_digest
from send_email import send_digest, send_no_results_email


def main():
    print("=" * 60)
    print("Radiology Literature Agent — starting run")
    print("=" * 60)

    # Step 1: Fetch papers from PubMed + arXiv
    print("\n[1/3] Fetching papers...")
    papers = fetch_all_papers()

    if not papers:
        print("No papers found. Sending 'nothing new' email.")
        send_no_results_email()
        sys.exit(0)

    # Step 2: Filter for relevance using Claude
    print(f"\n[2/3] Filtering {len(papers)} papers for relevance...")
    relevant = filter_relevant(papers, min_score=3)
    print(f"→ {len(relevant)} relevant papers after filtering")

    if not relevant:
        print("No relevant papers found. Sending 'nothing new' email.")
        send_no_results_email()
        sys.exit(0)

    # Step 3: Generate digest and send email
    print(f"\n[3/3] Generating digest and sending email...")
    digest_html = generate_digest(relevant)
    success = send_digest(digest_html, len(relevant))

    if success:
        print("\n✓ Agent run complete.")
    else:
        print("\n✗ Email delivery failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
