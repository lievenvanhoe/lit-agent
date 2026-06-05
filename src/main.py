import sys
from fetch_papers import fetch_all_papers
from summarize import filter_relevant, generate_digest
from send_email import send_digest

def main():
    print("=" * 50)
    print("Radiology Literature Agent - starting run")
    print("=" * 50)

    print("\n[1/3] Fetching papers...")
    papers = fetch_all_papers()

    if not papers:
        print("No papers found.")
        send_digest("<p>No new papers found today.</p>", 0)
        sys.exit(0)

    print("\n[2/3] Filtering " + str(len(papers)) + " papers...")
    relevant = filter_relevant(papers, min_score=2)
    print("-> " + str(len(relevant)) + " relevant papers after filtering")

    print("\n[3/3] Generating digest and sending email...")
    digest_html = generate_digest(relevant)
    send_digest(digest_html, len(relevant))
    print("\nAgent run complete.")

if __name__ == "__main__":
    main()
