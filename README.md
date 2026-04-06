# 📄 Radiology Literature Monitoring Agent

A lightweight agent that monitors **PubMed** and **arXiv** daily for new papers on radiology report generation and LLMs, filters them for relevance using Claude, and sends you an HTML email digest.

---

## How it works

```
PubMed API ──┐
             ├──► fetch_papers.py ──► summarize.py (Claude) ──► send_email.py
arXiv API  ──┘
```

1. **Fetch** — queries PubMed and arXiv for papers from the last 24 hours
2. **Filter** — Claude scores each paper 1–5 for relevance; only scores ≥ 3 pass
3. **Summarize** — Claude writes a plain-English digest in HTML
4. **Send** — SendGrid delivers the email to your inbox

---

## Setup (one-time, ~30 minutes)

### 1. Clone or fork this repo

```bash
git clone https://github.com/YOUR_USERNAME/lit-agent.git
cd lit-agent
```

### 2. Get your API keys

| Service | Where to get it | Cost |
|---|---|---|
| Anthropic (Claude) | https://console.anthropic.com → API Keys | Pay-per-use (~$0.01–0.05/day) |
| SendGrid | https://app.sendgrid.com → Settings → API Keys | Free tier: 100 emails/day |

For SendGrid, you also need to:
- Verify a sender email address (Settings → Sender Authentication)
- Use that verified email as your `FROM_EMAIL`

### 3. Add secrets to GitHub

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

Add these four secrets:

| Secret name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Claude API key |
| `SENDGRID_API_KEY` | Your SendGrid API key |
| `TO_EMAIL` | Your email address (where digest is sent) |
| `FROM_EMAIL` | Your verified SendGrid sender address |

### 4. Enable GitHub Actions

Go to your repo → **Actions** tab → Enable workflows (if prompted).

The agent will now run automatically every day at **07:00 CET**.

---

## Manual test run

To trigger it manually without waiting for the schedule:

1. Go to **Actions → Daily Literature Digest**
2. Click **Run workflow**

Or test locally (requires API keys in your shell environment):

```bash
cd src
export ANTHROPIC_API_KEY=sk-ant-...
export SENDGRID_API_KEY=SG....
export TO_EMAIL=you@example.com
export FROM_EMAIL=alerts@yourdomain.com
python main.py
```

---

## Customisation

All customisations are in `src/fetch_papers.py`:

```python
PUBMED_QUERIES = [
    "radiology report generation large language model",
    "radiology LLM report automation",
    "automated radiology reporting transformer",
]

ARXIV_QUERIES = [
    "radiology report generation LLM",
    "automated radiology report large language model",
]

DAYS_BACK = 1   # how many days to look back
```

To change the run schedule, edit `.github/workflows/daily_digest.yml`:

```yaml
- cron: "0 6 * * *"   # daily at 06:00 UTC = 07:00 CET
```

---

## Project structure

```
lit-agent/
├── src/
│   ├── main.py           # orchestrator (entry point)
│   ├── fetch_papers.py   # PubMed + arXiv fetching
│   ├── summarize.py      # Claude relevance filter + digest generation
│   └── send_email.py     # SendGrid email delivery
└── .github/
    └── workflows/
        └── daily_digest.yml   # GitHub Actions schedule
```

---

## Estimated costs

| Component | Cost |
|---|---|
| Claude API (filtering + summary, ~10 papers/day) | ~$0.02–0.05/day |
| SendGrid | Free (well within 100 emails/day limit) |
| GitHub Actions | Free (well within 2000 min/month limit) |
| **Total** | **< $2/month** |
