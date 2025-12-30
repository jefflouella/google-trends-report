# Google Trends Report Generator (CLI)

Generate a **single, shareable HTML report** from Google Trends data with:
- interactive Plotly charts
- optional AI commentary (Gemini) grounded in computed metrics
- optional **tabbed “bundle”** output (Competitive / Brand Health / Category in one HTML)

This uses `pytrends` (unofficial Google Trends wrapper). It’s great for analysis, but you must assume **rate limits happen** and design for caching + slower request cadence (this repo includes both).

---

## Quick start

### 1) Setup

```bash
cd /Users/jefflouella/projects/GTrends
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2) (Optional) Enable AI commentary

Set an API key via environment variable (recommended). Do **not** commit keys.

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

Or put it in `.env` (auto-loaded by the CLI):

```bash
GEMINI_API_KEY=YOUR_KEY_HERE
```

Also supported as aliases:
- `GOOGLE_API_KEY`
- `GOOGLE_GENAI_API_KEY`

Disable AI anytime with:
- `--no-ai` (deterministic commentary)
- or `--ai none`

### 3) Generate a bundled report (one HTML with tabs)

```bash
python3 -m gtrends_analyzer run \
  --bundle \
  --main "JR Cigars" \
  --terms "JR Cigars" "Holts" "Famous Smoke" "Cigars International" "Cigar Page" \
  --cluster "Retailers: JR Cigars, Holts, Famous Smoke" \
  --cluster "BigBox: Cigars International, Cigar Page"
```

The output goes to `reports/` by default. Use `--out` to control the file path.

---

## What reports can it generate?

### Report types

- `--report-type competitive` (default)\n  Multi-term competitive analysis: weekly trend, yearly dynamics + rebasing, seasonality, share-of-search proxy, similarity, and (when available) region/related/suggestions.\n- `--report-type brand-health`\n  Deep dive on the main term: smoothed trend, short-term drawdowns, seasonality, plus optional extras.\n- `--report-type category`\n  Cluster terms into categories using `--cluster` and analyze cluster trends, cluster share, and yearly dynamics.\n\n### Bundle mode (recommended for client delivery)\n\nUse `--bundle` to output **one HTML file** with tabs for:\n- Competitive\n- Brand Health\n- Category\n\nThis **reuses the same fetched dataset** across tabs to minimize Google requests.\n\n---\n\n## CLI reference\n\n### Basic inputs\n\n- `--main \"...\"`\n  Anchor term used for >5 term scaling (and as the “main term” in several sections).\n- `--terms term1 term2 ...`\n  Terms to compare.\n- `--geo US` (default `US`)\n  Use `\"\"` (empty string) for worldwide.\n- `--timeframe \"today 5-y\"` (default)\n- `--gprop \"\"` (default)\n  Google property filter. Values include `images`, `news`, `youtube`, `froogle`.\n\n### Output\n\n- `--out /path/to/report.html`\n- Plotly JS mode:\n  - default: `--include-js cdn` (smallest file; requires internet)\n  - fully offline: `--offline` (inlines Plotly JS; bigger file)\n\n### Category clusters\n\nRepeat `--cluster` to define clusters:\n\n```bash\n--cluster \"Retailers: JR Cigars, Holts, Famous Smoke\" \\\n--cluster \"BigBox: Cigars International, Cigar Page\"\n```\n\n### Data toggles (defaults ON)\n\nThe CLI tries to include as much as possible by default:\n- related queries/topics\n- interest by region\n- suggestions (term disambiguation)\n\nDisable any of them:\n- `--no-related`\n- `--no-region`\n- `--no-suggestions`\n\nNote: when running from CSV (`--csv`), related/region/suggestions are not available.\n\n---\n\n## Running from an exported CSV (recommended when rate-limited)\n\nIf Google blocks live fetching (HTTP 429), export a Google Trends `multiTimeline.csv` and run:\n\n```bash\npython3 -m gtrends_analyzer run \\\n  --bundle \\\n  --main \"JR Cigars\" \\\n  --terms \"JR Cigars\" \"Holts\" \"Famous Smoke\" \"Cigars International\" \"Cigar Page\" \\\n  --csv /Users/jefflouella/projects/GTrends/multiTimeline.csv \\\n  --out /Users/jefflouella/projects/GTrends/reports/jr-cigars-final.html\n```\n\n---\n\n## Rate-limit protection (important)\n\nLive `pytrends` fetching is unofficial and can be rate-limited (HTTP 429). This repo includes:\n\n- **Disk cache** for pytrends responses (default TTL: 24h)\n- **Throttling** between requests (default: 15s minimum)\n- **Exponential backoff** retry for transient 429 blocks\n\nUseful flags:\n- `--cache-dir ./.cache/gtrends_analyzer`\n- `--cache-ttl-hours 24`\n- `--refresh` (force fresh fetch)\n- `--min-request-interval-seconds 30` (go slower)\n- `--max-retries 4`\n\nBest practice for client work:\n- generate once, then rerun from cache for iterations\n- if blocked, switch to `--csv` mode\n\n---\n\n## Development\n\n### Run tests\n\n```bash\npytest -q\n```\n\n---\n\n## Troubleshooting\n\n- **HTTP 429 / “sorry” page**:\n  - wait and retry later\n  - run slower: `--min-request-interval-seconds 30`\n  - avoid extras temporarily: `--no-related --no-region --no-suggestions`\n  - use `--csv` mode\n\n- **Report opens but charts are blank**:\n  - if you used `--include-js cdn`, the client needs internet\n  - use `--offline` to inline Plotly\n*** End Patch"}}


