# Google Trends Report Generator (CLI)

Generate a **single, shareable HTML report** from Google Trends data with:
- interactive Plotly charts
- optional AI commentary (Gemini) grounded in computed metrics
- optional **tabbed “bundle”** output (Competitive / Brand Health / Category in one HTML)

This uses `pytrends` (an unofficial Google Trends wrapper). Rate limits can happen; this repo includes caching + throttling controls to help.

---

## Quick start

### 1) Setup

```bash
git clone https://github.com/jefflouella/google-trends-report.git
cd google-trends-report
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2) (Optional) Enable AI commentary

Set an API key (do **not** commit keys):

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

Or put it in `.env` (auto-loaded by the CLI):

```bash
GEMINI_API_KEY=YOUR_KEY_HERE
```

Aliases also supported:
- `GOOGLE_API_KEY`
- `GOOGLE_GENAI_API_KEY`

Disable AI anytime with:
- `--no-ai` (deterministic commentary)
- or `--ai none`

### 3) Generate a bundled report (one HTML with tabs)

```bash
python3 -m gtrends_analyzer run \
  --bundle \
  --main "Brand A" \
  --terms "Brand A" "Competitor 1" "Competitor 2" "Competitor 3" "Competitor 4" \
  --cluster "Incumbents: Brand A, Competitor 1, Competitor 2" \
  --cluster "Challengers: Competitor 3, Competitor 4"
```

The output goes to `reports/` by default. Use `--out` to control the file path.

---

## Report types

- **Competitive** (`--report-type competitive`, default): multi-term comparison (weekly trend, yearly dynamics + rebasing, seasonality, share-of-search proxy, similarity, and optional extras).
- **Brand health** (`--report-type brand-health`): deep dive on the main term (smoothed trend, drawdowns, seasonality, and optional extras).
- **Category** (`--report-type category`): cluster terms and analyze cluster trends/share/yearly dynamics.

### Bundle mode (recommended for client delivery)

Use `--bundle` to output **one HTML file** with tabs for Competitive / Brand Health / Category. This reuses the same fetched dataset across tabs to minimize Google requests.

---

## CLI reference

### Basic inputs

- `--main "..."`: anchor term used for >5 term scaling (and treated as the “main term” in several sections).
- `--terms term1 term2 ...`: terms to compare.
- `--geo US` (default `US`): use `""` (empty string) for worldwide.
- `--timeframe "today 5-y"` (default)
- `--gprop ""` (default): property filter (`images`, `news`, `youtube`, `froogle`).

### Output

- `--out ./reports/report.html`
- Plotly JS mode:
  - default: `--include-js cdn` (smallest file; requires internet)
  - fully offline: `--offline` (inlines Plotly JS; larger file)

### Category clusters

Repeat `--cluster`:

```bash
--cluster "Incumbents: Brand A, Competitor 1, Competitor 2" \
--cluster "Challengers: Competitor 3, Competitor 4"
```

### Data toggles (defaults ON)

By default the CLI tries to include as much as possible:
- related queries/topics
- interest by region
- suggestions (term disambiguation)

Disable any of them:
- `--no-related`
- `--no-region`
- `--no-suggestions`

Note: when running from CSV (`--csv`), related/region/suggestions are not available.

---

## Running from an exported CSV (recommended when rate-limited)

If Google blocks live fetching (HTTP 429), export a Google Trends `multiTimeline.csv` and run:

```bash
python3 -m gtrends_analyzer run \
  --bundle \
  --main "Brand A" \
  --terms "Brand A" "Competitor 1" "Competitor 2" "Competitor 3" "Competitor 4" \
  --csv ./multiTimeline.csv \
  --out ./reports/final.html
```

---

## Rate-limit protection

Live `pytrends` fetching can be rate-limited (HTTP 429). This repo includes:
- **Disk cache** for pytrends responses (default TTL: 24h)
- **Throttling** between requests (default: 15s minimum)
- **Exponential backoff** retry for transient 429 blocks

Useful flags:
- `--cache-dir ./.cache/gtrends_analyzer`
- `--cache-ttl-hours 24`
- `--refresh` (force fresh fetch)
- `--min-request-interval-seconds 30` (go slower)
- `--max-retries 4`

Best practice:
- generate once, then rerun from cache for iterations
- if blocked, switch to `--csv` mode

---

## Development

```bash
pytest -q
```

---

## Troubleshooting

- **HTTP 429 / “sorry” page**:
  - wait and retry later
  - run slower: `--min-request-interval-seconds 30`
  - avoid extras temporarily: `--no-related --no-region --no-suggestions`
  - use `--csv` mode

- **Report opens but charts are blank**:
  - if you used `--include-js cdn`, the client needs internet
  - use `--offline` to inline Plotly
