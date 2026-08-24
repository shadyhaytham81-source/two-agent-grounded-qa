# Grounded Q&A Assistant — Rich Dad Poor Dad (PDF corpus)

A two-agent, grounded question-answering assistant. It answers questions
strictly from an ingested PDF corpus, and refuses to answer anything the
retrieved evidence doesn't support.

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │              LangGraph pipeline            │
                     │                                            │
  user query ──────▶ │  retrieve ──▶ draft ──▶ review ──┬──▶ finalize ──▶ answer
                     │  (Researcher)  (Researcher)  │        (shown in Streamlit)
                     │                               │
                     │                     NOT_GROUNDED & retries left
                     │                               │
                     │                               ▼
                     │                            revise ──▶ back to review
                     │                          (Researcher)
                     └──────────────────────────────────────────┘
```

- **Agent 1 — Researcher** (`agents/researcher.py`): embeds the query, searches
  the remote Qdrant collection for relevant passages, and drafts an answer
  using *only* those passages, with inline `[n]` citations. Instructed to
  paraphrase rather than quote at length (copyright-safe by design — see
  "Copyright" below).
- **Agent 2 — Reviewer** (`agents/reviewer.py`): checks every claim in the
  draft against the retrieved passages, and also flags over-quoting. If
  anything isn't actually supported, it sends the draft back to the
  Researcher **once** with specific feedback on what to fix.
- **Orchestration** (`graph/pipeline.py`): a genuine LangGraph state machine
  with a conditional edge — the reviewer's verdict decides whether the graph
  goes to `revise` (looping back to `review`) or `finalize`. It is not a
  fixed linear chain.
- **UI** (`app.py`): a Streamlit chat interface showing the final answer,
  its cited sources (book title + page number), and the reviewer's verdict.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| Vector DB | Qdrant Cloud (remote) | required by the brief |
| LLM (both agents) | Claude API (`anthropic` SDK) | required by the brief |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), runs **locally** | no extra API key/signup needed just to embed text |
| PDF parsing | `pdfplumber` | reliable text extraction for text-heavy books |
| Orchestration | LangGraph | genuine conditional agent handoffs, not a sequential chain |
| UI | Streamlit | required by the brief |

## Project structure

```
grounded-qa-assistant/
├── app.py                    # Streamlit chat UI
├── check_setup.py             # pre-flight check for .env / PDF / Anthropic / Qdrant
├── config.py                  # loads all settings from environment variables
├── retrieval.py                # Qdrant + embedding search helper
├── agents/
│   ├── researcher.py           # Agent 1: retrieve + draft + revise
│   └── reviewer.py             # Agent 2: fact-check draft against passages
├── graph/
│   ├── state.py                 # shared LangGraph state schema
│   └── pipeline.py              # graph wiring + conditional routing
├── ingestion/
│   └── ingest_pdf.py            # extract -> chunk -> embed -> upsert to Qdrant
├── corpus/
│   └── README.md                # where to put your own PDF (git-ignored)
├── tests/
│   ├── test_questions.py         # 100 test questions across 5 categories
│   ├── run_tests.py              # runs them all through the live pipeline
│   └── logs/                     # test run output (jsonl + txt + summary)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

### 1. Clone and create a virtual environment

Requires **Python 3.10–3.12** (`sentence-transformers` pulls in PyTorch,
which does not yet publish wheels for 3.13+).

```bash
git clone <your-repo-url>
cd grounded-qa-assistant
python3.11 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create a remote Qdrant Cloud cluster (free tier)

1. Go to https://cloud.qdrant.io and sign up.
2. Create a new **free cluster** (1GB is plenty for one book).
3. From the cluster's page in the Qdrant Cloud Web UI, copy the **cluster
   URL** and generate an **API key**.
   Reference: https://qdrant.tech/documentation/web-ui/

### 3. Get an Anthropic API key

Get a key from https://console.anthropic.com if you don't already have one.

### 4. Add your own copy of the book

Place your own legally-obtained PDF at `corpus/rich_dad_poor_dad.pdf` (see
`corpus/README.md`). **This file is git-ignored on purpose — see Copyright
below.**

### 5. Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env` with `ANTHROPIC_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.
Everything else has sensible defaults. **Never commit `.env`.**

`ANTHROPIC_MODEL` defaults to `claude-opus-5`. Both agents run on it; set it
to `claude-sonnet-5` if you'd rather trade a little accuracy for a cheaper
100-question test run.

### 5b. Verify the setup before ingesting

```bash
python check_setup.py
```

This checks the `.env` variables, that the PDF exists and has an extractable
text layer, that the Anthropic key works, and that the Qdrant cluster is
reachable — so a misconfiguration surfaces here rather than halfway through
ingestion.

### 6. Ingest the PDF

```bash
python -m ingestion.ingest_pdf
```

This extracts text page-by-page, chunks it, embeds locally, and upserts into
your remote Qdrant collection. You'll see a per-run summary of pages and
chunks ingested. Re-running is safe — it reuses the existing collection.

### 7. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Try it

- **Grounded question**: "What's the difference between an asset and a
  liability?" → should return a cited, paraphrased answer plus a
  ✅ Grounded verdict.
- **Out-of-scope question**: "What's the current price of Bitcoin?" →
  should be refused, since nothing in the corpus supports it.

## Test suite (100 cases)

`tests/test_questions.py` has 100 questions across five categories:
`core_concept`, `specific_detail`, `out_of_scope`, `adversarial`, and
`other_book_ambiguous` (references to related-but-different content, to
test that the system doesn't over-generalize what counts as "grounded").

Run the whole suite against your live pipeline (real Anthropic + Qdrant
calls — this costs a small amount of API usage):

```bash
python -m tests.run_tests
```

Useful flags for a quicker pass while iterating:

```bash
python -m tests.run_tests --limit 10
python -m tests.run_tests --category out_of_scope
```

Each run writes three files to `tests/logs/`:
- `test_run_<timestamp>.jsonl` — one JSON record per question: the question,
  every retrieved chunk (title, page, relevance score, text snippet), the
  draft, revision count, reviewer verdict + feedback, and the final answer.
- `test_run_<timestamp>.txt` — the same data as a readable transcript.
- `test_run_<timestamp>_summary.txt` — pass/caution/error counts per
  category.

These logs are **not** git-ignored — they're committed as the documented
test evidence for this deliverable. (The book PDF itself stays out of git;
only the small extracted text snippets in the logs are committed, which is
consistent with the paraphrase-first, short-quote-only design of the
Researcher/Reviewer prompts.)

## Design notes

- **Why local embeddings instead of an embeddings API?** Keeps the required
  credential surface to just `ANTHROPIC_API_KEY` + Qdrant creds — no third
  signup needed, per the brief's credential-hygiene requirement.
- **Why cap revision at one retry?** Matches the brief exactly ("sends it
  back once"). If the reviewer still isn't satisfied after one revision, the
  app finalizes but clearly labels the answer with a ⚠️ caution verdict
  rather than looping indefinitely or silently upgrading the verdict.
- **Refusal behavior**: retrieval is score-thresholded
  (`MIN_RELEVANCE_SCORE` in `.env`) — if nothing clears the bar, the
  Researcher never even attempts a draft, it returns `NOT_GROUNDED` directly.

## Copyright

*Rich Dad Poor Dad* is a commercially sold, copyrighted book. This project:
- **Never commits or redistributes the PDF.** `corpus/*.pdf` is git-ignored;
  you're expected to supply your own legally-obtained copy locally.
- **Instructs both agents to paraphrase, not reproduce.** The Researcher's
  system prompt caps verbatim quoting at ~10 words and one quote per
  passage; the Reviewer independently flags over-quoting as a groundedness
  failure, the same way it flags unsupported claims.
- Test logs store only short (≤300 char) text snippets per retrieved chunk
  for auditability — not full page reproductions.

## Security

- All credentials are loaded from environment variables via `config.py`
  (`python-dotenv` reads `.env` locally; in production, set real env vars
  instead).
- `.env` is git-ignored; only `.env.example` (no real secrets) is committed.
