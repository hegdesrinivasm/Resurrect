# Resurrect

Tailor a Typst resume to each job description without hand-editing Typst.
Point the CLI at a job description and a content bank; it selects the
best-matching projects, decides how much education detail to show, rewrites
bullet points to mirror the JD's phrasing, deterministically verifies that
no new facts were invented, and compiles a standalone `.typ` + PDF per
application.

## How it works

`resume_agent.py` runs a LangGraph pipeline:

1. `select_projects` — LLM picks the project entries most relevant to the JD.
2. `select_education` — LLM decides `full` (SSLC + PUC + Engineering) or
   `engineering_only` (Engineering only) from the JD's seniority/domain.
3. `draft_rewrite` — LLM rephrases each selected entry's bullets to match
   the JD, constrained to the facts already in the entry's original bullets
   and tag list.
4. `verify` — a deterministic, no-LLM gate extracts every number and tagged
   skill from each rewritten bullet and flags anything that wasn't in the
   original facts or tags. Fabricated content is not allowed through.
5. retry — violations loop back to `draft_rewrite` (max 2 retries); if they
   remain, the entry falls back to its original bullets.
6. `generate_typst_file` — inlines the selected entries into a copy of
   `main.typ` as data, producing a self-contained `outputs/resume_<company>.typ`.
7. `compile_typst` — runs `typst compile` to produce the PDF.

The content bank lives in `bank/`:

- `projects.yaml` — one entry per project: `id`, `title`, `date`, `tags`,
  `bullets.full` (ground truth), `bullets.short`.
- `education.yaml` — tiered entries (`sslc`, `puc`, `engineering`) with the
  same shape, plus a `tier` field.

Internships, skills, and activities are static sections in `main.typ`;
education renders as a bullet-free listing, and projects are pulled from
the bank per application.

## Setup

Requires `typst` on your PATH and a Google AI Studio API key.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
export GOOGLE_API_KEY=your_key   # or let the CLI prompt for it
```

## Backend

The agent uses Gemini via Google AI Studio (`langchain-google-genai`). The
model defaults to `gemini-2.5-flash` and can be overridden per run with
`--model`. LLM calls run at `temperature=0` for deterministic output.

| Env var | Default | Meaning |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — (prompted) | Google AI Studio API key |

`-p`/`--preflight` makes one tiny capped call before the graph runs, so a
bad or missing key fails fast instead of silently burning a rewrite cycle.

## Usage

```bash
# Tailor a resume to a job description (prompts for GOOGLE_API_KEY if unset)
resurrect tailor job_description.txt --company acme

# ...or verify the backend is ready first (probes the Gemini key)
resurrect tailor job_description.txt --company acme --preflight

# Use a specific Gemini model
resurrect tailor job_description.txt --company acme --model gemini-2.5-flash
```

Output: `outputs/resume_acme.typ` and `outputs/resume_acme.pdf`.

## Project layout

```
bank/            content bank (projects.yaml, education.yaml)
main.typ         Typst template with an @@DATA@@ marker for entry data
resume_agent.py  LangGraph pipeline (selection, rewrite, verify, render)
cli.py           Typer CLI
outputs/         generated per-company .typ and .pdf files
```
