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

Internships, achievements, skills, and activities are static sections in
`main.typ` (internships currently a marked placeholder).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Requires `typst` on your PATH and at least one reachable LLM backend.

## Choose your backend

The agent is backend-agnostic. Pick one via `RESUME_BACKEND`
(`ollama` — the default —, `anthropic`, or `azure`); the CLI also asks
interactively if the variable is unset. All models run at `temperature=0`.

| Env var | Default | Meaning |
| --- | --- | --- |
| `RESUME_BACKEND` | `ollama` | `ollama` \| `anthropic` \| `azure` |
| `RESUME_MODEL` | per backend | `qwen2.5-coder:7b` / `claude-sonnet-4-6` / `gpt-4o-mini` |
| `RESUME_BASE_URL` | `http://localhost:11434` | API base URL (never prompted) |
| `RESUME_AZURE_DEPLOYMENT` | = model | Azure deployment name |
| `RESUME_AZURE_API_VERSION` | `2024-06-01` | Azure API version |

Credentials come from the standard env vars of each backend
(`ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`,
`OPENAI_API_KEY` for OpenAI-compatible endpoints) — the CLI prompts for
them if they are unset.

- **Ollama (local, free)** — run `ollama pull qwen2.5-coder:7b` once, then
  use it with zero configuration:
  ```bash
  resume tailor job_description.txt --company acme
  ```
- **Anthropic** — `export ANTHROPIC_API_KEY=sk-...` (or let the CLI prompt):
  ```bash
  export RESUME_BACKEND=anthropic
  resume tailor job_description.txt --company acme
  ```
- **Azure / Microsoft Foundry** — great if you have Azure for Students
  credits; `gpt-4o-mini` runs well under a cent per resume. Point
  `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` at your Foundry
  deployment:
  ```bash
  export RESUME_BACKEND=azure
  export RESUME_MODEL=gpt-4o-mini
  resume tailor job_description.txt --company acme
  ```

## Usage

```bash
# Tailor a resume to a job description (backend from env or prompts)
resume tailor job_description.txt --company acme

# List the content bank (never touches the LLM)
resume bank list
resume bank list --section projects
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
