# Resurrect — Architecture

Resurrect is a CLI tool that tailors a single Typst resume to each job
description (JD) you apply to. You keep one content bank and one Typst
template; the agent selects, rewrites, and verifies per application, then
emits a standalone `.typ` file that is compiled to PDF — no hand-editing
Typst per company.

## Data flow

```
job_description.txt ─┐
                     ├─► resume_agent.py (LangGraph) ─► outputs/resume_<company>.typ ─► typst compile ─► outputs/resume_<company>.pdf
bank/*.yaml ─────────┘
```

Three layers, mirroring the original design:

1. **Content bank (data layer)** — `bank/projects.yaml` and
   `bank/education.yaml`. Each entry has a stable `id`, `title`, `date`,
   `tags` (an allowed-vocabulary whitelist), and ground-truth
   `bullets.full`. `bank/education.yaml` adds a `tier` field
   (`sslc` / `puc` / `engineering`) used to decide how much schooling to
   show.
2. **Agent (selection + rewriting)** — a LangGraph state machine with one
   node per concern and a deterministic verification gate (see below).
3. **Template (render layer)** — `main.typ` is a static visual template
   with an `@@DATA@@` marker. The agent copies it and inlines the selected
   entries as a Typst dict at the marker, producing a fully self-contained
   per-company `.typ`. Layout knowledge stays in Typst; data knowledge
   stays in the agent.

## The pipeline

`resume_agent.py` assembles a `StateGraph` over a `ResumeState` that
carries the chat model itself (`model`), so no node touches an SDK
directly:

- `select_projects` (LLM) — relevance-match project entries against the JD.
- `select_education` (LLM) — tier decision: `full` or `engineering_only`.
- `draft_rewrite` (LLM) — rephrase selected entries' bullets to mirror the
  JD, constrained to each entry's original facts + tags. Prior violations
  are fed back on retries.
- `verify` (deterministic, no LLM) — extracts numbers and tagged-skill
  tokens from each rewritten bullet; anything not present in the entry's
  original bullets or tags is a violation.
- routing — violations with retries left (max 2) loop back to
  `draft_rewrite`; otherwise proceed to render.
- `generate_typst_file` — inlines verified (or original, on exhaustion)
  bullets into `main.typ` at `@@DATA@@` → `outputs/resume_<company>.typ`.
- `compile_typst` — `typst compile`.

## LLM layer

`build_chat_model(backend, model, *, base_url, azure_deployment,
azure_api_version)` is a pure factory returning a LangChain `BaseChatModel`:
`ChatOllama`, `ChatAnthropic`, `AzureChatOpenAI`, or — when an azure
backend also sets `RESUME_BASE_URL` — `ChatOpenAI` against an
OpenAI-compatible serverless endpoint. The factory never reads the
environment; credentials are resolved by each LangChain integration from
its standard env vars (`AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_ENDPOINT`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). The CLI resolves backend/model as
env var → prompt → default (`RESUME_BACKEND`, `RESUME_MODEL`,
`RESUME_BASE_URL`, `RESUME_AZURE_DEPLOYMENT`, `RESUME_AZURE_API_VERSION`)
and prompts for any missing credential, exporting it so construction sees
it. All models run at `temperature=0`; the `max_tokens` cap is passed to
paid backends only (Ollama rejects the kwarg and stops on its own).

## Why the deterministic guardrail

Prompt instructions are advisory. The only thing standing between a
plausible-sounding lie and a shipped resume is the `verify` node: it never
calls the LLM, so it can't be talked out of flagging an invented number,
tool, or outcome. Unresolved violations fall back to the entry's original
bullets rather than risking a false claim.

## Why LangGraph

The retry loop ("rewrite until verified, or fall back") is a real cycle,
which LangGraph expresses as a conditional edge. Each node is a plain
function `(state) -> partial update`, which keeps the selection, rewriting,
verification, and rendering concerns independent and testable in isolation.

## CLI

```bash
resurrect tailor <job_description.txt> [--company NAME]
resurrect bank list [--section projects|education]
```

`resurrect tailor` asks for the backend (unless `RESUME_BACKEND` is set),
then streams per-node progress ("Selecting projects...", "Deciding
education tier...", "Drafting rewrite...", "Violation found, retrying
(1/2)...", "Writing resume_acme.typ...", "Compiling..."). `resurrect bank
list` never touches the LLM.

## Status

- Built: two-bank selection (projects + education tier), deterministic
  verify + retry, inline-data rendering, Typer CLI, PDF compile.
- Not yet: `bullets.short` (condensed bullets) is still unused; no
  `human_review` node (the graph is structured so one can be inserted
  between `generate_typst_file` and `compile_typst` via LangGraph
  `interrupt()` + a checkpointer later); internships, SSLC, and PUC are
  placeholders awaiting real content.
