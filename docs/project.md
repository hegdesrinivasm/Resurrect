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

`build_gemini_model(model)` returns a `ChatGoogleGenerativeAI` at
`temperature=0` for deterministic output. Credentials come from the
`GOOGLE_API_KEY` env var (Google AI Studio); the CLI prompts for it when
unset and exports it so construction sees it. All calls pass a `max_tokens`
cap (there is no longer an Ollama exception). The model defaults to
`gemini-2.5-flash`, overridable via `--model`.

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
resurrect tailor <job_description.txt> [--company NAME] [--model MODEL] [-p|--preflight]
resurrect bank list [--section projects|education]
```

`resurrect tailor` prompts for `GOOGLE_API_KEY` if unset, then streams
per-node progress ("Selecting projects...", "Deciding education tier...",
"Drafting rewrite...", "Violation found, retrying (1/2)...", "Writing
resume_acme.typ...", "Compiling..."). `resurrect bank list` never touches
the LLM.

### Preflight (`-p`/`--preflight`)

Runs after the model is built but before the graph starts, so a bad/missing
key fails fast instead of silently wasting rewrite cycles: `probe_gemini(model)`
makes one tiny, capped call (`"Reply with OK."`); auth (401) and network
failures are surfaced with actionable hints and a non-zero exit.

## Status

- Built: two-bank selection (projects + education tier), deterministic
  verify + retry, inline-data rendering, Typer CLI, PDF compile.
- Not yet: `bullets.short` (condensed bullets) is still unused; no
  `human_review` node (the graph is structured so one can be inserted
  between `generate_typst_file` and `compile_typst` via LangGraph
  `interrupt()` + a checkpointer later); achievements/activities remain
  static, and `main.typ` has no per-company override for internships yet.
