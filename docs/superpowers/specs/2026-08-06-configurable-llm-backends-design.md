# Configurable LLM Backends — Design

**Date:** 2026-08-06
**Branch:** project-revamp (continues the resume-agent refactor)
**Status:** Approved by user; pending implementation plan

## Goal

Make the resume agent's LLM layer backend-agnostic so `resume tailor` can run on a
local Ollama model (default), the Anthropic API, or Azure OpenAI (Microsoft Foundry
Models) — selected purely through environment variables, with no code changes.

## Motivation

- The current agent hardcodes `Anthropic()` + `claude-sonnet-4-6`, so every run
  requires an `ANTHROPIC_API_KEY`.
- The user has Ollama installed locally (`qwen2.5-coder:7b`, `gemma4:e4b`) and an
  Azure for Students subscription ($100 credit, 12 months, no credit card).
  Azure OpenAI (Foundry Models) is pay-per-token: `gpt-4o-mini` is $0.15/1M input
  and $0.60/1M output, so one tailor run (~10–30K tokens) costs well under $0.001
  — effectively free against the $100 credit.
- GitHub Models (Azure-hosted, free with monthly rate limits) closed to new
  customers in June 2026, so it is not a viable free tier.
- The user chose a LangChain ChatModel abstraction so one call interface serves
  all backends, including any OpenAI-compatible serverless endpoint.

## Constraints (bind this feature)

- Graph structure is unchanged: `select_projects → select_education →
  draft_rewrite → verify → route_after_verify → generate_typst_file →
  compile_typst → END`. Only the three LLM nodes' internals change.
- The `verify` node stays deterministic — no LLM, no changes.
- Guardrail semantics are unchanged: `draft_rewrite` may only reorder/rephrase/
  re-emphasize facts already in the entry's bullets and tags; unresolved
  violations fall back to the entry's original `bullets.full`.
- Files stay at repo root; `resume_agent.py` and `cli.py` are the only modules.
- No `selected.json` is ever written.
- Verification stays pytest-free: `python -m py_compile` plus targeted
  `.venv/bin/python` heredocs.
- The default backend (`ollama`) must not require a reachable server or network
  at import time (ChatModel construction is lazy; only `.invoke()` hits the wire).
- All backends are constructed with temperature 0 for deterministic JSON output.

## Configuration

Environment variables, all optional except where noted:

| Variable | Purpose | Default |
|---|---|---|
| `RESUME_BACKEND` | `ollama` \| `anthropic` \| `azure` | `ollama` |
| `RESUME_MODEL` | Model name override | `qwen2.5-coder:7b` / `claude-sonnet-4-6` / `gpt-4o-mini` (per backend) |
| `RESUME_BASE_URL` | Endpoint override | `http://localhost:11434` (ollama); unset otherwise |
| `RESUME_AZURE_DEPLOYMENT` | Azure deployment name | `RESUME_MODEL` |
| `RESUME_AZURE_API_VERSION` | Azure API version | `2024-06-01` |
| `AZURE_OPENAI_API_KEY` | Azure key (standard Azure env var) | — |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint (standard Azure env var) | — |
| `ANTHROPIC_API_KEY` | Anthropic key (only for `anthropic` backend) | — |

Rules:

- An invalid `RESUME_BACKEND` value fails fast at import with a `ValueError`
  listing the valid options — never silently falls back.
- When `RESUME_BACKEND=azure` and `RESUME_BASE_URL` is set, the endpoint is
  treated as an OpenAI-compatible serverless deployment (Foundry "instant
  access") and `ChatOpenAI` is used with that base URL.
- When `RESUME_BACKEND=azure` and `RESUME_BASE_URL` is unset, `AzureChatOpenAI`
  is used with `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, the deployment,
  and the API version.
- `RESUME_BASE_URL` with the `anthropic` backend points ChatAnthropic at an
  Anthropic-compatible gateway (supported, not primary).

## Architecture

### `get_chat_model()` → `BaseChatModel`

Reads the env vars and returns the appropriate LangChain ChatModel:

- `ollama` → `ChatOllama(model=..., base_url=..., temperature=0)`
- `anthropic` → `ChatAnthropic(model=..., base_url=optional, temperature=0)`
- `azure` → `AzureChatOpenAI(...)` or `ChatOpenAI(base_url=..., temperature=0)`

Module-level `client = get_chat_model()` replaces the current
`client = Anthropic()`; the `MODEL` constant is removed (per-backend defaults
replace it). Constructing the ollama default performs no I/O.

### `_llm_call(system, user, max_tokens)` → `str`

The single invocation wrapper used by all three LLM nodes:

1. `client.invoke([SystemMessage(content=system), HumanMessage(content=user)])`
2. Normalize `AIMessage.content` to plain text: a bare string is returned as-is;
   a list of content blocks joins the `text` blocks.
3. Return the text.

### `_extract_json(text)` → parsed value | `None`

Robust JSON extraction for weak models:

1. Strip markdown fences (```` ```json ````, ```` ``` ````) and surrounding
   whitespace.
2. Locate the first JSON-start character — `{`, `[`, or `"`.
3. Decode from that position with `json.JSONDecoder().raw_decode()`, which
   consumes exactly one complete JSON value (handles trailing prose).
4. A leading `"` (as in `"full"`) is decoded as a JSON string.
5. Any failure returns `None`.

### Node behavior changes (internals only)

- **`select_projects`** — `text = _llm_call(...)`; `parsed = _extract_json(text)`;
  if `not isinstance(parsed, list)`: `parsed = []`. Then the existing
  known-id filter. (Robust to fences *and* to valid-but-non-list JSON — also
  fixes the previously recorded T4 minor.)
- **`select_education`** — `text = _llm_call(...)`; `parsed = _extract_json(text)`;
  if `parsed is None`: `parsed = text.strip().strip('"')` (legacy fallback for
  bare tokens); tier = `parsed` if in `("full", "engineering_only")` else
  `engineering_only`. Tier→ids mapping unchanged.
- **`draft_rewrite`** — `text = _llm_call(...)`; `parsed = _extract_json(text)`;
  if `not isinstance(parsed, list)`: fall back to `entry["bullets"]["full"]`
  for that entry instead of crashing the graph (fixes the line-250 gap where a
  7B model's non-JSON reply raised `JSONDecodeError`).

### Dependencies

`pyproject.toml` adds `langchain-core`, `langchain-ollama`,
`langchain-anthropic`, and `langchain-openai`; the direct `anthropic` dependency
is dropped (it comes in transitively via `langchain-anthropic`). The
implementation plan records the exact minimum version floors for each new
dependency after installing them into the venv (latest stable at the time of
this change).

## Verification

- `python -m py_compile resume_agent.py cli.py`.
- Fake-model heredoc: a fake `client` object whose `invoke()` returns an object
  with `.content`, monkeypatched onto `resume_agent.client`. Exercises:
  - `_extract_json`: fenced JSON, prose-wrapped JSON, bare quoted string,
    garbage → `None`.
  - `select_projects`: fenced JSON array → ids selected; non-list → `[]`.
  - `select_education`: `"full"` (quoted), bare `engineering_only`, garbage →
    `engineering_only`.
  - `draft_rewrite`: fenced array → drafts; garbage → falls back to original
    `bullets.full`.
- End-to-end Task 9 against the real local Ollama `qwen2.5-coder:7b` with no
  env vars set (defaults), plus a `RESUME_MODEL` override run.
- The anthropic and azure paths are verified by construction tests only
  (a fake `invoke` plus factory selection given env vars) — they are not run
  live because no credentials exist in this environment.

## Docs

- `README.md` gains an env-var reference table and a short "choose your
  backend" section with ollama (default), azure (student credits), and
  anthropic examples.
- `docs/project.md` updates the LLM-layer description to the backend-agnostic
  model.
