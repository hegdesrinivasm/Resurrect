# Configurable LLM Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the resume agent talk to Ollama (local, default), Anthropic, or Azure-hosted OpenAI models, configured by env vars and — when the CLI runs interactively — prompted choices, so no `ANTHROPIC_API_KEY` is required.

**Architecture:** Replace the module-level `client = Anthropic()` with a pure `build_chat_model(backend, model, ...)` factory that returns a LangChain `BaseChatModel`; thread the model through `ResumeState["model"]` so the three LLM nodes call `_llm_call(state["model"], ...)` instead of an SDK directly. `cli.py` resolves backend/model/credentials as env var → prompt → default, exports prompted keys into the environment (which each LangChain integration reads), and passes the built model into the graph's initial state. Graph structure is unchanged.

**Tech Stack:** langchain-core, langchain-ollama (ChatOllama), langchain-anthropic (ChatAnthropic), langchain-openai (ChatOpenAI, AzureChatOpenAI), on top of the existing langgraph/typer/PyYAML stack.

## Global Constraints

- Python >= 3.11; files stay at repo root (no package dir); only `resume_agent.py`, `cli.py`, `pyproject.toml`, `README.md`, `docs/project.md` are touched.
- Backends are exactly `ollama`, `anthropic`, `azure` (order matters for messages). Invalid backend → `ValueError` naming the options.
- Every model is constructed with `temperature=0`.
- `build_chat_model` is a PURE factory: signature `build_chat_model(backend: str, model: str, *, base_url: str | None = None, azure_deployment: str | None = None, azure_api_version: str | None = None) -> BaseChatModel`. It NEVER reads env vars; keys/endpoints reach each LangChain integration through its standard env vars (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), which `cli.py` exports after prompting.
- NO module-level chat client. The old module-level `client` and `MODEL` constants are removed. `MAX_RETRIES = 2` stays.
- Env vars: `RESUME_BACKEND` (default `ollama`), `RESUME_MODEL` (defaults per backend: `qwen2.5-coder:7b` / `claude-sonnet-4-6` / `gpt-4o-mini`), `RESUME_BASE_URL` (ollama default `http://localhost:11434`; azure + base_url → OpenAI-compatible serverless `ChatOpenAI`), `RESUME_AZURE_DEPLOYMENT` (default = model), `RESUME_AZURE_API_VERSION` (default `2024-06-01`). `RESUME_BASE_URL` is never prompted.
- CLI resolution precedence: env var → interactive prompt → default. `bank` subcommand never touches the LLM.
- `max_tokens` is passed to every backend EXCEPT ollama: verified empirically that `ChatOllama.invoke(messages, max_tokens=N)` raises `TypeError` (the kwarg forwards to the ollama client's `chat()`). Ollama models stop on their own, so the cap is skipped there via `isinstance(model, ChatOllama)`.
- Graph structure is UNCHANGED: same node names, same edges, same conditional route. Only the three LLM-node bodies and the initial state change.
- Deps added to pyproject: `langchain-core>=1.0.0`, `langchain-ollama>=1.0.0`, `langchain-anthropic>=1.0.0`, `langchain-openai>=1.0.0`; the direct `anthropic>=0.40.0` dependency is dropped (now transitive via langchain-anthropic).
- No pytest. Verification = `python -m py_compile` + `.venv/bin/python` heredocs + `typst compile` smoke tests. Every task ends with a conventional commit.

---

### Task 1: Dependencies + model factory + LLM call helpers

**Files:**
- Modify: `pyproject.toml:11-16` (dependencies list)
- Modify: `resume_agent.py:23-40` (imports/globals block, add-only) and add new functions after `serialize_typst` (ends line 132)

**Interfaces:**
- Consumes: current `resume_agent.py` (reads cleanly at this commit; `client`/`MODEL` still present and used by the old node bodies — do NOT touch the nodes in this task).
- Produces: `build_chat_model(backend, model, *, base_url=None, azure_deployment=None, azure_api_version=None) -> BaseChatModel`; `_content_text(content) -> str`; `_llm_call(model, system, user, max_tokens) -> str`; `_extract_json(text)` (returns parsed JSON value or `None`). Task 2 uses all four; Task 3 uses `build_chat_model`.

- [ ] **Step 1: Update `pyproject.toml` dependencies**

Replace lines 11-16:

```toml
dependencies = [
    "langgraph>=0.2.0",
    "langchain-core>=1.0.0",
    "langchain-ollama>=1.0.0",
    "langchain-anthropic>=1.0.0",
    "langchain-openai>=1.0.0",
    "PyYAML>=6.0",
    "typer>=0.12.0",
]
```

- [ ] **Step 2: Install the new dependencies**

Run:
```bash
.venv/bin/pip install -e .
```
Expected: `Successfully installed ...`; no dependency conflict (langgraph 1.2.10 already installed and compatible with langchain-core 1.x).

- [ ] **Step 3: Update the imports/globals block in `resume_agent.py`**

Replace lines 23-40 (the whole import + globals block) with:

```python
import json
import os
import re
import subprocess
from pathlib import Path
from typing import TypedDict

import yaml

from langgraph.graph import StateGraph, END
from langchain_core.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

MAX_RETRIES = 2
BACKENDS = ("ollama", "anthropic", "azure")
DEFAULT_MODELS = {
    "ollama": "qwen2.5-coder:7b",
    "anthropic": "claude-sonnet-4-6",
    "azure": "gpt-4o-mini",
}
AZURE_API_VERSION = "2024-06-01"

BANK_DIR = Path("bank")
OUTPUT_DIR = Path("outputs")
MAIN_TYP = Path("main.typ")
```

Note: this removes `from anthropic import Anthropic`, `client = Anthropic()`, and `MODEL = "claude-sonnet-4-6"` — but the old node bodies still reference `client`/`MODEL` until Task 2, so the module will only fail at runtime if invoked. That is expected and fine (Task 2 removes the references); the module must still `py_compile` and import cleanly (construction of `Anthropic()` required no network anyway, but the import removal must not break import time).

- [ ] **Step 4: Add the LLM-layer functions**

Insert the following block immediately after `serialize_typst` (line 132), before the `# 3. Nodes` banner comment (line 135):

```python
# ---------------------------------------------------------------------------
# 2.5. LLM layer — one factory plus small call helpers. Nodes never talk to
#    an SDK directly; they call _llm_call against whatever model is in state.
# ---------------------------------------------------------------------------


def build_chat_model(
    backend: str,
    model: str,
    *,
    base_url: str | None = None,
    azure_deployment: str | None = None,
    azure_api_version: str | None = None,
) -> BaseChatModel:
    """Construct the chat model for a backend, temperature pinned to 0 for
    deterministic output. Keys/endpoints are resolved by each LangChain
    integration from the standard env vars, so this factory never reads env
    itself — the CLI exports prompted values before calling this."""
    if backend == "ollama":
        return ChatOllama(model=model, temperature=0, base_url=base_url or "http://localhost:11434")
    if backend == "anthropic":
        return ChatAnthropic(model=model, temperature=0)
    if backend == "azure":
        if base_url:
            return ChatOpenAI(model=model, temperature=0, base_url=base_url)
        return AzureChatOpenAI(
            model=model,
            temperature=0,
            azure_deployment=azure_deployment or model,
            api_version=azure_api_version or AZURE_API_VERSION,
        )
    raise ValueError(f"Unknown LLM backend: {backend!r} (choose from ollama, anthropic, azure)")


def _content_text(content) -> str:
    """Normalize an AIMessage content — either a plain string or a list of
    content blocks — into plain text."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _llm_call(model: BaseChatModel, system: str, user: str, max_tokens: int) -> str:
    """One chat call that returns plain text. max_tokens caps the paid
    backends; Ollama stops on its own and would reject the extra kwarg."""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    kwargs = {} if isinstance(model, ChatOllama) else {"max_tokens": max_tokens}
    return _content_text(model.invoke(messages, **kwargs).content)


def _extract_json(text: str):
    """Pull the first complete JSON value out of a model reply. Handles
    markdown fences and trailing prose. Returns None when nothing parses.
    The leading-quote branch exists for bare JSON strings like "full"."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
    hits = [(text.find(marker), marker) for marker in ('"', "{", "[")]
    hits = [(i, m) for i, m in hits if i >= 0]
    if not hits:
        return None
    start, _ = min(hits)
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except (json.JSONDecodeError, ValueError):
        return None
```

- [ ] **Step 5: Verify**

Run:
```bash
.venv/bin/python -m py_compile resume_agent.py
```
Expected: exit 0, no output.

Then run the heredoc:
```bash
.venv/bin/python - <<'PY'
import os
os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "test-key"

from resume_agent import build_chat_model, _content_text, _extract_json, _llm_call
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI

assert isinstance(build_chat_model("ollama", "qwen2.5-coder:7b"), ChatOllama)
assert isinstance(build_chat_model("anthropic", "claude-sonnet-4-6"), ChatAnthropic)
assert isinstance(build_chat_model("azure", "gpt-4o-mini"), AzureChatOpenAI)
assert isinstance(build_chat_model("azure", "gpt-4o-mini", base_url="http://localhost:8080"), ChatOpenAI)
try:
    build_chat_model("bogus", "x")
    raise SystemExit("FAIL: expected ValueError for bogus backend")
except ValueError as e:
    assert "ollama" in str(e)

assert _extract_json('```json\n["a", "b"]\n```') == ["a", "b"]
assert _extract_json('["a"] trailing prose') == ["a"]
assert _extract_json('"full"') == "full"
assert _extract_json('full') is None
assert _extract_json('nothing here') is None

class FakeResp:
    def __init__(self, content):
        self.content = content

class FakeModel:
    def __init__(self, content):
        self._content = content
    def invoke(self, messages, **kwargs):
        return FakeResp(self._content)

assert _content_text("plain") == "plain"
assert _content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
assert _content_text([]) == ""
assert _llm_call(FakeModel("hello"), "sys", "user", 50) == "hello"
assert _llm_call(FakeModel([{"type": "text", "text": "x"}]), "sys", "user", 50) == "x"
print("OK: factory picks the right model per backend; helpers parse and normalize")
PY
```
Expected: `OK: factory picks the right model per backend; helpers parse and normalize`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml resume_agent.py
git commit -m "feat: add configurable chat model factory and LLM call helpers"
```

---

### Task 2: Route the three LLM nodes through `state["model"]`

**Files:**
- Modify: `resume_agent.py` — `ResumeState` (line 50), `select_projects` (141-171), `select_education` (174-210), `draft_rewrite` (213-251), and `__main__` (344-359).

**Interfaces:**
- Consumes: `build_chat_model`, `_llm_call`, `_extract_json`, `_content_text`, `BACKENDS`, `DEFAULT_MODELS`, `AZURE_API_VERSION` from Task 1.
- Produces: `ResumeState["model"]: BaseChatModel`; `build_model_from_env() -> BaseChatModel` (used by `__main__`, and importable by tests). Task 3's CLI builds the model from its own settings and puts it in state.

- [ ] **Step 1: Add `model` to `ResumeState`**

After the `company: str` line (line 52), add:

```python
    model: object                  # chat model (BaseChatModel) built per backend
```

Note: typed as `object` (not `BaseChatModel`) because LangChain's `BaseChatModel` is a pydantic model and using it as a TypedDict value type triggers type-checker noise; the graph doesn't validate values, only keys. Runtime usage is via `_llm_call(state["model"], ...)`.

- [ ] **Step 2: Rewrite `select_projects`**

Replace the body of `select_projects` (lines 141-171) with:

```python
def select_projects(state: ResumeState) -> dict:
    """LLM call #1: pick which project entries fit this JD."""
    index = [
        {"id": e["id"], "title": e["title"], "tags": e["tags"]}
        for e in state["bank"].values()
        if e.get("_section") == "projects"
    ]
    selected = _extract_json(_llm_call(
        state["model"],
        (
            "You select resume project entries relevant to a job description. "
            "Return ONLY a JSON array of entry ids, most relevant first. "
            "No preamble, no markdown fences."
        ),
        (
            f"Job description:\n{state['job_description']}\n\n"
            f"Available project entries:\n{json.dumps(index, indent=2)}"
        ),
        max_tokens=500,
    ))
    if not isinstance(selected, list):
        selected = []
    known = {e["id"] for e in index}
    selected = [eid for eid in selected if eid in known]
    return {"selected_projects": selected,
            "retry_count": 0, "violations": []}
```

- [ ] **Step 3: Rewrite `select_education`**

Replace the body of `select_education` (lines 174-210) with:

```python
def select_education(state: ResumeState) -> dict:
    """LLM call #2: decide how much education history fits this JD."""
    index = [
        {"id": e["id"], "title": e["title"], "tags": e["tags"]}
        for e in state["bank"].values()
        if e.get("_section") == "education"
    ]
    text = _llm_call(
        state["model"],
        (
            'You decide how much education history to show in a resume for a '
            'given job description. Return ONLY the JSON string "full" or '
            '"engineering_only", no preamble.\n'
            '  "full": include SSLC, PUC, and Engineering (fresher roles or '
            'roles that value early schooling)\n'
            '  "engineering_only": include only Engineering (senior or '
            'domain-specific roles where only the degree matters)'
        ),
        (
            f"Job description:\n{state['job_description']}\n\n"
            f"Available education entries:\n{json.dumps(index, indent=2)}"
        ),
        max_tokens=100,
    )
    tier = _extract_json(text)
    if not isinstance(tier, str):
        tier = text.strip().strip('"')   # legacy fallback for unquoted answers
    if tier not in ("full", "engineering_only"):
        tier = "engineering_only"
    tiers = ("sslc", "puc", "engineering") if tier == "full" else ("engineering",)
    selected = [e["id"] for e in index if e["id"] in state["bank"]
                and state["bank"][e["id"]].get("tier") in tiers]
    return {"education_tier": tier, "selected_education": selected}
```

- [ ] **Step 4: Rewrite `draft_rewrite`**

Replace the body of `draft_rewrite` (lines 213-251) with:

```python
def draft_rewrite(state: ResumeState) -> dict:
    """LLM call #3: rewrite bullets for every selected entry (projects
    then education), constrained to each entry's own facts + tags. On a
    retry, prior violations for that entry are fed back so the model can
    correct itself."""
    drafts = {}
    for entry_id in state["selected_projects"] + state["selected_education"]:
        entry = state["bank"][entry_id]
        relevant_violations = [v for v in state["violations"] if v.startswith(entry_id)]
        violation_note = ""
        if relevant_violations:
            violation_note = (
                "\n\nYour previous attempt had these violations — fix them:\n"
                + "\n".join(relevant_violations)
            )
        text = _llm_call(
            state["model"],
            (
                "Rewrite resume bullets to better match a job description's phrasing. "
                "You may ONLY reorder, rephrase, or re-emphasize facts already present "
                "in the original bullets and the entry's tag list. "
                "NEVER add a number, tool, or outcome that isn't already there, "
                "even if the job description asks for it. "
                "Return ONLY a JSON array of bullet strings."
            ),
            (
                f"Job description:\n{state['job_description']}\n\n"
                f"Entry tags (allowed vocabulary): {entry['tags']}\n"
                f"Original bullets:\n" + "\n".join(entry["bullets"]["full"])
                + violation_note
            ),
            max_tokens=400,
        )
        bullets = _extract_json(text)
        if not isinstance(bullets, list):
            bullets = entry["bullets"]["full"]   # unparseable rewrite -> original
        drafts[entry_id] = bullets
    return {"drafts": drafts}
```

- [ ] **Step 5: Add `build_model_from_env`**

Insert immediately after `build_chat_model` (the function added in Task 1):

```python
def build_model_from_env() -> BaseChatModel:
    """Build a model from env vars with per-backend defaults. Used by the
    __main__ fallback path, which never prompts; the CLI resolves its own
    settings (prompts included) and calls build_chat_model directly."""
    backend = os.environ.get("RESUME_BACKEND", "ollama")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown LLM backend: {backend!r} (choose from {', '.join(BACKENDS)})")
    return build_chat_model(
        backend=backend,
        model=os.environ.get("RESUME_MODEL") or DEFAULT_MODELS[backend],
        base_url=os.environ.get("RESUME_BASE_URL") or None,
        azure_deployment=os.environ.get("RESUME_AZURE_DEPLOYMENT") or None,
        azure_api_version=os.environ.get("RESUME_AZURE_API_VERSION") or AZURE_API_VERSION,
    )
```

- [ ] **Step 6: Update `__main__`**

Replace the `__main__` block (lines 344-359) with:

```python
if __name__ == "__main__":
    jd_text = Path("job_description.txt").read_text()
    result = app.invoke({
        "job_description": jd_text,
        "company": "untitled",
        "bank": load_bank(),
        "model": build_model_from_env(),
        "selected_projects": [],
        "selected_education": [],
        "education_tier": "",
        "drafts": {},
        "violations": [],
        "retry_count": 0,
        "output_typ": "",
        "output_pdf": "",
    })
    print(f"Resume compiled at: {result['output_pdf']}")
```

- [ ] **Step 7: Verify**

Run:
```bash
.venv/bin/python -m py_compile resume_agent.py
```
Expected: exit 0. Also confirm no lingering references:
```bash
.venv/bin/python -c "import resume_agent; assert 'client' not in resume_agent.__dict__ and 'MODEL' not in resume_agent.__dict__; print('OK: no module-level client/MODEL')"
```

Then the fake-model heredoc — runs the FULL graph with a canned model, plus the env-based `build_model_from_env` checks:
```bash
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
from resume_agent import app, load_bank, build_model_from_env

bank = load_bank()

class FakeResp:
    def __init__(self, content):
        self.content = content

class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def invoke(self, messages, **kwargs):
        self.calls.append(kwargs)
        return FakeResp(self.responses.pop(0))

def bullets_for(eid):
    return json.dumps(bank[eid]["bullets"]["full"])

model = FakeModel([
    '["stutter-detection-app", "portfolio-website"]',
    '"full"',
    bullets_for("stutter-detection-app"), bullets_for("portfolio-website"),
    bullets_for("sslc"), bullets_for("puc"), bullets_for("engineering"),
])

result = app.invoke({
    "job_description": "ML engineer role for speech processing.",
    "company": "fake-corp",
    "bank": bank,
    "model": model,
    "selected_projects": [], "selected_education": [], "education_tier": "",
    "drafts": {}, "violations": [], "retry_count": 0,
    "output_typ": "", "output_pdf": "",
})
assert result["selected_projects"] == ["stutter-detection-app", "portfolio-website"]
assert result["education_tier"] == "full"
assert set(result["selected_education"]) == {"sslc", "puc", "engineering"}
assert result["retry_count"] == 1          # one verify pass, no violations
assert not result["violations"]
assert Path(result["output_pdf"]).exists()  # typst actually compiled
assert not Path("selected.json").exists()
# exactly the paid-backend calls carried a max_tokens kwarg
for kwargs in model.calls:
    assert "max_tokens" in kwargs, f"missing max_tokens: {kwargs}"
print("OK: full graph runs on an injected model and compiles a PDF")

# build_model_from_env: default + explicit + invalid
os.environ.pop("RESUME_BACKEND", None)
assert build_model_from_env().__class__.__name__ == "ChatOllama"
os.environ["RESUME_BACKEND"] = "anthropic"
os.environ["RESUME_MODEL"] = "claude-sonnet-4-6"
assert build_model_from_env().__class__.__name__ == "ChatAnthropic"
os.environ["RESUME_BACKEND"] = "bogus"
try:
    build_model_from_env()
    raise SystemExit("FAIL: expected ValueError")
except ValueError as e:
    assert "ollama" in str(e) and "azure" in str(e)
print("OK: build_model_from_env honors env with defaults and fails fast on bad backends")
PY
```
Expected output (both lines):
```
OK: full graph runs on an injected model and compiles a PDF
OK: build_model_from_env honors env with defaults and fails fast on bad backends
```

Clean up the fake output:
```bash
rm -f outputs/resume_fake-corp.typ outputs/resume_fake-corp.pdf
```

- [ ] **Step 8: Commit**

```bash
git add resume_agent.py
git commit -m "refactor: route LLM calls through the configured chat model"
```

---

### Task 3: CLI backend selection

**Files:**
- Modify: `cli.py` — imports (lines 8-13), add settings resolution, and `tailor` (lines 27-65).

**Interfaces:**
- Consumes: `BACKENDS`, `DEFAULT_MODELS`, `AZURE_API_VERSION`, `build_chat_model` from `resume_agent`; `ResumeState["model"]` from Task 2.
- Produces: `_resolve_llm_settings(prompt) -> dict` (backend/model/base_url/azure_deployment/azure_api_version) and `_prompt_env(prompt, name, text, *, hide_input=False)` — both importable for tests. `tailor` builds the model and passes it in state.

- [ ] **Step 1: Replace imports and add constants/helpers**

Replace lines 8-13 (the current import block) with:

```python
import os
from pathlib import Path
from typing import Optional

import typer

from resume_agent import (
    AZURE_API_VERSION,
    BACKENDS,
    DEFAULT_MODELS,
    MAX_RETRIES,
    app as graph_app,
    build_chat_model,
    load_bank,
)
```

Note: this drops the unused `sanitize` import (fixes the long-standing F401). `app = typer.Typer()` at line 15 and the `bank_app`/`add_typer` lines are unchanged.

Insert the following after the `app.add_typer(bank_app, name="bank")` line (17), before the `PROGRESS` dict:

```python
DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _prompt_env(prompt, name: str, text: str, *, hide_input: bool = False) -> None:
    """If env var `name` is unset, prompt for it and export the answer."""
    if os.environ.get(name):
        return
    value = prompt(text, hide_input=hide_input)
    if value:
        os.environ[name] = value


def _resolve_llm_settings(prompt) -> dict:
    """Resolve backend/model/connection settings as env var -> prompt ->
    default. `prompt` is injected so tests can fake typer.prompt.
    Prompted credentials are exported to env, which is how they reach the
    LangChain model constructors. RESUME_BASE_URL is never prompted."""
    backend = os.environ.get("RESUME_BACKEND", "")
    if backend not in BACKENDS:
        while backend not in BACKENDS:
            backend = prompt("LLM backend (ollama|anthropic|azure)", default=DEFAULT_BACKEND)
        os.environ["RESUME_BACKEND"] = backend

    model = os.environ.get("RESUME_MODEL") or prompt(
        "LLM model", default=DEFAULT_MODELS[backend]
    )

    base_url = os.environ.get("RESUME_BASE_URL") or None
    azure_deployment = os.environ.get("RESUME_AZURE_DEPLOYMENT") or None
    azure_api_version = os.environ.get("RESUME_AZURE_API_VERSION") or AZURE_API_VERSION

    if backend == "ollama" and base_url is None:
        base_url = DEFAULT_OLLAMA_BASE_URL
    elif backend == "azure":
        azure_deployment = azure_deployment or prompt("Azure deployment name", default=model)
        if base_url is None:
            _prompt_env(prompt, "AZURE_OPENAI_API_KEY", "Azure API key", hide_input=True)
            _prompt_env(prompt, "AZURE_OPENAI_ENDPOINT", "Azure endpoint URL")
        else:
            _prompt_env(prompt, "OPENAI_API_KEY", "OpenAI-compatible API key", hide_input=True)
    elif backend == "anthropic":
        _prompt_env(prompt, "ANTHROPIC_API_KEY", "Anthropic API key", hide_input=True)

    return {
        "backend": backend,
        "model": model,
        "base_url": base_url,
        "azure_deployment": azure_deployment,
        "azure_api_version": azure_api_version,
    }
```

- [ ] **Step 2: Update `tailor`**

In `tailor`, replace the state construction (lines 37-50) with:

```python
    settings = _resolve_llm_settings(typer.prompt)
    company_name = company or job_description.stem
    state = {
        "job_description": job_description.read_text(),
        "company": company_name,
        "bank": load_bank(),
        "model": build_chat_model(**settings),
        "selected_projects": [],
        "selected_education": [],
        "education_tier": "",
        "drafts": {},
        "violations": [],
        "retry_count": 0,
        "output_typ": "",
        "output_pdf": "",
    }
```

The `graph_app.stream(...)` loop below is unchanged. The `bank_list` command is unchanged (it never touches the LLM).

- [ ] **Step 3: Verify**

Run:
```bash
.venv/bin/python -m py_compile cli.py
```
Expected: exit 0.

Then the resolution heredoc (fakes `typer.prompt`; exercises env→prompt→default precedence and credential export):
```bash
.venv/bin/python - <<'PY'
import os
from cli import _resolve_llm_settings

def prompt(text, **kwargs):
    print(f"  prompt: {text} (default={kwargs.get('default')!r})")
    return kwargs.get("default") or "ENTERED"

# 1. all env unset -> ollama defaults, no key prompts
os.environ.clear()
s = _resolve_llm_settings(prompt)
assert s == {"backend": "ollama", "model": "qwen2.5-coder:7b",
             "base_url": "http://localhost:11434",
             "azure_deployment": None, "azure_api_version": "2024-06-01"}, s

# 2. env backend wins, model prompted, model env wins
os.environ.clear()
os.environ["RESUME_BACKEND"] = "azure"
s = _resolve_llm_settings(prompt)
assert s["backend"] == "azure" and s["model"] == "gpt-4o-mini", s
assert os.environ.get("AZURE_OPENAI_API_KEY") == "ENTERED"
assert os.environ.get("AZURE_OPENAI_ENDPOINT") == "ENTERED"
assert s["azure_deployment"] == "gpt-4o-mini" and s["azure_api_version"] == "2024-06-01"

os.environ.clear()
os.environ["RESUME_BACKEND"] = "anthropic"
os.environ["RESUME_MODEL"] = "claude-opus-4-6"
s = _resolve_llm_settings(prompt)
assert s["model"] == "claude-opus-4-6"
assert os.environ.get("ANTHROPIC_API_KEY") == "ENTERED"

# 3. azure + RESUME_BASE_URL -> serverless ChatOpenAI key path, no endpoint prompt
os.environ.clear()
os.environ["RESUME_BACKEND"] = "azure"
os.environ["RESUME_BASE_URL"] = "https://foundry.example.com"
s = _resolve_llm_settings(prompt)
assert s["base_url"] == "https://foundry.example.com"
assert os.environ.get("OPENAI_API_KEY") == "ENTERED"
assert "AZURE_OPENAI_ENDPOINT" not in os.environ

# 4. invalid env backend -> re-prompted until valid
os.environ.clear()
os.environ["RESUME_BACKEND"] = "gemini"
count = {"n": 0}
def fake_prompt(text, **kwargs):
    count["n"] += 1
    print(f"  prompt: {text}")
    if count["n"] > 1:
        return kwargs.get("default")
    return "gemini"
s = _resolve_llm_settings(fake_prompt)
assert s["backend"] == "ollama", s
assert s["model"] == "qwen2.5-coder:7b", s
print("OK: settings resolve env -> prompt -> default with credential export")
PY
```
Expected: the resolver prints one `prompt:` line per question asked across the four cases (backend/model for the default case; model/deployment/key/endpoint for Azure; model/deployment/key for the serverless case; backend re-prompt + model for the invalid-backend case), ending with `OK: settings resolve env -> prompt -> default with credential export`.

Then verify the CLI still works and `bank` never touches the LLM:
```bash
.venv/bin/resume --help
.venv/bin/resume bank list
```
Expected: help lists `tailor` + `bank`; `bank list` prints `[projects]` (3 entries) and `[education]` (3 entries).

- [ ] **Step 4: Commit**

```bash
git add cli.py
git commit -m "feat: add interactive backend selection to the CLI"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md` — Setup (lines 39-47), Usage (lines 48-61).
- Modify: `docs/project.md` — pipeline node descriptions (lines 35-49) and CLI section (lines 66-75).

**Interfaces:** Consumes the env-var names/defaults from the Global Constraints and the behavior implemented in Tasks 1-3. No new code.

- [ ] **Step 1: Update `README.md` Setup + Usage**

Replace lines 39-61 (the `## Setup` and `## Usage` sections through the output line) with:

````markdown
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
````

- [ ] **Step 2: Update `docs/project.md`**

Replace lines 35-49 (the `## The pipeline` node list) with:

```
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
```

Then replace lines 68-75 (the CLI section body) with:

```bash
resume tailor <job_description.txt> [--company NAME]
resume bank list [--section projects|education]
```

`resume tailor` asks for the backend (unless `RESUME_BACKEND` is set),
then streams per-node progress ("Selecting projects...", "Deciding
education tier...", "Drafting rewrite...", "Violation found, retrying
(1/2)...", "Writing resume_acme.typ...", "Compiling..."). `resume bank
list` never touches the LLM.

- [ ] **Step 3: Verify**

Run:
```bash
.venv/bin/resume bank list
```
Expected: unchanged output, `[projects]` + `[education]`. Grep both docs for the new env vars:
```bash
grep -n "RESUME_BACKEND" README.md docs/project.md
```
Expected: at least one hit in each file.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/project.md
git commit -m "docs: document configurable LLM backends"
```

---

### Task 5: End-to-end verification

**Files:**
- Modify: `job_description.txt` (untracked; sample JD — NOT committed).
- No production code changes. This task verifies the whole feature for real against the local Ollama backend.

**Interfaces:** Consumes the finished Tasks 1-4. Requires Ollama running (`ollama list` shows `qwen2.5-coder:7b`, already pulled) and `typst` on PATH (both present).

- [ ] **Step 1: Write a sample JD**

Overwrite `job_description.txt` (currently placeholder comments) with:

```
Machine Learning Engineer — Speech & Audio

We are looking for an ML engineer to join our audio intelligence team.
You will build and ship production speech-processing systems and desktop
tooling for clinicians.

Requirements:
- Hands-on experience with machine learning and speech processing
- Comfort building desktop applications and usable UIs
- Python, and exposure to NLP and sentiment analysis
- Understanding of model evaluation and baselines
```

This is a sample; leave it as the default JD afterward (it exercises the
speech/ML/desktop/NLP entries in the bank).

- [ ] **Step 2: Full default-backend run (no prompts)**

Run:
```bash
RESUME_BACKEND=ollama RESUME_MODEL=qwen2.5-coder:7b .venv/bin/resume tailor job_description.txt --company acme
```
Expected output order: `Selecting projects...` → `Deciding education tier...` → `Drafting rewrite...` → (optionally one or two `Violation found, retrying (1/2)...` lines, depending on model behavior) → `Writing resume_acme.typ...` → `Compiling...` → `Resume compiled at: outputs/resume_acme.pdf`.

Then inspect:
```bash
grep -c "@@DATA@@" outputs/resume_acme.typ; ls -la outputs/resume_acme.*
```
Expected: `0` occurrences of `@@DATA@@` in the generated `.typ` (marker fully replaced) and both `resume_acme.typ` + `resume_acme.pdf` present. Open the PDF to confirm the sections render.

- [ ] **Step 3: Interactive prompt path**

Run (feeding answers on stdin):
```bash
printf 'ollama\nqwen2.5-coder:7b\n' | RESUME_BACKEND="" .venv/bin/resume tailor job_description.txt --company acme2
```
Expected: the two prompt lines appear (`LLM backend (ollama|anthropic|azure)` then `LLM model`), and the same pipeline runs to `Resume compiled at: outputs/resume_acme2.pdf`.

- [ ] **Step 4: Model override**

Run:
```bash
RESUME_BACKEND=ollama RESUME_MODEL=gemma4:e4b .venv/bin/resume tailor job_description.txt --company gemma
```
Expected: same progress sequence, `outputs/resume_gemma.typ` + `.pdf` produced. (Confirms `RESUME_MODEL` override reaches the model.)

- [ ] **Step 5: Verify the retry loop still gates (fake model, no LLM)**

Heredoc forcing a persistent fabrication, asserting the graph retries and then falls back to original bullets:
```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from resume_agent import app, load_bank

bank = load_bank()
bad = json.dumps(bank["stutter-detection-app"]["bullets"]["full"] + ["Achieved 99% accuracy"])
clean = lambda eid: json.dumps(bank[eid]["bullets"]["full"])

class FakeResp:
    def __init__(self, content): self.content = content
class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
    def invoke(self, messages, **kwargs):
        return FakeResp(self.responses.pop(0))

# 2 projects + engineering only = 3 entries per draft pass; bad draft every pass
model = FakeModel([
    '["stutter-detection-app", "portfolio-website"]',
    '"engineering_only"',
    bad, clean("portfolio-website"), clean("engineering"),
    bad, clean("portfolio-website"), clean("engineering"),
    bad, clean("portfolio-website"), clean("engineering"),
])

result = app.invoke({
    "job_description": "Senior speech engineer, no early schooling needed.",
    "company": "retry-test",
    "bank": bank,
    "model": model,
    "selected_projects": [], "selected_education": [], "education_tier": "",
    "drafts": {}, "violations": [], "retry_count": 0,
    "output_typ": "", "output_pdf": "",
})
assert result["retry_count"] == 3, result["retry_count"]   # verify ran 3x, cap hit
assert result["violations"], "expected unresolved violations after exhausting retries"
out = Path(result["output_typ"]).read_text()
assert "99%" not in out and "Built a desktop app" in out   # fell back to original bullet
assert not Path("selected.json").exists()
print("OK: persistent fabrication is rejected, retries cap at 2, original bullets restored")
PY
```
Expected: `OK: persistent fabrication is rejected, retries cap at 2, original bullets restored`

Clean up:
```bash
rm -f outputs/resume_retry-test.typ outputs/resume_retry-test.pdf
```

- [ ] **Step 6: Final review of the branch**

Run:
```bash
git status --short
git log --oneline --no-decorate main..HEAD
```
Expected: `git status` shows only untracked `coding_agent_prompt.md` and `job_description.txt` (plus `outputs/` ignored); the log shows the five new commits (Tasks 1-4 + this feature's plan commit) sitting on the existing `project-revamp` stack. Nothing is committed in this task.

---

## Self-Review Notes

- **Spec coverage:** every spec item maps to a task — factory + purity (T1), model-in-state + node rewrites + `__main__` env fallback (T2), CLI env→prompt→default + credential export (T3), docs (T4), end-to-end + retry gate (T5). Invalid-backend `ValueError` is present in both `build_chat_model` (T1) and `build_model_from_env` (T2). `RESUME_BASE_URL` is never prompted (T3 resolver). `bank` never touches the LLM (T3 step 3 verification).
- **Known design choice:** `ResumeState["model"]` is typed `object` rather than `BaseChatModel` to avoid pydantic-vs-TypedDict type-checker friction; the graph only validates keys, so runtime behavior is unaffected.
- **Known limitation (spec-mandated):** anthropic/azure invoke paths are verified by construction + fake-model tests only; no credentials exist to hit those APIs. The ollama path is exercised for real in Task 5.
