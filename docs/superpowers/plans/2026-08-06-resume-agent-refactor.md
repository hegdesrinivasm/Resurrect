# Resume Agent Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Resurrect into a CLI agent that selects, rewrites, verifies, and renders a standalone tailored Typst resume (`.typ` + PDF) per job description.

**Architecture:** Two LLM selection nodes (projects relevance + education tier), a constrained LLM rewrite node, a deterministic no-LLM verify gate with a bounded retry loop, then a render node that inlines selected entries into a copy of `main.typ` at an `@@DATA@@` marker, producing a self-contained `outputs/resume_<company>.typ` that `typst compile` turns into a PDF. A Typer CLI exposes `resume tailor` and `resume bank list`.

**Tech Stack:** Python 3.11+, LangGraph (StateGraph, conditional edges), Anthropic SDK, PyYAML, Typer, `typst` CLI (subprocess).

## Global Constraints

- Python >= 3.11; files stay at the repo root (no package directory).
- Bank files: `bank/projects.yaml` and `bank/education.yaml` only. No `bank/internships.yaml`, no `bank/academics.yaml`, no `bank/achievements.yaml`.
- Internships, achievements, skills, profile, and extracurricular activities are static in `main.typ`; internships are a clearly-marked placeholder.
- LLM model is `claude-sonnet-4-6` (constant `MODEL`); max rewrite retries `MAX_RETRIES = 2`.
- `draft_rewrite` may only reorder/rephrase/re-emphasize facts already in an entry's original bullets + tags; `verify` is deterministic with NO LLM call.
- Entries with unresolved violations fall back to their original `bullets.full`.
- Output: `outputs/resume_<company>.typ` + `.pdf`. `selected.json` must NOT be written.
- No pytest suite — verification is `python -m py_compile` + targeted `.venv/bin/python` script checks + `typst compile` smoke tests (repo has no test infra).
- Every task ends with a commit; commit messages use conventional format (`feat:`, `refactor:`, `docs:`, `chore:`).

---

### Task 1: Foundation — gitignore, venv, dependencies, education bank

**Files:**
- Create: `.gitignore`
- Create: `bank/education.yaml`
- Modify: `bank/projects.yaml:1-7` (header comment only)
- Delete: `bank/academics.yaml`, `bank/internships.yaml`

**Interfaces:**
- Produces: `bank/education.yaml` with entries `sslc`, `puc` (placeholders), `engineering` (real). Each entry has `id`, `title`, `tier`, `date`, `tags`, `bullets.full` (+ optional `company`, `bullets.short`).
- Produces: `.venv/` with `langgraph`, `anthropic`, `pyyaml`, `typer` installed — used by every later task's verification.

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
outputs/
```

- [ ] **Step 2: Create venv and install dependencies**

Run (from repo root `/Users/hegdesrinivasm/Development/Personal Projects/Resurrect`):
```bash
python3 -m venv .venv
.venv/bin/pip install langgraph anthropic pyyaml typer
```
Expected: pip completes with no errors (already-installed packages print "Requirement already satisfied").

- [ ] **Step 3: Create `bank/education.yaml`**

```yaml
# Education entries only — degree, coursework, GPA. Hackathon wins,
# competitions, and IEEE/club involvement live as static sections in
# main.typ and are not selected by the agent.
#
# Entries are tiered: the agent picks "full" (SSLC + PUC + Engineering)
# or "engineering_only" (Engineering only) based on the job description.
#
# Same schema as projects.yaml, plus `tier` (sslc | puc | engineering)
# and an optional `company` for the degree name (rendered as the subtitle
# under the institution in the Education section).
#
# TODO: fill in real SSLC and PUC details below.

- id: sslc
  title: "TODO: your SSLC school name"
  tier: sslc
  date: "TODO: year"
  tags: [education]
  bullets:
    full:
      - "TODO: add your SSLC percentage/GPA and relevant highlights"

- id: puc
  title: "TODO: your PUC college name"
  tier: puc
  date: "TODO: year"
  tags: [education, science]
  bullets:
    full:
      - "TODO: add your PUC percentage and relevant highlights"

- id: engineering
  title: "Vivekananda College of Engineering & Technology, Puttur (VTU, Belagavi)"
  company: "Bachelor of Engineering in Artificial Intelligence and Machine Learning"
  tier: engineering
  date: "August 2027*"
  tags: [education, machine-learning, deep-learning, nlp, operating-systems, dsa]
  bullets:
    full:
      - "CGPA: 8.78 (up to 6th semester)"
      - "Relevant Coursework: Machine Learning frameworks and algorithms, Generative AI pipelines for text generation and summarization, NLP systems for sentiment analysis, OS fundamentals, DSA and Algorithm designing"
    short:
      - "B.E. in Artificial Intelligence and Machine Learning, CGPA 8.78"
```

- [ ] **Step 4: Update `bank/projects.yaml` header comment**

Replace lines 1-7 (the old comment about "three bank files" and `selected.json`):
```yaml
# Each entry is one project. `id` must be unique across the bank — it's
# the key used in the generated resume data and in verification violation
# messages.
#
# tags: the allowed-vocabulary whitelist. Used both by select_projects
# (to match against the JD) and by verify (a rewritten bullet can't
# introduce a tag/number that isn't in the original bullets or here).
#
# date: rendered right-aligned in the entry heading.
# subtitle: optional one-liner rendered under the title (e.g. "Academic
# Major Project: Python, PyTorch"); falls back to `company`, then tags.
#
# bullets.full: the complete, ground-truth bullet points for this entry.
# bullets.short: optional condensed variants for space-constrained
# resumes — not yet wired into resume_agent.py, reserved for later.
```

- [ ] **Step 5: Delete the old bank files**

```bash
rm bank/academics.yaml bank/internships.yaml
```

- [ ] **Step 6: Verify**

```bash
.venv/bin/python -c "import yaml; d=yaml.safe_load(open('bank/education.yaml')); assert [e['id'] for e in d]==['sslc','puc','engineering'], d; assert [e['tier'] for e in d]==['sslc','puc','engineering']; print('OK: education bank has 3 tiered entries')"
```
Expected: `OK: education bank has 3 tiered entries`. Also run `ls bank/` — must show only `education.yaml` and `projects.yaml`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore bank/education.yaml bank/projects.yaml
git rm --cached -r .venv 2>/dev/null; git rm bank/academics.yaml bank/internships.yaml
git commit -m "feat: restructure education bank into sslc/puc/engineering tiers"
```
(Note: `.venv/` is ignored; `git rm bank/academics.yaml bank/internships.yaml` records the deletions.)

---

### Task 2: Template — inline data marker and static internships

**Files:**
- Modify: `main.typ:25-29` (data block), `main.typ:81` (education section key), `main.typ:92-93` (internships section)

**Interfaces:**
- Consumes: `bank/education.yaml` (section key `education`).
- Produces: `main.typ` template with an `@@DATA@@` marker at `#let data = @@DATA@@`; `#section("education", "Education")`; literal `= Internships` placeholder. `main.typ` itself does NOT compile until the marker is replaced.

- [ ] **Step 1: Replace the DATA block**

Replace lines 25-29:
```typst
// -------------------- DATA --------------------
// This template is never hand-edited per company: the agent copies this
// file and inlines the selected entries at the @@DATA@@ marker below,
// producing a standalone outputs/resume_<company>.typ. This template
// does not compile until that replacement happens.
#let data = @@DATA@@
```

- [ ] **Step 2: Change the education section key**

Replace `main.typ:81`:
```typst
#section("education", "Education")
```

- [ ] **Step 3: Replace the internships section**

Replace `main.typ:92-93`:
```typst
// -------------------- INTERNSHIPS ----------------------
// Internships are static for now — replace the placeholder below with
// real roles, one resumeHeading + bullets block each.
= Internships
#resumeHeading("Role Title", "Month Year – Month Year", subtitle: "Company Name")
- TODO: add responsibilities, impact, and outcomes
```

- [ ] **Step 4: Verify the template compiles once the marker is replaced**

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess

template = Path("main.typ").read_text()
assert "@@DATA@@" in template
assert 'json("selected.json")' not in template
assert '#section("education", "Education")' in template

check = template.replace(
    "@@DATA@@",
    '(\n  "education": (\n      "engineering": (\n'
    '        "title": "Vivekananda College of Engineering & Technology, Puttur (VTU, Belagavi)",\n'
    '        "company": "B.E. in AI and ML",\n'
    '        "date": "August 2027*",\n'
    '        "tags": ("education",),\n'
    '        "bullets": ("CGPA: 8.78",),\n'
    "      )\n  )\n)",
)
out = Path("/tmp/resurrect-check.typ")
out.write_text(check)
subprocess.run(["typst", "compile", str(out), "/tmp/resurrect-check.pdf"], check=True)
print("OK: template compiles with inline education data")
PY
```
Expected: `OK: template compiles with inline education data` (also validates the single-element-array trailing-comma rule `("education",)`).

- [ ] **Step 5: Commit**

```bash
git add main.typ
git commit -m "feat: make main.typ a template with an inline data marker and static internships"
```

---

### Task 3: Agent — state, helpers, serializer, bank loading

**Files:**
- Modify: `resume_agent.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (self-contained module edit).
- Produces:
  - `ResumeState` TypedDict with fields `job_description: str`, `company: str`, `bank: dict`, `selected_ids: list[str]`, `selected_projects: list[str]`, `selected_education: list[str]`, `education_tier: str`, `drafts: dict[str, list[str]]`, `violations: list[str]`, `retry_count: int`, `output_typ: str`, `output_pdf: str`.
  - `load_bank() -> dict` loading sections `["projects", "education"]`, tagging each entry with `_section`.
  - `serialize_typst(data: dict) -> str` — dict of sections → Typst literal; dict → `(k: v, ...)` with quoted keys, list → `(v1, v2, ...)` (single-element lists get a trailing comma), str → escaped `"..."`, `None`/empty-list dict values omitted.
  - `sanitize(name: str) -> str` — lowercase alphanumerics + dashes for filenames.
  - Globals: `OUTPUT_DIR = Path("outputs")` replaces `OUTPUT_DATA`/`OUTPUT_PDF`.

- [ ] **Step 1: Update the module docstring**

Replace lines 1-22:
```python
"""
resume_agent.py — LangGraph agent that tailors a Typst resume to a job
description by selecting and rewriting entries from a content bank,
with a hard verification gate before anything is rendered.

This is written as a learning example, not a polished library: comments
call out *why* each piece exists, since the point is understanding
LangGraph's primitives (typed state, nodes as pure functions,
conditional edges for cycles), not just getting a working script.

Expects:
  bank/projects.yaml, bank/education.yaml
  main.typ            (your existing Typst template; @@DATA@@ marker)

Each bank entry should look like:
  - id: stutter-detection-app
    title: "Cadence — Stutter Detection App"
    tags: [ml, speech, pytorch, react]
    bullets:
      full: ["Built a ...", "Trained a ...", "Deployed ..."]
"""
```

- [ ] **Step 2: Update imports and globals**

Replace lines 24-40:
```python
import json
import re
import subprocess
from pathlib import Path
from typing import TypedDict

import yaml

from langgraph.graph import StateGraph, END
from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 2

BANK_DIR = Path("bank")
OUTPUT_DIR = Path("outputs")
MAIN_TYP = Path("main.typ")
```

- [ ] **Step 3: Update `ResumeState`**

Replace lines 50-58:
```python
class ResumeState(TypedDict):
    job_description: str
    company: str
    bank: dict                     # full content bank, loaded once at start
    selected_ids: list[str]        # temp: projects + education ids (removed in Task 6)
    selected_projects: list[str]   # project entry ids the agent picked
    selected_education: list[str]  # education entry ids implied by the tier
    education_tier: str            # "full" | "engineering_only"
    drafts: dict[str, list[str]]   # entry_id -> rewritten bullets (latest attempt)
    violations: list[str]          # human-readable strings describing what failed
    retry_count: int
    output_typ: str                # path of the generated .typ file
    output_pdf: str                # path of the compiled .pdf file
```

- [ ] **Step 4: Update `load_bank` and add `sanitize` + the Typst serializer**

Replace lines 79-90:
```python
def load_bank() -> dict:
    bank = {}
    for section in ["projects", "education"]:
        path = BANK_DIR / f"{section}.yaml"
        if not path.exists():
            continue
        with open(path) as f:
            for entry in yaml.safe_load(f) or []:
                entry["_section"] = section
                bank[entry["id"]] = entry
    return bank


def sanitize(name: str) -> str:
    """Lowercase alphanumerics and dashes for use in a filename."""
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


def _typst_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _typst_value(value) -> str:
    if isinstance(value, str):
        return _typst_str(value)
    if isinstance(value, list):
        inner = ", ".join(_typst_value(x) for x in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(value, dict):
        items = []
        for key, v in value.items():
            if v is None or (isinstance(v, list) and not v):
                continue
            items.append(f"{_typst_str(key)}: {_typst_value(v)}")
        return "(" + ", ".join(items) + ")"
    raise TypeError(f"unsupported type in Typst data: {type(value)}")


def serialize_typst(data: dict) -> str:
    """Render selected entries as an inline Typst dict literal, grouped by
    section, to be spliced into main.typ at the @@DATA@@ marker."""
    sections = []
    for section, entries in data.items():
        items = [
            f"      {_typst_str(entry_id)}: {_typst_value(entry)}"
            for entry_id, entry in entries.items()
        ]
        sections.append(f'  {_typst_str(section)}: (\n' + ",\n".join(items) + "\n  )")
    return "(\n" + ",\n".join(sections) + "\n)"
```

Do NOT touch `select_entries`, `draft_rewrite`, `verify`, `write_resume_data`, `compile_typst`, `route_after_verify`, or the graph assembly yet.

- [ ] **Step 5: Verify compile + serializer + bank loading**

```bash
.venv/bin/python -m py_compile resume_agent.py
```
Expected: no output, exit 0.

```bash
.venv/bin/python - <<'PY'
import resume_agent as r

out = r.serialize_typst({
    "projects": {"demo-app": {"title": "Demo App", "date": "2025",
                              "tags": ["ml"], "bullets": ["One bullet"]}}
})
expected = (
    '(\n  "projects": (\n'
    '      "demo-app": ("title": "Demo App", "date": "2025", '
    '"tags": ("ml",), "bullets": ("One bullet",))\n  )\n)'
)
assert out == expected, out
assert r.sanitize("Acme Corp!") == "acme-corp"
bank = r.load_bank()
assert sorted({e["_section"] for e in bank.values()}) == ["education", "projects"]
assert set(bank) == {"stutter-detection-app", "movie-review-sentiment",
                     "portfolio-website", "sslc", "puc", "engineering"}
print("OK: serializer, sanitize, and load_bank")
PY
```
Expected: `OK: serializer, sanitize, and load_bank`

- [ ] **Step 6: Commit**

```bash
git add resume_agent.py
git commit -m "refactor: update agent state, bank loading, and add Typst serializer"
```

---

### Task 4: Selection nodes — `select_projects` + `select_education`

**Files:**
- Modify: `resume_agent.py` (replace `select_entries`; rewire graph)

**Interfaces:**
- Consumes: `ResumeState`, `client`, `MODEL`, `load_bank` data from Task 3.
- Produces:
  - `select_projects(state: ResumeState) -> dict` returning `{"selected_projects": [...], "selected_ids": [...], "retry_count": 0, "violations": []}`.
  - `select_education(state: ResumeState) -> dict` returning `{"education_tier": "full"|"engineering_only", "selected_education": [...], "selected_ids": [...combined with projects...]}`.
  - Graph entry `select_projects -> select_education -> draft_rewrite -> verify -> (conditional) -> write_resume_data -> compile_typst -> END`.
- `selected_ids` is kept as a compatibility field so the not-yet-refactored `draft_rewrite`/`write_resume_data` keep working until Task 6.

- [ ] **Step 1: Replace `select_entries` with the two selection nodes**

Replace lines 99-122 (the whole `select_entries` function):
```python
def select_projects(state: ResumeState) -> dict:
    """LLM call #1: pick which project entries fit this JD."""
    index = [
        {"id": e["id"], "title": e["title"], "tags": e["tags"]}
        for e in state["bank"].values()
        if e.get("_section") == "projects"
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=(
            "You select resume project entries relevant to a job description. "
            "Return ONLY a JSON array of entry ids, most relevant first. "
            "No preamble, no markdown fences."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Job description:\n{state['job_description']}\n\n"
                f"Available project entries:\n{json.dumps(index, indent=2)}"
            ),
        }],
    )
    try:
        selected = json.loads(resp.content[0].text)
    except json.JSONDecodeError:
        selected = []
    known = {e["id"] for e in index}
    selected = [eid for eid in selected if eid in known]
    return {"selected_projects": selected, "selected_ids": selected,
            "retry_count": 0, "violations": []}


def select_education(state: ResumeState) -> dict:
    """LLM call #2: decide how much education history fits this JD."""
    index = [
        {"id": e["id"], "title": e["title"], "tags": e["tags"]}
        for e in state["bank"].values()
        if e.get("_section") == "education"
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=100,
        system=(
            'You decide how much education history to show in a resume for a '
            'given job description. Return ONLY the JSON string "full" or '
            '"engineering_only", no preamble.\n'
            '  "full": include SSLC, PUC, and Engineering (fresher roles or '
            'roles that value early schooling)\n'
            '  "engineering_only": include only Engineering (senior or '
            'domain-specific roles where only the degree matters)'
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Job description:\n{state['job_description']}\n\n"
                f"Available education entries:\n{json.dumps(index, indent=2)}"
            ),
        }],
    )
    try:
        tier = json.loads(resp.content[0].text)
    except json.JSONDecodeError:
        tier = resp.content[0].text.strip().strip('"')
    if tier not in ("full", "engineering_only"):
        tier = "engineering_only"
    tiers = ("sslc", "puc", "engineering") if tier == "full" else ("engineering",)
    selected = [e["id"] for e in index if e["id"] in state["bank"]
                and state["bank"][e["id"]].get("tier") in tiers]
    return {"education_tier": tier, "selected_education": selected,
            "selected_ids": state.get("selected_projects", []) + selected}
```

- [ ] **Step 2: Rewire the graph assembly**

Replace lines 224-242:
```python
graph = StateGraph(ResumeState)
graph.add_node("select_projects", select_projects)
graph.add_node("select_education", select_education)
graph.add_node("draft_rewrite", draft_rewrite)
graph.add_node("verify", verify)
graph.add_node("write_resume_data", write_resume_data)
graph.add_node("compile_typst", compile_typst)

graph.set_entry_point("select_projects")
graph.add_edge("select_projects", "select_education")
graph.add_edge("select_education", "draft_rewrite")
graph.add_edge("draft_rewrite", "verify")
graph.add_conditional_edges(
    "verify",
    route_after_verify,
    {"draft_rewrite": "draft_rewrite", "write_resume_data": "write_resume_data"},
)
graph.add_edge("write_resume_data", "compile_typst")
graph.add_edge("compile_typst", END)

app = graph.compile()
```

- [ ] **Step 3: Verify both nodes with a fake client**

```bash
.venv/bin/python - <<'PY'
import json
import resume_agent as r

class FakeContent:
    def __init__(self, text): self.text = text

class FakeResponse:
    def __init__(self, text): self.content = [FakeContent(text)]

class FakeMessages:
    def __init__(self, texts): self._texts = list(texts)
    def create(self, **kwargs): return FakeResponse(self._texts.pop(0))

class FakeClient:
    def __init__(self, texts): self.messages = FakeMessages(texts)

base = {
    "job_description": "ML engineer role touching speech processing and the web.",
    "bank": r.load_bank(),
    "selected_ids": [],
    "selected_projects": [],
    "selected_education": [],
    "education_tier": "",
    "drafts": {},
    "violations": [],
    "retry_count": 0,
    "output_typ": "",
    "output_pdf": "",
}

r.client = FakeClient(['["stutter-detection-app", "bogus-id", "portfolio-website"]'])
out = r.select_projects(dict(base))
assert out["selected_projects"] == ["stutter-detection-app", "portfolio-website"], out
assert out["selected_ids"] == out["selected_projects"]

r.client = FakeClient(['"full"'])
state = dict(base); state["selected_projects"] = ["stutter-detection-app"]
out = r.select_education(state)
assert out["education_tier"] == "full", out
assert out["selected_education"] == ["sslc", "puc", "engineering"], out
assert out["selected_ids"] == ["stutter-detection-app", "sslc", "puc", "engineering"]

r.client = FakeClient(['not json at all'])
out = r.select_education(dict(base))
assert out["education_tier"] == "engineering_only", out
assert out["selected_education"] == ["engineering"], out

print("OK: select_projects and select_education")
PY
```
Expected: `OK: select_projects and select_education`

- [ ] **Step 4: Commit**

```bash
git add resume_agent.py
git commit -m "feat: split selection into projects relevance and education tier nodes"
```

---

### Task 5: Rewrite, verify, and routing over both sections

**Files:**
- Modify: `resume_agent.py` (`draft_rewrite` only; `verify` and `route_after_verify` already correct)

**Interfaces:**
- Consumes: `state["selected_projects"] + state["selected_education"]`, `state["violations"]` from Task 4.
- Produces: `draft_rewrite(state) -> {"drafts": {entry_id: [bullets]}}` covering every selected project + education entry; unchanged `verify`/`route_after_verify`.

- [ ] **Step 1: Update `draft_rewrite` to iterate both selections**

Replace lines 125-162 (the whole `draft_rewrite` function):
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

        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=(
                "Rewrite resume bullets to better match a job description's phrasing. "
                "You may ONLY reorder, rephrase, or re-emphasize facts already present "
                "in the original bullets and the entry's tag list. "
                "NEVER add a number, tool, or outcome that isn't already there, "
                "even if the job description asks for it. "
                "Return ONLY a JSON array of bullet strings."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Job description:\n{state['job_description']}\n\n"
                    f"Entry tags (allowed vocabulary): {entry['tags']}\n"
                    f"Original bullets:\n" + "\n".join(entry["bullets"]["full"])
                    + violation_note
                ),
            }],
        )
        drafts[entry_id] = json.loads(resp.content[0].text)
    return {"drafts": drafts}
```

`verify` (lines 165-177) and `route_after_verify` (lines 214-217) stay unchanged — `verify` already scans `state["drafts"]` for every entry, and the routing already loops while `violations and retry_count <= MAX_RETRIES`.

- [ ] **Step 2: Verify with a fake client, including violation feedback**

```bash
.venv/bin/python -m py_compile resume_agent.py
.venv/bin/python - <<'PY'
import resume_agent as r

class FakeContent:
    def __init__(self, text): self.text = text

class FakeResponse:
    def __init__(self, text): self.content = [FakeContent(text)]

class FakeMessages:
    def __init__(self, texts): self._texts = list(texts)
    def create(self, **kwargs): return FakeResponse(self._texts.pop(0))

class FakeClient:
    def __init__(self, texts): self.messages = FakeMessages(texts)

state = {
    "job_description": "JD asking for ML work.",
    "bank": r.load_bank(),
    "selected_ids": ["stutter-detection-app", "engineering"],
    "selected_projects": ["stutter-detection-app"],
    "selected_education": ["engineering"],
    "education_tier": "engineering_only",
    "drafts": {},
    "violations": ["stutter-detection-app: introduced unverified facts {'x'}"],
    "retry_count": 1,
    "output_typ": "",
    "output_pdf": "",
}
r.client = FakeClient([
    '["Built a desktop app for stutter detection with a two-screen flow"]',
    '["CGPA: 8.78"]',
])
out = r.draft_rewrite(state)
assert list(out["drafts"]) == ["stutter-detection-app", "engineering"], out
assert len(out["drafts"]) == 2

# verify: rewritten bullet introduces nothing new -> clean; retry_count ticks up
clean = {"drafts": {"stutter-detection-app": ["Built a desktop app for stutter detection with a two-screen flow"]}}
vstate = dict(state, drafts=clean["drafts"], violations=[])
vout = r.verify(vstate)
assert vout["violations"] == [], vout
assert vout["retry_count"] == 2

# verify: fabricated number is caught
bad = {"drafts": {"stutter-detection-app": ["99% accuracy"]}}
bout = r.verify(dict(state, drafts=bad["drafts"], retry_count=0))
assert any("99%" in v for v in bout["violations"]), bout
assert r.route_after_verify({"violations": ["x"], "retry_count": 1}) == "draft_rewrite"
assert r.route_after_verify({"violations": [], "retry_count": 1}) == "write_resume_data"

print("OK: draft_rewrite, verify, and routing")
PY
```
Expected: `OK: draft_rewrite, verify, and routing`

- [ ] **Step 3: Commit**

```bash
git add resume_agent.py
git commit -m "feat: rewrite selected projects and education entries with violation feedback"
```

---

### Task 6: Render — `generate_typst_file`, `compile_typst`, final graph

**Files:**
- Modify: `resume_agent.py` (replace `write_resume_data` + `compile_typst`, final graph, `__main__`)

**Interfaces:**
- Consumes: `serialize_typst`, `sanitize`, `OUTPUT_DIR`, `MAIN_TYP` (Tasks 2-3), `select_projects`/`select_education` (Task 4), `draft_rewrite`/`verify` (Task 5).
- Produces:
  - `generate_typst_file(state) -> {"output_typ": str, "output_pdf": str}` — reads `main.typ`, replaces `@@DATA@@` with `serialize_typst(...)`, writes `outputs/resume_<company>.typ`; entries with unresolved violations fall back to original `bullets.full`; raises `RuntimeError` if the marker is missing.
  - `compile_typst(state) -> {}` — `typst compile <output_typ> <output_pdf>`.
  - Final graph: `select_projects -> select_education -> draft_rewrite -> verify -> (conditional draft_rewrite | generate_typst_file) -> compile_typst -> END`.
  - Removes: `selected_ids` field, `write_resume_data`, `OUTPUT_DATA`, `OUTPUT_PDF`.

- [ ] **Step 1: Replace `write_resume_data` with `generate_typst_file`**

Replace lines 180-200 (the whole `write_resume_data` function):
```python
def generate_typst_file(state: ResumeState) -> dict:
    """Replaces write_resume_data: combine the main.typ template with the
    selected (and possibly rewritten) entries, inlined as data, into a
    standalone outputs/resume_<company>.typ. Entries with unresolved
    violations fall back to their original bullets. No JSON is written."""
    data = {}
    for entry_id in state["selected_projects"] + state["selected_education"]:
        entry = state["bank"][entry_id]
        section = entry["_section"]
        bullets = state["drafts"].get(entry_id, entry["bullets"]["full"])
        if any(v.startswith(entry_id) for v in state["violations"]):
            bullets = entry["bullets"]["full"]
        payload_entry = {"title": entry["title"], "date": entry.get("date"),
                         "bullets": bullets}
        for field in ("subtitle", "company", "tags"):
            if entry.get(field):
                payload_entry[field] = entry[field]
        data.setdefault(section, {})[entry_id] = payload_entry

    template = MAIN_TYP.read_text()
    marker = "@@DATA@@"
    if marker not in template:
        raise RuntimeError(f"{MAIN_TYP} is missing the {marker!r} data marker")
    source = template.replace(marker, serialize_typst(data))

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_typ = OUTPUT_DIR / f"resume_{sanitize(state['company'])}.typ"
    output_typ.write_text(source)
    return {"output_typ": str(output_typ), "output_pdf": str(output_typ.with_suffix(".pdf"))}
```

- [ ] **Step 2: Replace `compile_typst`**

Replace lines 203-205:
```python
def compile_typst(state: ResumeState) -> dict:
    subprocess.run(["typst", "compile", state["output_typ"], state["output_pdf"]], check=True)
    return {}
```

- [ ] **Step 3: Update routing + graph + `__main__`**

Change `route_after_verify` return (line 217) from `"write_resume_data"` to `"generate_typst_file"`:
```python
def route_after_verify(state: ResumeState) -> str:
    if state["violations"] and state["retry_count"] <= MAX_RETRIES:
        return "draft_rewrite"      # loop back and try again
    return "generate_typst_file"    # passed, or retries exhausted -> proceed with fallback
```

Replace the graph assembly (lines 224-242) and `__main__` (lines 245-257):
```python
graph = StateGraph(ResumeState)
graph.add_node("select_projects", select_projects)
graph.add_node("select_education", select_education)
graph.add_node("draft_rewrite", draft_rewrite)
graph.add_node("verify", verify)
graph.add_node("generate_typst_file", generate_typst_file)
graph.add_node("compile_typst", compile_typst)

graph.set_entry_point("select_projects")
graph.add_edge("select_projects", "select_education")
graph.add_edge("select_education", "draft_rewrite")
graph.add_edge("draft_rewrite", "verify")
graph.add_conditional_edges(
    "verify",
    route_after_verify,
    {"draft_rewrite": "draft_rewrite", "generate_typst_file": "generate_typst_file"},
)
graph.add_edge("generate_typst_file", "compile_typst")
graph.add_edge("compile_typst", END)

app = graph.compile()


if __name__ == "__main__":
    jd_text = Path("job_description.txt").read_text()
    result = app.invoke({
        "job_description": jd_text,
        "company": "untitled",
        "bank": load_bank(),
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

Also remove the `selected_ids` field from `ResumeState` (Task 3) and delete the `selected_ids` keys from the two selection-node returns in Task 4.

- [ ] **Step 4: Verify render + compile with a canned state (no LLM)**

```bash
.venv/bin/python -m py_compile resume_agent.py
.venv/bin/python - <<'PY'
import subprocess
import resume_agent as r

state = {
    "job_description": "ML engineer at Acme.",
    "company": "acme corp",
    "bank": r.load_bank(),
    "selected_projects": ["stutter-detection-app"],
    "selected_education": ["engineering"],
    "education_tier": "engineering_only",
    # rewritten bullet that gets REJECTED below -> must fall back to original
    "drafts": {"stutter-detection-app": ["Built an app"]},
    "violations": ["stutter-detection-app: introduced unverified facts {'app'}"],
    "retry_count": 3,
    "output_typ": "",
    "output_pdf": "",
}
out = r.generate_typst_file(state)
subprocess.run(["typst", "compile", out["output_typ"], out["output_pdf"]], check=True)

source = open(out["output_typ"]).read()
assert "@@DATA@@" not in source
assert "stutter-detection-app" in source
assert "Cadence — Stutter Detection & Classification App" in source
# the draft "Built an app" was rejected -> original bullet must be present
assert "Built a desktop app for stutter detection" in source
# education has no draft -> original engineering bullets present
assert "CGPA: 8.78" in source
# selected.json must not be written
from pathlib import Path
assert not Path("selected.json").exists()
print("OK:", out["output_typ"], "->", out["output_pdf"])
PY
```
Expected: `OK: outputs/resume_acme-corp.typ -> outputs/resume_acme-corp.pdf`, and `outputs/resume_acme-corp.pdf` exists.

- [ ] **Step 5: Commit**

```bash
git add resume_agent.py
git commit -m "feat: render standalone resume_<company>.typ with inline entry data"
```

---

### Task 7: CLI — `cli.py`, `pyproject.toml`, editable install

**Files:**
- Create: `cli.py`
- Create: `pyproject.toml`

**Interfaces:**
- Consumes: `resume_agent.app` (compiled graph), `load_bank`, `MAX_RETRIES`, `sanitize`.
- Produces: console script `resume = "cli:app"`; commands `resume tailor <jd.txt> [--company NAME]` and `resume bank list [--section projects|education]`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "resurrect"
version = "0.1.0"
description = "Tailor a Typst resume to each job description from a content bank"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "anthropic>=0.40.0",
    "PyYAML>=6.0",
    "typer>=0.12.0",
]

[project.scripts]
resume = "cli:app"

[tool.setuptools]
py-modules = ["cli", "resume_agent"]
```

- [ ] **Step 2: Create `cli.py`**

```python
"""cli.py — Typer command line interface for the resume tailoring agent.

Commands:
  resume tailor <job_description.txt> [--company NAME]
  resume bank list [--section projects|education]
"""

from pathlib import Path
from typing import Optional

import typer

from resume_agent import MAX_RETRIES, app as graph_app, load_bank, sanitize

app = typer.Typer()
bank_app = typer.Typer()
app.add_typer(bank_app, name="bank")

PROGRESS = {
    "select_projects": "Selecting projects...",
    "select_education": "Deciding education tier...",
    "draft_rewrite": "Drafting rewrite...",
    "compile_typst": "Compiling...",
}


@app.command()
def tailor(
    job_description: Path = typer.Argument(
        ..., help="Path to the job description text file"
    ),
    company: Optional[str] = typer.Option(
        None, help="Company name used for the output filename (defaults to the JD file name)"
    ),
) -> None:
    """Tailor a resume to a job description and compile it to PDF."""
    company_name = company or job_description.stem
    state = {
        "job_description": job_description.read_text(),
        "company": company_name,
        "bank": load_bank(),
        "selected_projects": [],
        "selected_education": [],
        "education_tier": "",
        "drafts": {},
        "violations": [],
        "retry_count": 0,
        "output_typ": "",
        "output_pdf": "",
    }
    pdf_path = ""
    for update in graph_app.stream(state, stream_mode="updates"):
        for node, result in update.items():
            if (
                node == "verify"
                and result.get("violations")
                and result["retry_count"] <= MAX_RETRIES
            ):
                print(f"Violation found, retrying ({result['retry_count']}/{MAX_RETRIES})...")
            elif node == "generate_typst_file":
                pdf_path = result["output_pdf"]
                print(f"Writing {Path(result['output_typ']).name}...")
            elif node in PROGRESS:
                print(PROGRESS[node])
    print(f"Resume compiled at: {pdf_path}")


@bank_app.command("list")
def bank_list(
    section: Optional[str] = typer.Option(
        None, help="projects or education (default: both)"
    ),
) -> None:
    """List entries in the content bank."""
    sections = [section] if section in ("projects", "education") else ["projects", "education"]
    bank = load_bank()
    for section_name in sections:
        entries = [e for e in bank.values() if e.get("_section") == section_name]
        print(f"\n[{section_name}]")
        for entry in entries:
            print(f"  {entry['id']}: {entry['title']} (tags: {', '.join(entry['tags'])})")
```

- [ ] **Step 3: Install editable and verify the CLI**

```bash
.venv/bin/pip install -e .
.venv/bin/resume --help
.venv/bin/resume bank list
.venv/bin/resume bank list --section projects
```
Expected: `--help` shows `tailor` and `bank`; `bank list` prints `[projects]` with 3 entries and `[education]` with 3 entries (titles may show the `TODO:` placeholders for sslc/puc — expected).

- [ ] **Step 4: Commit**

```bash
git add cli.py pyproject.toml
git commit -m "feat: add Typer CLI with tailor and bank list commands"
```

---

### Task 8: Documentation — `docs/project.md` and `README.md`

**Files:**
- Rewrite: `docs/project.md`
- Rewrite: `README.md`

- [ ] **Step 1: Rewrite `docs/project.md`**

```markdown
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

`resume_agent.py` assembles a `StateGraph`:

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
resume tailor <job_description.txt> [--company NAME]
resume bank list [--section projects|education]
```

`resume tailor` streams per-node progress ("Selecting projects...",
"Deciding education tier...", "Drafting rewrite...", "Violation found,
retrying (1/2)...", "Writing resume_acme.typ...", "Compiling...").

## Status

- Built: two-bank selection (projects + education tier), deterministic
  verify + retry, inline-data rendering, Typer CLI, PDF compile.
- Not yet: `bullets.short` (condensed bullets) is still unused; no
  `human_review` node (the graph is structured so one can be inserted
  between `generate_typst_file` and `compile_typst` via LangGraph
  `interrupt()` + a checkpointer later); internships, SSLC, and PUC are
  placeholders awaiting real content.
```

- [ ] **Step 2: Rewrite `README.md`**

```markdown
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

Requires `typst` on your PATH and an `ANTHROPIC_API_KEY` in the environment.

## Usage

```bash
export ANTHROPIC_API_KEY=sk-...

# Tailor a resume to a job description
resume tailor job_description.txt --company acme

# List the content bank
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
```

- [ ] **Step 3: Commit**

```bash
git add docs/project.md README.md
git commit -m "docs: rewrite architecture doc and README for per-company output"
```

---

### Task 9: End-to-end verification (requires user input)

**Files:** none changed — this task validates the whole tool. You need a real JD in `job_description.txt` (or a sample file) and an `ANTHROPIC_API_KEY`.

- [ ] **Step 1: Provide a job description**

Replace the placeholder content of `job_description.txt` (currently only comment lines) with a real or realistic JD text. If you only have a sample, write it to `job_description.txt`:
```text
# Sample — replace with the real job description
Machine Learning Engineer, Acme Corp. Final-year freshers welcome.
Skills: PyTorch, NLP, sentiment analysis, speech processing, web apps.
Responsibilities: build and deploy ML models, collaborate on AI product features.
```

- [ ] **Step 2: Export the API key and run tailor**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/resume tailor job_description.txt --company acme
```
Expected stdout in order: `Selecting projects...`, `Deciding education tier...`, `Drafting rewrite...`, then (only if a bullet introduced new facts) `Violation found, retrying (1/2)...` and a second `Drafting rewrite...`, then `Writing resume_acme.typ...`, `Compiling...`, `Resume compiled at: outputs/resume_acme.pdf`.

- [ ] **Step 3: Inspect the artifacts**

- `outputs/resume_acme.typ` — contains `#let data = (...)` with an inline dict of the selected `"projects"` and `"education"` entries; no `@@DATA@@`; each selected entry's bullets are present.
- `outputs/resume_acme.pdf` — open it: header, profile, education (tier-dependent), skills, projects, internships placeholder, achievements, extracurriculars all render; dates right-aligned.
- Run the compile path once more manually to confirm reproducibility: `typst compile outputs/resume_acme.typ /tmp/resume_acme.pdf`.

- [ ] **Step 4: Re-run for a different tier**

Run `resume tailor` against a JD that reads as a senior/domain-specific role (e.g. a lead NLP role) and confirm the education section shows Engineering only (no SSLC/PUC rows). If the model still returns `"full"`, that is a prompt-tuning matter, not a structural one — note it and continue.

- [ ] **Step 5: Final review**

```bash
git status
git log --oneline -12
```
Expected: working tree clean (or only intended untracked files); history shows the conventional-commit trail from this plan. Address any stray files, then the refactor is complete.

---

## Self-Review Checklist

**Spec coverage:**
- [x] `select_projects` node — Task 4
- [x] `select_education` tier node (full/engineering_only) — Task 4
- [x] `draft_rewrite` constrained rewrite + violation feedback — Task 5
- [x] `verify` deterministic gate — Task 5 (unchanged logic, re-tested)
- [x] conditional edge, max 2 retries — Tasks 4-6
- [x] `generate_typst_file` standalone `.typ` with inline data + fallback — Task 6
- [x] `compile_typst` — Task 6
- [x] `resume tailor` with `app.stream()` + progress strings — Task 7
- [x] `resume bank list` — Task 7
- [x] bank schema (projects + education tiers) — Task 1
- [x] internships not a bank file; static placeholder — Tasks 1-2
- [x] achievements static in main.typ — untouched (Task 2)
- [x] human_review seam (discrete nodes between generate_typst_file and compile_typst) — Task 6 graph
- [x] packaging: pyproject + `resume = "cli:app"` + editable venv — Tasks 1, 7
- [x] docs rewrite — Task 8
- [x] verification commands — every task

**No placeholders:** every step has exact file content, exact commands, and expected output. The only `TODO:` strings are intentional user-facing placeholders (SSLC/PUC data, internship content) specified by the design.

**Type consistency:** `ResumeState` field names match across Tasks 3-7 (`selected_projects`, `selected_education`, `education_tier`, `drafts`, `violations`, `retry_count`, `output_typ`, `output_pdf`); `serialize_typst`/`sanitize`/`load_bank` signatures are used identically in Tasks 6-7; graph node names match function names and route targets throughout.
