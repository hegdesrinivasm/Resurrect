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


# ---------------------------------------------------------------------------
# 1. State — this is the object LangGraph threads through every node.
#    Unlike a hand-rolled while-loop, this is explicit and typed: each
#    node reads a slice of it and returns a *partial* update, and
#    LangGraph merges that into the running state for you.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Plain helpers — no LangGraph here. This is what the verify node
#    actually relies on to enforce the guardrail deterministically.
# ---------------------------------------------------------------------------

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def extract_facts(text: str, allowed_tags: list[str]) -> set[str]:
    """Numbers plus any tagged skill token mentioned in the text.
    This is intentionally simple — the point is to catch fabrication,
    not to be a complete NLP fact-extractor."""
    facts = set(NUMBER_RE.findall(text))
    lowered = text.lower()
    facts |= {tag for tag in allowed_tags if tag.lower() in lowered}
    return facts


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


# ---------------------------------------------------------------------------
# 3. Nodes — each is a plain function: (state) -> partial state update.
#    Nodes never call each other directly. They just describe what
#    changed; the graph below wires the transitions between them.
# ---------------------------------------------------------------------------

def select_entries(state: ResumeState) -> dict:
    """LLM call #1: pick which bank entries fit this JD."""
    index = [
        {"id": e["id"], "title": e["title"], "tags": e["tags"]}
        for e in state["bank"].values()
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=(
            "You select resume entries relevant to a job description. "
            "Return ONLY a JSON array of entry ids, most relevant first. "
            "No preamble, no markdown fences."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Job description:\n{state['job_description']}\n\n"
                f"Available entries:\n{json.dumps(index, indent=2)}"
            ),
        }],
    )
    selected = json.loads(resp.content[0].text)
    return {"selected_ids": selected, "retry_count": 0, "violations": []}


def draft_rewrite(state: ResumeState) -> dict:
    """LLM call #2: rewrite bullets for selected entries, constrained
    to each entry's own facts + tags. On a retry, prior violations for
    that entry are fed back so the model can correct itself."""
    drafts = {}
    for entry_id in state["selected_ids"]:
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


def verify(state: ResumeState) -> dict:
    """Deterministic gate — no LLM call. This is what actually enforces
    the guardrail; the prompt instructions in draft_rewrite are advisory
    only and can't be trusted on their own."""
    violations = []
    for entry_id, bullets in state["drafts"].items():
        entry = state["bank"][entry_id]
        original_facts = extract_facts(" ".join(entry["bullets"]["full"]), entry["tags"])
        for bullet in bullets:
            extra = extract_facts(bullet, entry["tags"]) - original_facts
            if extra:
                violations.append(f"{entry_id}: introduced unverified facts {extra}")
    return {"violations": violations, "retry_count": state["retry_count"] + 1}


def write_resume_data(state: ResumeState) -> dict:
    """Reached once verify() passes, or once retries are exhausted —
    in which case offending entries fall back to their original,
    unedited bullets rather than risk a false claim.

    Output is grouped by section so the Typst template can render
    each section (projects, internships, academics) in order."""
    payload = {}
    for entry_id in state["selected_ids"]:
        entry = state["bank"][entry_id]
        section = entry.get("_section", "other")
        bullets = state["drafts"].get(entry_id, entry["bullets"]["full"])
        if any(v.startswith(entry_id) for v in state["violations"]):
            bullets = entry["bullets"]["full"]
        payload_entry = {"title": entry["title"], "bullets": bullets}
        for field in ("date", "subtitle", "company", "tags"):
            if entry.get(field):
                payload_entry[field] = entry[field]
        payload.setdefault(section, {})[entry_id] = payload_entry
    OUTPUT_DATA.write_text(json.dumps(payload, indent=2))
    return {"resume_data": payload}


def compile_typst(state: ResumeState) -> dict:
    subprocess.run(["typst", "compile", str(MAIN_TYP), str(OUTPUT_PDF)], check=True)
    return {"pdf_path": str(OUTPUT_PDF)}


# ---------------------------------------------------------------------------
# 4. Routing — this conditional edge is the whole reason LangGraph fits
#    better here than a linear chain: "retry until verified, or give up
#    and fall back" is a real cycle, not a straight line.
# ---------------------------------------------------------------------------

def route_after_verify(state: ResumeState) -> str:
    if state["violations"] and state["retry_count"] <= MAX_RETRIES:
        return "draft_rewrite"      # loop back and try again
    return "write_resume_data"      # passed, or retries exhausted -> proceed with fallback


# ---------------------------------------------------------------------------
# 5. Graph assembly
# ---------------------------------------------------------------------------

graph = StateGraph(ResumeState)
graph.add_node("select_entries", select_entries)
graph.add_node("draft_rewrite", draft_rewrite)
graph.add_node("verify", verify)
graph.add_node("write_resume_data", write_resume_data)
graph.add_node("compile_typst", compile_typst)

graph.set_entry_point("select_entries")
graph.add_edge("select_entries", "draft_rewrite")
graph.add_edge("draft_rewrite", "verify")
graph.add_conditional_edges(
    "verify",
    route_after_verify,
    {"draft_rewrite": "draft_rewrite", "write_resume_data": "write_resume_data"},
)
graph.add_edge("write_resume_data", "compile_typst")
graph.add_edge("compile_typst", END)

app = graph.compile()


if __name__ == "__main__":
    jd_text = Path("job_description.txt").read_text()
    result = app.invoke({
        "job_description": jd_text,
        "bank": load_bank(),
        "selected_ids": [],
        "drafts": {},
        "violations": [],
        "retry_count": 0,
        "resume_data": {},
        "pdf_path": "",
    })
    print(f"Resume compiled at: {result['pdf_path']}")
