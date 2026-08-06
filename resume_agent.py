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
import os
import re
import subprocess
from pathlib import Path
from typing import TypedDict

import yaml

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
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


# ---------------------------------------------------------------------------
# 3. Nodes — each is a plain function: (state) -> partial state update.
#    Nodes never call each other directly. They just describe what
#    changed; the graph below wires the transitions between them.
# ---------------------------------------------------------------------------

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
    return {"selected_projects": selected,
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
    return {"education_tier": tier, "selected_education": selected}


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


def compile_typst(state: ResumeState) -> dict:
    subprocess.run(["typst", "compile", state["output_typ"], state["output_pdf"]], check=True)
    return {}


# ---------------------------------------------------------------------------
# 4. Routing — this conditional edge is the whole reason LangGraph fits
#    better here than a linear chain: "retry until verified, or give up
#    and fall back" is a real cycle, not a straight line.
# ---------------------------------------------------------------------------

def route_after_verify(state: ResumeState) -> str:
    if state["violations"] and state["retry_count"] <= MAX_RETRIES:
        return "draft_rewrite"      # loop back and try again
    return "generate_typst_file"    # passed, or retries exhausted -> proceed with fallback


# ---------------------------------------------------------------------------
# 5. Graph assembly
# ---------------------------------------------------------------------------

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
