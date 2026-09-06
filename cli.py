"""cli.py — Typer command line interface for the resume tailoring agent.

Commands:
  resurrect tailor <job_description.txt> [--company NAME] [--model MODEL] [-p/--preflight]
  resurrect bank list [--section projects|education]

Uses Google AI Studio (Gemini) via GOOGLE_API_KEY.
"""

import os
from pathlib import Path
from typing import Optional

import typer

from resume_agent import (
    MAX_RETRIES,
    DEFAULT_MODEL,
    app as graph_app,
    build_gemini_model,
    load_bank,
    probe_gemini,
)

app = typer.Typer()
bank_app = typer.Typer()
app.add_typer(bank_app, name="bank")


def _get_api_key(prompt) -> str:
    """Return GOOGLE_API_KEY from env, or prompt for it and export it."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        key = prompt("Google AI Studio API key (GOOGLE_API_KEY)", hide_input=True)
        if key:
            os.environ["GOOGLE_API_KEY"] = key
    return key


PROGRESS = {
    "select_projects": "Selecting projects...",
    "select_education": "Deciding education tier...",
    "draft_rewrite": "Drafting rewrite...",
    "compile_typst": "Compiling...",
}


@app.command()
def tailor(
    job_description: Path = typer.Argument(..., help="Path to the job description text file"),
    company: Optional[str] = typer.Option(
        None, help="Company name used for the output filename (defaults to the JD file name)"
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="Gemini model to use (default: gemini-2.5-flash)"
    ),
    preflight: bool = typer.Option(
        False, "-p", "--preflight",
        help="Check the Gemini backend is reachable before tailoring (probes for a bad key / no network)",
    ),
) -> None:
    """Tailor a resume to a job description and compile it to PDF."""
    _get_api_key(typer.prompt)
    llm = build_gemini_model(model)
    if preflight:
        probe_gemini(llm)
    company_name = company or job_description.stem
    state = {
        "job_description": job_description.read_text(),
        "company": company_name,
        "bank": load_bank(),
        "model": llm,
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
