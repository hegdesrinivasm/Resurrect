"""cli.py — Typer command line interface for the resume tailoring agent.

Commands:
  resurrect tailor <job_description.txt> [--company NAME]
  resurrect bank list [--section projects|education]
"""

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

app = typer.Typer()
bank_app = typer.Typer()
app.add_typer(bank_app, name="bank")

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
