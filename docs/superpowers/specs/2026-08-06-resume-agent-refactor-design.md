# Resume Agent Refactor Design

## Goal

Refactor the Resurrect project from the current design — one
`select_entries` node covering all four bank sections, `write_resume_data`
writing `selected.json`, internships as a bank section — into the new
architecture: per-section selection nodes (projects + education tier),
a per-JD standalone `.typ` output file, internships as a static section,
and a Typer CLI. The refactor must keep the deterministic verification
guardrail and the LangGraph retry cycle intact.

## Decisions (confirmed with user)

- **Project name:** Resurrect.
- **Templating mechanism:** inline data into a copy of `main.typ`. The
  generator replaces a `@@DATA@@` marker line with the selected entries as
  an inline Typst dict literal, producing a fully self-contained
  `outputs/resume_<company>.typ`.
- **Achievements/hackathon/IEEE content:** stays static in `main.typ`.
  No `bank/achievements.yaml`.
- **Internships:** static section in `main.typ` as clearly-marked
  placeholder markup (user fills in real content later). No bank file.
- **Education:** `bank/education.yaml` with three tiered entries
  (`sslc`, `puc`, `engineering`). SSLC/PUC are clearly-marked placeholders;
  engineering keeps the real B.E. entry.
- **Packaging:** `pyproject.toml` with `[project.scripts] resume =
  "cli:app"`, installed editable in a venv. Files stay at repo root.

## Target structure

```
bank/
  projects.yaml            # unchanged
  education.yaml           # renamed from academics.yaml, restructured (sslc/puc/engineering tiers)
main.typ                   # template: static sections + @@DATA@@ marker; internships = marked placeholder
resume_agent.py            # refactored LangGraph graph
cli.py                     # new Typer entrypoint
pyproject.toml             # new: deps + console script `resume`
outputs/                   # generated resume_<company>.typ + .pdf (gitignored)
docs/project.md            # rewritten for the new data flow
```

Deleted: `bank/academics.yaml` (renamed), `bank/internships.yaml`.
`selected.json` is no longer written by the agent. `.gitignore` added
(outputs/, venv, __pycache__).

## Content bank

Two bank files feed the agent; internships and achievements are static in
`main.typ`.

**`bank/projects.yaml`** (unchanged) — one entry per project:
`id`, `title`, `date`, `tags`, `bullets.full`, `bullets.short`.

**`bank/education.yaml`** — one entry per education level, renamed from
`academics.yaml`, restructured:

```yaml
- id: sslc | puc | engineering
  title: "..."
  tier: sslc | puc | engineering
  date: "..."
  tags: [...]
  bullets:
    full: ["..."]
```

Plus optional `company` (degree name) used by the engineering entry, which
renders as the subtitle. SSLC and PUC entries are placeholders with a
`# TODO` comment; engineering keeps the real B.E. entry (institution as
`title`, degree as `company`, CGPA + coursework bullets).

## Templating mechanism

`main.typ` is the visual template. Changes:

1. The line `#let data = json("selected.json")` becomes a marked line:
   ```
   // @@DATA@@ — replaced by the generator with selected entries as an inline dict.
   #let data = @@DATA@@
   ```
   `main.typ` is not expected to compile as-is; the generated file is the
   compilable artifact.
2. Education section becomes `#section("education", "Education")` —
   rendered from the embedded data (the tier decision determines which
   education entries are present).
3. Projects section stays `#section("projects", "Projects & Experiences")`.
4. Internships: the `#section("internships", "Internships")` line is
   replaced with literal, clearly-marked placeholder markup using the
   existing `#resumeHeading` helper (user fills in later).
5. All other static sections (Header, Profile, Technical Skills,
   Achievements & Certifications, Extracurricular Activities) unchanged.
   The `resumeHeading`, `subtitleOf`, and `section` helpers are reused.

**Data embedded at `@@DATA@@`** — a Typst dict literal, e.g.:

```typst
#let data = (
  "projects": (
    "stutter-detection-app": (
      "title": "Cadence — Stutter Detection & Classification App",
      "date": "2025",
      "tags": ("ml", "speech-processing", "desktop-app", "ui-design"),
      "bullets": ("...", "..."),
    ),
  ),
  "education": (
    "engineering": (
      "title": "...",
      "company": "...",
      "date": "...",
      "tags": ("education", "..."),
      "bullets": ("...", "..."),
    ),
  ),
)
```

**Python serializer** (small helper in `resume_agent.py`):
- dict → `(key: value, ...)`, keys always quoted as strings
- list/tuple → `(v1, v2, ...)`
- str → `"..."` with backslash and quote escaping
- optional empty fields omitted (no nulls/empty arrays in the literal)

`generate_typst_file` reads `main.typ`, replaces the `@@DATA@@` marker,
writes `outputs/resume_<company>.typ`. Fails loudly if the marker is
missing from `main.typ`.

## Agent graph (`resume_agent.py`)

Nodes:

| Node | What it does |
|---|---|
| `select_projects` | LLM call. Given JD + projects index (id/title/tags), returns relevant project ids. Validates returned ids against the bank (drops unknown ids). |
| `select_education` | LLM call. Decides a tier from JD seniority/domain: `"full"` (SSLC + PUC + Engineering) or `"engineering_only"` (Engineering only). Maps tier → selected education entry ids. |
| `draft_rewrite` | LLM call. Rewrites bullets for each selected project + education entry to mirror JD phrasing, constrained to each entry's original facts + tags. On retry, feeds back the specific violations from the previous attempt. |
| `verify` | Deterministic, no LLM. Extracts numbers + tagged skill tokens from original vs. rewritten bullets; anything new is a violation. Logic unchanged. |
| conditional edge | `violations and retry_count <= MAX_RETRIES (2)` → `draft_rewrite`; otherwise proceed. |
| `generate_typst_file` | Builds the data dict from selected entries (entries with unresolved violations fall back to their original bullets), serializes to a Typst literal, inlines into a copy of `main.typ`, writes `outputs/resume_<company>.typ`. |
| `compile_typst` | `typst compile outputs/resume_<company>.typ` → `outputs/resume_<company>.pdf`. |

**`ResumeState`** (extends current shape): `job_description`, `company`,
`bank`, `selected_projects`, `education_tier`, `selected_education`,
`drafts`, `violations`, `retry_count`, `output_typ`, `output_pdf`.
(`resume_data`/`pdf_path` replaced.)

**Human-review readiness:** nodes stay discrete; a `human_review` node
using LangGraph's `interrupt()` + a checkpointer can later be inserted
between `generate_typst_file` and `compile_typst` without a redesign.
`compile_typst` stays a separate node; the checkpointer isn't added now.

## CLI (`cli.py`, Typer)

```
resume tailor <job_description.txt> [--company NAME]
resume bank list [--section projects|education]
```

- `tailor`: loads JD text + bank, compiles the graph, streams
  `app.stream()` with `stream_mode="updates"`, printing per-node progress:
  "Selecting projects...", "Deciding education tier...", "Drafting
  rewrite...", "Violation found, retrying (1/2)...", "Writing
  resume_<company>.typ...", "Compiling...". `--company` defaults to the JD
  filename stem; sanitized (lowercased, alnum + dashes) for the output
  filename. Prints the final PDF path.
- `bank list`: loads the bank YAML(s) and prints id/title/tags per entry;
  prints both sections when `--section` is omitted.

## Docs

- `docs/project.md` rewritten: two-file bank, tier-based education,
  per-company standalone `.typ` output (inline-data templating), CLI usage,
  removal of internships bank and `selected.json`.
- `README.md` updated to match (the modified-but-uncommitted README now
  describes the old flow).

## Verification

- `python -m py_compile resume_agent.py cli.py`
- Create a fresh venv; `pip install -e .` (installs langgraph, anthropic,
  pyyaml, typer, and the `resume` console script).
- `resume bank list` prints entries for both sections.
- Smoke-test the serializer + `generate_typst_file` with a canned state
  (no LLM): generated file contains the inline data literal and compiles
  with `typst compile`.
- Full `resume tailor` run requires a real JD and `ANTHROPIC_API_KEY`
  (user provides).

## Out of scope

- A `human_review` node with a checkpointer (only the seam for it is kept).
- Real internship and SSLC/PUC content (placeholders).
- Automated pytest suite (repo has none; verification is py_compile +
  manual compile smoke tests, matching prior practice).
- Rich terminal output (plain print for v1; Rich optional later).
