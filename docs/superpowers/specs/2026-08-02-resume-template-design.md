# Data-Driven Resume Template Design

## Goal

Rewrite `main.typ` so Projects, Internships, and Education render from
`selected.json` (written by `resume_agent.py`), while personal-constant
sections (Header, Profile, Technical Skills, Achievements, Extracurricular)
stay static in the template. Extend the agent's `write_resume_data` to emit
the full entry data needed for rich headings.

## Context / current state

- The only unmet requirement in `docs/project.md` is the template layer:
  `main.typ` is fully hand-written and never reads `selected.json`.
- `write_resume_data` currently emits only `title` + `bullets` per entry,
  dropping `date`, `company`, and `tags`.
- `bank/academics.yaml` contains hackathon/IEEE placeholder entries whose
  content already lives statically in `main.typ`; if selected, the template
  would render them under the wrong (Education) section.

## Data contract (`selected.json`)

Per-entry payload emitted by the agent (optional fields omitted when empty):

```json
{
  "projects": {
    "<entry-id>": {
      "title": "...",
      "date": "2025",
      "subtitle": "...",
      "company": "...",
      "tags": ["ml", "speech-processing"],
      "bullets": ["...", "..."]
    }
  },
  "internships": { "...": { "title": "...", "date": "...", "company": "...", "subtitle": "...", "tags": [...], "bullets": [...] } },
  "academics": { "...": { "title": "...", "date": "...", "company": "...", "tags": [...], "bullets": [...] } }
}
```

## Template behavior (`main.typ`)

1. `#let data = json("selected.json")` loaded once at top.
2. Static Header, Profile, Technical Skills, Achievements, Extracurricular
   sections unchanged from the current resume.
3. Education: `= Education` renders `academics` entries — bold title
   (institution), italic company (degree) as subtitle, date on the right;
   CGPA and coursework as bullets.
4. Projects: `= Projects & Experiences` — bold title / date right, subtitle
   resolved as `subtitle` field else joined `tags`, then bullets.
5. Internships: `= Internships` — bold title / date right, subtitle resolved
   as `subtitle` field else `company`, then bullets.
6. Subtitle resolution: explicit `subtitle` field → `company` → joined `tags`.
7. Empty sections render nothing (heading omitted too), via
   `data.at(section, default: (:))` and a `.len() > 0` guard.
8. Preserve existing page setup, fonts, link colors, level-1 heading
   show-rule, and the `resumeHeading` helper (generalized to a two-argument
   title/right form with optional subtitle).

## Agent change (`resume_agent.py`)

`write_resume_data`: emit `date`, `subtitle`, `company`, `tags` alongside
`title` and `bullets`, omitting any optional field that is empty/None so
`selected.json` stays clean (no nulls). Violation fallback to original
bullets is unchanged.

## Bank changes

- `bank/academics.yaml`: education-only. Add a real B.Tech AIML entry
  (institution as `title`, degree as `company`, `date`, `tags`, bullets =
  CGPA + relevant coursework). Remove the hackathon/IEEE placeholder entries.
- Document the optional `subtitle` field in the bank header comments
  (`projects.yaml`, `internships.yaml`).

## Error handling

- Missing `selected.json` → Typst compile error. Acceptable; the agent always
  writes it before compiling.
- Missing optional fields → `.at(field, default: ...)` guards in the template.

## Verification

- `python -m py_compile resume_agent.py`; exercise `write_resume_data` with a
  fake state to confirm the payload shape.
- Compile `main.typ` with a sample `selected.json` covering all three
  sections, and again with a section missing, and confirm PDFs are produced
  without errors.
- Remove test artifacts afterward.
