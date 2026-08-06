#set page(
  paper: "us-letter",
  margin: (x: 0.5in, y: 0.5in),
)

// Set the base font to look professional, similar to LaTeX's default.
#set text(
  font: ("New York", "New Computer Modern"),
  size: 10pt,
)

// Configure link colors to match hyperref colorlinks=true, urlcolor=black
#show link: set text(fill: black)

// Custom section formatting (uppercase, bold, with horizontal rule)
#show heading.where(level: 1): it => [
  #set text(size: 12pt, weight: "bold")
  #v(4pt)
  #upper(it.body)
  #v(-8pt)
  #line(length: 100%, stroke: 0.7pt)
  #v(4pt)
]

// -------------------- DATA --------------------
// This template is never hand-edited per company: the agent copies this
// file and inlines the selected entries as data at the marker below,
// producing a standalone outputs/resume_<company>.typ. This template
// does not compile until that replacement happens.
#let data = @@DATA@@

// Heading helper: bold title with a right-aligned date, and an optional
// italic subtitle line beneath (mirrors \resumeHeading).
#let resumeHeading(title, right, subtitle: none) = block[
  #strong(title) #h(1fr) #emph(right) \
  #if subtitle != none [#emph(subtitle)] \
  #v(3pt)
]

// Subtitle resolution: explicit subtitle field, else company, else tags.
#let subtitleOf(entry) = {
  let s = entry.at("subtitle", default: none)
  if s == none {
    s = entry.at("company", default: none)
  }
  if s == none {
    s = entry.at("tags", default: ()).join(", ")
  }
  s
}

// Render every entry of one section, skipping the heading if empty.
#let section(section, heading) = {
  let entries = data.at(section, default: (:))
  if entries.len() > 0 {
    [= #heading] + block[
      #for entry in entries.values() [
        #resumeHeading(entry.title, entry.at("date", default: ""), subtitle: subtitleOf(entry))
        #for bullet in entry.at("bullets", default: ()) [- #bullet]
        #v(4pt)
      ]
    ]
  }
}

// -------------------- HEADER --------------------
#align(center)[
  #text(size: 24pt, weight: "bold")[SRINIVAS HEGDE M] \
  #v(4pt)
  +91 9060159605 | 
  #link("mailto:hegdesrinivasm+queries@gmail.com")[hegdesrinivasm+queries\@gmail.com] | 
  #link("https://linkedin.com/in/hegdesrinivasm")[LinkedIn] | 
  #link("https://github.com/hegdesrinivasm")[GitHub] | 
  #link("https://hegdesrinivasm.vercel.app")[Portfolio]
]

// -------------------- PROFILE --------------------
= Profile
Passionate final-year AIML engineering student eager to apply machine learning and deep learning expertise to real-world challenges. Hands-on experience building and training neural network models, designing ML pipelines, and translating AI architectures into working systems, thriving in high-energy hackathon environments and driven to deliver measurable impact in efficiency, sustainability, and innovation as an entry-level ML/AI engineer.

// -------------------- EDUCATION --------------------
#section("education", "Education")

// -------------------- SKILLS --------------------
= Technical Skills
- *Industry 6.0 and Emerging Tech:* DevOps, MLOps, Sustainable Engineering Practices, AI/ML for Process Optimizations, LangChain, LangGraph.
- *Core ML tools:* PyTorch, NumPy, Matplotlib, Librosa, Seaborn, Pandas, TensorFlow
- *Tools & Methodologies:* Git, GitHub, Linux, Data Synthesis, Agile Collaboration, Problem Solving, Leadership

// -------------------- PROJECTS ------------------
#section("projects", "Projects & Experiences")

// -------------------- INTERNSHIPS ----------------------
// Internships are static for now — replace the placeholder below with
// real roles, one resumeHeading + bullets block each.
= Internships
#resumeHeading("Role Title", "Month Year – Month Year", subtitle: "Company Name")
- TODO: add responsibilities, impact, and outcomes

// -------------------- ACHIEVEMENTS --------------------
= Achievements & Certifications
- *Coursera:* Completed 'Supervised Machine Learning: Regression and Classification' (Machine Learning Specialization).
- *Winner, Nexthon-2025:* Secured 1st place in a 24-hour national hackathon working in a high-pressure, collaborative team "Luminous" at Shree Devi Institute of Technology, Mangalore.
- *Winner, DevHack-2025:* Secured 1st place in a 36-hour national hackathon at Sahyadri College of Engineering, yet again with the same team "Luminous".

// -------------------- EXTRACURRICULAR LEADERSHIP --------------------
= Extracurricular Activities
- *Chair:* IEEE VCET Student Branch (STB11471).
- *Student Co-ordinator (2025-26):* Ankuram Annual College Magazine Editorial Board.
- *Joint Cultural Secretary (2025-26):* Bhoomika Kala Sangha, Cultural organization of VCET.
- *Member*: Center of Research Excellency (CoRE).
