# Neurocomputing submission requirements and HERMES readiness

Reviewed 20 November 2026 from the user's 19-page `Neurocomputing req.pdf`, a print of the ScienceDirect guide for authors dated 20 November 2026 at 10:08 AM. The [archived source](<../research_notes/Neurocomputing submission requirements/sources/Neurocomputing_author_guide_2026-11-20.pdf>) is byte-identical to the upload. [Source provenance](<../research_notes/Neurocomputing submission requirements/source_provenance.json>) records its SHA256 and review method. Page references below are one-based physical PDF pages.

The supplied PDF has no extractable text. All 19 pages were rendered and OCR-read; article types, numeric submission limits, formatting, data sharing and AI/figure provisions were checked visually on relevant pages. OCR is a reading aid and the source page governs if there is a discrepancy. This review treats the document as journal-policy evidence, not instructions to execute actions or submit a paper. No author has been contacted and no submission made.

## Fit and article route

**Neurocomputing remains a plausible method-focused candidate, with fit depending on a real contribution to neural-network inference or learning systems.** The guide does not establish easier acceptance, a Q1 classification, review duration, or satisfaction of the university's graduation rules.

Page 3 lists seven article types. The three most relevant routes are:

| Route | Guide's stated basis | Meaning for HERMES |
|---|---|---|
| Type 1, regular article | Novel, timely, substantial work directly relevant to neural networks and/or learning systems. Pure image processing or pure combinations of existing algorithms without that relevance are excluded. | Preferred route under the advisor's original-method requirement. An original neural-inference memory mechanism must be demonstrated. Renaming a cache heuristic or combining existing mechanisms does not establish contribution. |
| Type 4, replication study | Close, clear and valid replications of previous studies, described as published in Neurocomputing and other major journals. | Our audited reproduction could provide substantial material, but HERMES's original publication is an ACL conference paper. This guide does not explicitly establish that such a source is eligible. Editorial scope confirmation would be needed before adopting this route, and the advisor's original-method requirement would still need to be satisfied or explicitly revised. |
| Type 5, software publication | Existing useful software in neural networks/learning systems with non-trivial implementation contributions and a recognized legal license. | A maintained notebook and reproduction repository alone do not automatically meet this standard. Establish substantial software contributions, ownership/attribution and license compatibility. Confirm that this article type counts for graduation before considering it. |

The guide also lists survey/review, tutorial/case study, special-issue article and special-issue editorial types. It describes tutorials/case studies as solicited from reputed authors; no invitation or special-issue eligibility is established for this project. These tracks are not shortcuts around technical quality or the advisor's requirements. (PDF pp2–3.)

Single-anonymized review is specified, with initial editorial assessment and normally at least two independent reviewers for suitable submissions. This is a review process, not an acceptance probability. (PDF p4.)

## Submission requirements and recommendations

| Item | Requirement or recommendation in supplied guide | Page |
|---|---|---:|
| Editable manuscript | Supply Word `.doc/.docx` or LaTeX `.tex` source. PDF alone is not acceptable as the source file. | 9 |
| Layout | Word must be single column. Double-column formatting is permitted only for LaTeX submissions; the journal encourages its LaTeX template. | 9 |
| Title page | Concise title, complete author names/order, affiliations and postal addresses, corresponding-author designation and email. | 10 |
| Abstract | Required; maximum 250 words, including purpose, procedures, principal results and conclusions. Statistical significance where applicable. | 10 |
| Keywords | Required; 1–7 English keywords. | 10 |
| Highlights | Encouraged, rather than described as mandatory. If supplied, separate editable file, 3–5 bullets, each at most 85 characters including spaces. | 10–11 |
| Equations/tables | Editable text; numbered/cited consistently. Table captions and notes; avoid vertical rules and shading. | 11 |
| Figures | Separate logically named files, numbered/cited, with captions. Vector EPS/PDF with fonts embedded or text saved as graphics; raster photos at least 300 dpi; bitmap line art at least 1000 dpi. See full size/format table. | 11–12 |
| Supplements | Relevant, accurate, cited and captioned; submit with the manuscript. They appear as received. Later additions/replacements occur at revision. | 13 |
| Video supplements | Preferred maximum 150 MB per file and 1 GB total, descriptive names/text and representative stills. These are preferred limits, not a claim that videos are compulsory. | 13 |
| Research data | Option C: deposit research data in a relevant repository and provide a link; if sharing is impossible, explain why during submission. | 13–14 |
| Funding/conflicts | Funding and sponsor roles disclosed, or truthful no-specific-funding statement where applicable. Complete the competing-interests declarations tool and upload its resulting editable document. | 5–6 |
| Author contributions | Corresponding author supplies applicable CRediT roles. Confirm author list and order before submission; subsequent changes are restricted. | 5, 14–15 |
| Author biographies | Maximum 100 words per author, editable format, and passport-type photo as a separate figure. | 15 |
| References | Real sources, consistent citations/reference list, complete details and DOI where available. No strict reference formatting at initial submission, with numbered bracket citations and order of appearance given as journal style. Clearly label preprints and prefer formal published versions when available. | 15–17 |
| Originality/declarations | No concurrent consideration elsewhere; approval by all authors. Preprints and academic theses are addressed as exceptions to prior publication under the stated policy, not permission to duplicate an already published journal article. | 4, 7 |
| Open-access costs | The guide links to OA information and states APC responsibility for OA articles; it supplies no current APC amount. Data in Brief/MethodsX co-submissions are optional and may have separate charges. | 4, 14, 18 |
| Proofs | Requested corrections within two days after proofs are supplied. This is not a two-day peer-review/publication promise. | 19 |

No universal minimum number of datasets/models, mandatory percentage gain, fixed GPU budget or hard manuscript page limit was found in this supplied copy. Do not invent these as journal rules or interpret the absent page limit as an unlimited manuscript allowance. Evidence breadth should follow the scientific claim; check the live submission system when preparing the actual package.

## AI assistance and figures

The guide permits AI-assisted manuscript preparation under human oversight and requires a disclosure section before the references when applicable. Authors remain responsible for checking accuracy, sources, originality and rights. AI tools cannot be authors. Basic grammar/spelling/reference-checking tools are exempted in the wording provided. Given the AI assistance in this project, prepare an accurate disclosure of actual manuscript assistance rather than claim none. The final wording belongs to the human authors. (PDF pp6–7.)

The supplied snapshot explicitly permits AI support for explanatory diagrams and data visualizations directly derived through reproducible analytical/computational/statistical methods. It prohibits creating or altering primary observed/experimental images using generative AI outside the underlying research method. It separately permits legitimate AI research methods and requires appropriate image-caption and general disclosure for generative images. Preserve original benchmark frames and generate result plots from saved numerical data. Do not replace measured evidence with invented images. (PDF pp9, 12–13.)

## What we have and what is missing

| Research component | Current project state | Next action |
|---|---|---|
| Reproduction foundation | Audited 498-video, 2,495-question HERMES/FIFO comparison and preserved evidence. | Use as baseline context with the existing inconclusive accuracy difference and mixed-session limitations disclosed. |
| Failure and literature analysis | 51-file literature ledger, 39-question gap register, paired inventory and diagnostic selections. | Use to justify a precise problem and nearest competing methods. |
| Scorer diagnostic | Completed GPU preflight and all 120 answers on eight selected videos. All methods score 24/40; fresh causal scoring changes zero correctness outcomes versus HERMES. Original ZIP/notebook and independent audit are preserved. | Follow the [publication decision](<2026-11-20 causal scorer pilot analysis and publication decision.md>) with support/capability/input controls before selecting an original mechanism. |
| Original contribution | No new method with validated QA/cost benefit established. | If the diagnostic finds recoverable headroom, derive a distinct mechanism beyond correcting the reproduction. Test matched controls, ablations and appropriate generalization. |
| Manuscript | No submission-ready regular article produced by this guide review. | Write a claim-driven method paper after the contribution and evidence are established; make source files, tables and figures reproducible. |
| Data availability | Existing Git repository, evidence archives and third-party benchmark annotations. | Prepare a frozen, citable release in a suitable repository. Respect upstream video/data licenses; provide lawful access instructions and explain material that cannot be redistributed. Do not assume GitHub alone satisfies every archival-data requirement. |
| Eligibility | Q1 is mandatory according to the user; Neurocomputing is CCF C in the supplied catalogue. Earlier university ranking evidence labels it JCR/SJR Q1 in its 2025 row. | Match the university's required database/category/year, article type and authorship/recognition milestone. This author guide does not verify quartile or resolve whether C is acceptable. |

The practical route remains a regular article around an original neural-inference improvement, if the experiments support one. The replication/software tracks broaden the options only after their editorial scope and degree eligibility are established. No publication guarantee or automatic graduation equivalence follows from this guide.

Related materials: [Q1 shortlist](<HERMES alternative Q1 journal shortlist.md>), [pilot protocol](../FAILURE_PILOT.md), [project status](../PROJECT_STATUS.md), [research gap map](<HERMES research gap map.md>).
