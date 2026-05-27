# Manuscript — IP&M MMLowRes Special Issue submission

Working title: *Tonal-Perceptual Caching for On-Device Speech Translation of Vietnamese: A Typology-Aware Fingerprint with Two-Stream Voting*

## Files

- `main.tex` — manuscript skeleton (Elsevier `cas-dc.cls`, single-column, double-spaced for initial submission)
- `references.bib` — bibliography database (placeholder entries; verify before camera-ready)
- `cas-dc.cls`, `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` — Elsevier CAS template files (copied from `template/` for local pdflatex)
- `template/els-cas-templates/` — full upstream template bundle as downloaded from CTAN, kept for reference (sample tex, docs, figs, manifest)

## Compile

Standard pdflatex + bibtex cycle (CAS class is `natbib`-based, not biblatex):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Run pdflatex **twice** after bibtex to resolve cross-refs and citation labels (per CLAUDE.md global preferences).

If you use latexmk:

```bash
latexmk -pdf main.tex
```

## Double-blind status

All author identifying information is **redacted** in `main.tex` (author name, affiliation, email, ORCID, acknowledgments, GitHub URLs). The `cortext`/`fntext` blocks contain placeholders. Restore real author info **before camera-ready** but **after** acceptance.

Anonymization checklist before each IP&M upload:

- [ ] `\author{}` blocks contain `Anonymous Author`
- [ ] `\affiliation{}` is empty or `Anonymous Institution`
- [ ] `\ead{}` is empty or `anonymous@example.org`
- [ ] No GitHub / Zenodo / personal URLs in body text or figures
- [ ] No acknowledgments section
- [ ] `references.bib` entries with self-citations marked or anonymized

## Section structure

Matches `papers/paper_outline.md`:

1. Introduction
2. Related Work (six subsections)
3. Method (ten subsections — fingerprint design, two-stream voting, INT8 substrate)
4. Experiments (nine subsections — ablations, minimal-pair, latency, scalability)
5. Deployment Notes
6. Limitations and Negative Results (five honest-failure subsections)
7. Conclusion
- Appendix A–D

## Template source

Downloaded from CTAN mirror on 2026-05-27:
`https://mirrors.ctan.org/macros/latex/contrib/els-cas-templates.zip` (3.3 MB).

`template/els-cas-templates/cas-dc-template.tex` is the upstream skeleton this manuscript was derived from. `cas-dc-sample.tex` and the rendered `cas-dc-sample.pdf` show the expected output formatting.

## Submission portal

- Journal: Information Processing & Management
- Special Issue code: `VSI: MMLowRes`
- Portal: https://submit.elsevier.com/IPM
- Window: 2026-04-31 to 2027-08-31 (rolling)
- Target submit: Q1–Q2 2027
