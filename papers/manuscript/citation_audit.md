# Citation Audit — references.bib (IP&M MMLowRes submission)

Date: 2026-05-28. Tool: WebFetch to arXiv abstract pages, OpenAlex API. Semantic Scholar and ACL Anthology endpoints blocked at fetch time.

## Per-entry verification log

| Key | Source verified at | Status | Fixes applied |
|---|---|---|---|
| conneau2022fleurs | arXiv 2205.12446 + OpenAlex (DOI) | OK | none |
| ctranslate2 | n/a (github) | unverified | comment |
| delafuente2024tone | arXiv 2408.13678 | OK | none (Antón confirmed) |
| edgeasr2025 | arXiv 2507.07877 | OK | none |
| fade2026 | arXiv 2601.02455 | OK | none (year=2026 matches prefix) |
| fasterwhisper | n/a (github) | unverified | comment |
| frantar2023gptq | arXiv 2210.17323 | OK | none (ICLR 2023, year=2023 OK) |
| ggml2023 | n/a (github) | unverified | comment |
| le2024phowhisper | arXiv 2406.02555 | OK | none (Le, Nguyen LT, Nguyen DQ order confirmed) |
| le2025privacyios | arXiv 2505.07583 | OK | none (withdrawn 2025-08-22 confirmed) |
| lin2024awq | arXiv 2306.00978 | OK | none (MLSys 2024 confirmed) |
| luong2016vivos | OpenAlex | partial | comment (pages 51–55 from existing entry, not re-confirmed) |
| nllb2022 | arXiv 2207.04672 | OK | none ({NLLB Team} + Costa-jussà convention confirmed) |
| osakuade2026lexicaltone | arXiv 2604.07467 | OK | none (year=2026 matches prefix) |
| ondevice_streaming_asr_2026 | arXiv 2604.14493 | OK | none |
| qwen3asr2025 | n/a (corporate report) | unverified | comment |
| seallms_audio2025 | arXiv 2511.01670 | OK | none |
| seamlessm4t2023 | arXiv 2308.11596 | OK | none ({Seamless Communication} convention confirmed) |
| taq2025 | arXiv 2511.06516 | FIXED | replaced "others" with full author list (Lapid, Himelstein, Baskin, Shwartz Ziv, Mendelson); "LeVi" capital V confirmed |
| tao2024toneunit | arXiv 2406.08989 | OK | none |
| vietasr2025 | arXiv 2505.21527 | OK | none (Interspeech 2025 acceptance noted) |
| wang2003shazam | n/a (ISMIR 2003, fetch blocked) | partial | comment; well-known canonical reference |
| xiao2023smoothquant | arXiv 2211.10438 | OK | none (ICML 2023 confirmed) |
| xu2025mixedprecision | arXiv 2501.03643 | OK | none (ICASSP 2025 confirmed) |
| bertinmahieux2011msd | n/a (ISMIR 2011, fetch blocked) | partial | comment; well-known canonical reference |
| kvcache2024 | arXiv 2402.02750 | OK | none (KIVI author list confirmed) |
| post2018call | OpenAlex (DOI 10.18653/v1/W18-6319) | FIXED | added DOI |
| thiemann2013demand | OpenAlex (Thiemann/Ito/Vincent, ICA 2013) | OK | comment noting JASA companion DOI |
| snyder2015musan | arXiv 1510.08484 | OK | none |
| pratap2024mms | arXiv 2305.13516 | OK | none (title exact) |
| tiedemann2020tatoeba | OpenAlex (DOI 10.18653/v1/2020.wmt-1.139) | FIXED | added DOI |

## Summary

- 31 entries audited.
- 24 fully verified via arXiv or OpenAlex.
- 3 fixes applied: (1) `taq2025` author list expanded, (2) DOI added to `post2018call`, (3) DOI added to `tiedemann2020tatoeba`.
- 4 entries marked `unverified` (software / corporate reports with no DOI / arXiv).
- 3 entries marked `partial-verification` (pre-2012 ISMIR / WLSI papers — direct API fetch returned no match; metadata left as-is and flagged inline).

## Notes for reviewers / future passes

- **Anton vs Antón**: bib uses `Ant\'on` LaTeX accent — renders to "Antón", correct per arXiv author block.
- **NLLB Team / Seamless Communication**: braced corporate authors are standard for these papers; preserve braces or BibTeX will reformat.
- **`le2025privacyios` withdrawn**: paper exists on arXiv but was withdrawn by the author. Cited for framing only; consider whether to keep — reviewers may flag a withdrawn citation. Flagged in the entry `note`.
- **arXiv future-dated IDs** (`fade2026`, `osakuade2026lexicaltone`, `ondevice_streaming_asr_2026`): the YYMM prefix encodes the submission month, so year=2026 is correct.
- **`thiemann2013demand` DOI 10.1121/1.4799597**: this is the JASA Proc. Meetings on Acoustics record companion to the ICA 2013 paper. Several DEMAND-citing papers use the Zenodo DOI instead (10.5281/zenodo.1227121); JASA is preferred for a print-journal citation.
