# PE–MIPS spine study

Research code for *Private Equity Acquisition and Performance in the Merit-Based Incentive Payment System Among Spine Surgeons*.

The study includes 127 acquired surgeons across 24 parent organizations and 2,258 distinct comparison surgeons. The primary weighted stacked difference-in-differences estimate is 4.71 MIPS points (95% CI 0.17–9.25; P=.043). Service ascertainment spans 2013–2022 and performance ascertainment spans 2017–2024. Both surgical specialties require documented operative spine billing before acquisition.

## Reproduction

Use Python 3.12 and install `requirements.txt`. Supply the authorized prepared model inputs described in `INPUT_SCHEMAS.json`, then run:

```sh
python run_analysis.py
python run_supporting.py
python verify_outputs.py
```

The prepared primary stack and supporting context are available from the corresponding author on reasonable request, subject to source terms. These runners reproduce model fitting from the prepared inputs. They do not download the national CMS files or reconstruct the transaction and affiliation ledger from proprietary records.

`analysis/` implements the fixed effects, comparison weights, and practice-cluster CR1/t inference. Supporting analyses cover five secondary outcomes, specialty-specific estimates and a pooled interaction, baseline balance and missingness, thirteen robustness comparisons, and all parent omissions. Each restriction reconstructs eligibility and weights. The operative code list is in `config/`. Aggregate reference outputs are in `expected/`; generated outputs are written to `results/`. Verification compares numerical results at absolute tolerance 1e-9.

## Data access

CMS provider/service, performance, and affiliation data are publicly available subject to source terms. Transaction decisions and the professional-affiliation index require appropriate access. This repository distributes research software, schemas, and aggregate results. It does not redistribute clinician-level source records, model arrays, licensed transaction material, or third-party affiliation data. No source-data redistribution rights are granted by this release.

## Citation

Bachu A, Rajesh R, Germanwala A. Private Equity Acquisition and Performance in the Merit-Based Incentive Payment System Among Spine Surgeons. Analysis code. 2026.
