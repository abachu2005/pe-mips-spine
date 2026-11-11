# PE–MIPS spine study

Research code for *Private Equity Acquisition and Performance in the Merit-Based Incentive Payment System Among Spine Surgeons*.

## Study
Neurosurgeons and orthopedic surgeons qualify through a published operative spine-service record in a calendar year preceding acquisition. Service ascertainment spans 2013–2022; MIPS performance spans 2017–2024. The primary analysis fits a weighted stacked difference-in-differences model of MIPS final score and an event study in the same cohort. Supporting analyses examine quality score, positive payment adjustment, specialty-specific estimates, baseline balance, seven robustness checks, parent-organization omissions, and complete follow-up. The code list is in `config/procedure_codes.json`.

## Setup and execution
Use Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`.

Supply the authorized inputs described in `INPUT_SCHEMAS.json`, then run:

```sh
python run_analysis.py
python run_supporting.py
python verify_outputs.py
```

`study/scripts/` constructs the cohort; `study/analysis/` fits the model. `run_supporting.py` runs the secondary and robustness analyses with the same operative eligibility. Results are written to `study/results/`. `expected/` contains aggregate reference outputs. `verification/` summarizes numerical reproduction. The verification script compares numeric outputs with tolerance 1e-9. No patient- or clinician-level source files are included.

## Inputs and access
CMS provider/service, performance, and affiliation files are available from CMS subject to source terms. The professional-affiliation index, transaction decisions, and organization-matching inputs require appropriate source access. Input schemas document file structure; they do not provide those restricted records. Exact cohort reproduction requires the authorized inputs. No source-data redistribution rights are granted by this release.

## Software rights
Software remains subject to the authors’ applicable rights. This repository does not grant a license to redistribute restricted source data.

## Citation
Bachu A, Rajesh R, Germanwala A. Private Equity Acquisition and Performance in the Merit-Based Incentive Payment System Among Spine Surgeons. Analysis code. 2026.
