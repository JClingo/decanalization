# Corrected rerun analysis and paper update — 17 October 2026

This is the current primary analysis. It supersedes the legacy primary Results in `paper-update-2026-10-17` while retaining those files as historical evidence. All four selected runs passed snapshot, schema, completion, timing, and scene-integrity checks. The original manuscript was not overwritten.

Read [Results](RESULTS.md), [proposed manuscript edits](MANUSCRIPT_EDITS.md), and [consolidated changes and hyperparameters](CHANGE_INVENTORY.md).

| Condition | Selected run |
| --- | --- |
| HRL-CDR | HRL-CDR-6C224C4240886C1BAEB3EF9EFCE7DE7E |
| HRL-CCC | HRL-CCC-A137EBFA4FB0CC61D2F3D6AD20FCE332 |
| NF-CDR | NF-CDR-530BED5D490A8446975BA4A62D01FF8B |
| NF-CCC | NF-CCC-231FD6EC46C680D5E2717183D5BA294C |

`inputs/` contains the exact YAML, metrics sidecar, completion receipt, loaded snapshots, and compressed scene records for every selected run. `selection.json` is the explicit dataset contract; `validation.json` records checks and baseline alignment. `trial_metrics.csv` is the analysis-ready long table. `evidence/` retains the original logger, versioned executed logger sources, file-write recovery tests, and selected training manifests. The NF runs use a bounded file-write retry; reward and state calculations are unchanged. `paper-baseline/` freezes the manuscript source used for the proposals. `historical-audit/` is superseded context, and `legacy-supplement/` is deliberately separate from the corrected primary measurements.

Reproduction: with Python 3.12, numpy, scipy, PyYAML, and matplotlib, run `build_analysis.py`, `build_supplement.py`, and `write_report.py` from this folder. The scripts also detect the project's `Saved/AnalysisPython` installation. `prepare_inputs.py` is only needed when collecting selected runs in the original workspace. `package_results.py` refreshes the file-hash manifest and ZIP. Do not use the old analysis script or mix schema-1 and schema-2 measurements.

The two earlier NF-labeled runs that loaded HRL weights are not included. Switching the scoring flag is not sufficient to select an NF-trained policy; snapshot identity is checked explicitly. The accepted rerun triplets match HRL 10000 and NF 5000.
