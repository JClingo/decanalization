# Modeling decanalization with homeostatic reinforcement learning

Unreal Engine simulation, frozen evaluation data, and analysis for the paper by
Joshua Clingo and Jeff Yoshimi. Two selected trained policies (HRL and NF) are
evaluated under CDR (canalization, decanalization, recanalization) and CCC
(uninterrupted canalization). The study has 250 trials per
condition, 1,000 trials and 3,000 phase records in total.

The preprint (since significantly revised) is available [on PsyArXive](https://doi.org/10.31234/osf.io/y4x29_v1).

## Reproduce the paper analysis

No Unreal installation, GPU, or commercial plugin is needed to analyze the data.
Clone this repository, or download and extract its GitHub source ZIP:

```sh
git clone https://github.com/JClingo/decanalization.git
cd decanalization
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r analysis/requirements-paper.txt
python analysis/reproduce.py --paper-figures
```

Use Python 3.12. `--paper-figures` additionally needs R with `Rscript` on PATH
(base R only). Omit the flag to regenerate all statistics, reports, and Python
figures without R. Windows can use `py -3.12 -m venv .venv` for the first command.
Outputs go to `analysis/generated/paper/`; an existing output directory is never
overwritten. Choose another with `--output analysis/generated/another-run`.

To independently verify raw records and primary statistics with Python's standard
library only (no dependency installation):

```sh
python analysis/reproduce.py --verify-only
```

The runner checks every archived file against its SHA-256 manifest, verifies all
scene and snapshot hashes, regenerates outputs in a separate directory, compares
six primary/supplementary CSV tables numerically with the archive, and repeats the
independent primary statistical verification. See [the analysis guide](docs/ANALYSIS.md)
for definitions, provenance, output mapping, and limitations.

## Run or adapt the Unreal simulation

Start with [the Unreal setup and experiment guide](docs/UNREAL.md). The project is
`ue_canalization.uproject`, associated with Unreal Engine 5.5. The implementation
is mainly Blueprint assets, supplemented by Python training and logging helpers.
Engine and Marketplace dependencies must be installed separately. The reference
execution workflow is Windows; statistical analysis runs independently of it.

## Repository map

| Location | Purpose |
| --- | --- |
| `Content/`, `Config/`, `ue_canalization.uproject` | Editable Unreal project |
| `Content/Python/` | Runtime logging, resets, NF trainer selection |
| `analysis/reproduce.py` | Portable paper reproduction entry point |
| `analysis/paper-update-2026-10-17-reruns/` | Frozen corrected data, scripts, results, selected weights, scene records, provenance |
| `analysis/homeostatic/` | HRL/NF objectives, PPO adapters, implementation validation |
| `analysis/trial_metrics/`, `analysis/timing/` | Evaluation protocol and timing/reset validation |
| `analysis/implementation-audit-2026-09-14/` | Historical implementation audit, not current 2026 settings |
| `analysis/paper-update-2026-10-17/`, `analysis/data/`, `logs/` | Earlier analyses and archived logs; superseded for primary results |
| `analysis/generated/` | Local regenerated outputs (ignored by Git) |

The earlier ZIP bundles are historical distribution copies. The unpacked corrected
package above is the source used by the reproduction command. Do not mix legacy
schema-1 scores or dispersion fields with corrected schema-2 measurements.

## Validation of this packaging change

The frozen-file manifest and independent standard-library verification pass:
1,000 trials, 3,000 phases/scenes, 12 loaded snapshot hashes, 108 descriptive rows,
and 144 contrasts. Both R manuscript figures reproduce byte-for-byte.
Full NumPy/SciPy regeneration was attempted with the pinned Python 3.12
dependencies, but this machine's macOS 27 loader rejected SciPy 1.15.3's compiled
PROPACK library on both ARM and x86 before analysis could run.
The historical audit passes all 22 import hashes and 189 archived evidence-file
checks, including 87 native exports. The included
[CI workflow](.github/workflows/reproduce-paper.yml) runs reproduction on Windows,
macOS, and Linux, plus manuscript figure generation on Linux; its results are
pending. A fresh Unreal launch with the external plugins remains untested here.

## Licensing and attribution

Original project code, including original Blueprint logic, is licensed under
[MIT](LICENSE). Author-produced evaluation data are licensed under
[CC BY 4.0](LICENSE-DATA.md). Unreal Engine, plugins, and third-party assets retain
their own licenses; see [license scope](docs/LICENSING.md). Cite the paper when
reusing the study.

The retained raw records identify the selected policies and executed settings.
Repeated evaluations do not establish generalization across independently trained
policies.
