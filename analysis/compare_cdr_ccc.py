"""Compare current CDR/CCC YAML and the previously saved CDR results.

Run: python analysis/compare_cdr_ccc.py
Uses the dependencies in analysis/requirements.txt.
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

from analyze_cdr import METRICS, holm, table

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/HRL-CDR-vs-CCC-10000'
TOTALS = ['reward-total', 'loss-total', 'gain-total', 'gains-harvested']


def load(path, order, reward_function='HRL'):
    rows, seen = [], set()
    expected = ['canalized', 'decanalized', 'recanalized'] if order == 'cdr' else ['canalized']*3
    for trial in yaml.safe_load(path.read_text(encoding='utf-8'))['trials']:
        if trial['id'] in seen or trial['sequence-order'] != order or trial['reward-function'] != reward_function:
            raise ValueError(f'Invalid trial in {path}: {trial["id"]}')
        seen.add(trial['id'])
        phases = sorted(trial['sequences'], key=lambda s: s['sequence-idx'])
        if [s['sequence-idx'] for s in phases] != [1, 2, 3] or [s['state'] for s in phases] != expected:
            raise ValueError(f'Invalid phases in {path}: {trial["id"]}')
        for phase in phases:
            if phase['duration'] <= 0 or not all(np.isfinite(phase[m]) for m in METRICS):
                raise ValueError('Nonfinite metric or invalid duration')
            rows.append(dict(trial_id=trial['id'], **phase))
    return pd.DataFrame(rows)


def contrast(a, b, paired=False):
    """Difference a minus b, with explicit constant-data handling."""
    a, b = np.asarray(a), np.asarray(b)
    if paired:
        delta = a-b
        diff, se, dof = delta.mean(), stats.sem(delta), len(delta)-1
    else:
        diff = a.mean()-b.mean()
        va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
        se = np.sqrt(va+vb)
        dof = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if se else np.inf
    p = 2*stats.t.sf(abs(diff/se), dof) if se else (1. if diff == 0 else 0.)
    margin = stats.t.ppf(.975, dof)*se
    return dict(difference=diff, ci95_low=diff-margin, ci95_high=diff+margin, p=p)


def adjusted(rows):
    frame = pd.DataFrame(rows)
    frame['p_holm'] = holm(frame.p)
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {name: ROOT / f'data/HRL-{name}-10000.yml' for name in ['CDR', 'CCC']}
    frames = {name: load(path, name.lower()) for name, path in paths.items()}
    if any(f.duration.nunique() != 1 for f in frames.values()) or frames['CDR'].duration.iloc[0] != frames['CCC'].duration.iloc[0]:
        raise ValueError('Current files must have equal, constant phase durations')
    summary, comparisons, changes, sensitivity, baseline = [], [], [], [], []
    for name, frame in frames.items():
        frame.assign(dataset=name).to_csv(OUT / f'{name.lower()}_sequences.csv', index=False)
        for metric in METRICS:
            for phase in [1, 2, 3]:
                x = frame.loc[frame['sequence-idx'] == phase, metric].to_numpy()
                margin = stats.t.ppf(.975, len(x)-1)*stats.sem(x)
                summary.append(dict(dataset=name, metric=metric, phase=phase, n=len(x),
                                    mean=x.mean(), sd=x.std(ddof=1), median=np.median(x),
                                    ci95_low=x.mean()-margin, ci95_high=x.mean()+margin))
    for metric in METRICS:
        wide = {name: f.pivot(index='trial_id', columns='sequence-idx', values=metric).sort_index()
                for name, f in frames.items()}
        aligned = wide['CDR'].index.equals(wide['CCC'].index)
        if aligned:
            delta = wide['CDR'][1]-wide['CCC'][1]
            baseline.append(dict(metric=metric, exactly_equal_trials=int((delta == 0).sum()),
                                 n=len(delta), max_absolute_difference=abs(delta).max()))
        for phase in [1, 2, 3]:
            comparisons.append(dict(metric=metric, phase=phase,
                                    **contrast(wide['CDR'][phase], wide['CCC'][phase])))
            if aligned:
                sensitivity.append(dict(metric=metric, phase=phase,
                                        **contrast(wide['CDR'][phase], wide['CCC'][phase], paired=True)))
        for phase in [2, 3]:
            a = wide['CDR'][phase]-wide['CDR'][1]
            b = wide['CCC'][phase]-wide['CCC'][1]
            changes.append(dict(metric=metric, phase=phase, cdr_change=a.mean(), ccc_change=b.mean(),
                                **contrast(a, b)))
    summary = pd.DataFrame(summary)
    comparisons, changes, sensitivity = map(adjusted, [comparisons, changes, sensitivity])
    summary.to_csv(OUT / 'phase_summary.csv', index=False)
    comparisons.to_csv(OUT / 'between_runs_welch.csv', index=False)
    changes.to_csv(OUT / 'baseline_adjusted_changes.csv', index=False)
    sensitivity.to_csv(OUT / 'paired_id_sensitivity.csv', index=False)
    pd.DataFrame(baseline).to_csv(OUT / 'baseline_alignment.csv', index=False)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for name, color, shift in [('CDR', '#d46a35', -.04), ('CCC', '#277f91', .04)]:
            s = summary[(summary.dataset == name) & (summary.metric == metric)].sort_values('phase')
            ax.errorbar(s.phase+shift, s['mean'],
                        yerr=[s['mean']-s.ci95_low, s.ci95_high-s['mean']],
                        color=color, marker='o', capsize=4, label=name, lw=2)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3], ['Phase 1\nC / C', 'Phase 2\nD / C', 'Phase 3\nR / C'])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend()
    fig.suptitle('Current HRL-CDR vs HRL-CCC: '
                 f'{frames["CDR"].trial_id.nunique()} / {frames["CCC"].trial_id.nunique()} trials\n'
                 'Means and 95% confidence intervals; equal '
                 f'{frames["CDR"].duration.iloc[0]}-unit phases', fontsize=14)
    fig.savefig(OUT / 'overview.png', dpi=180)
    fig.savefig(OUT / 'overview.pdf')
    plt.close(fig)

    # Preserve the user's linked earlier report and compare its actual saved rows.
    saved_path = ROOT / 'results/HRL-CDR-10000/sequences.csv'
    saved = pd.read_csv(saved_path)
    prior_rows = []
    for metric in METRICS:
        for phase in [1, 2, 3]:
            old = saved[saved['sequence-idx'] == phase]
            new = frames['CCC'][frames['CCC']['sequence-idx'] == phase]
            a, b = old[metric], new[metric]
            units = 'as logged'
            if metric in TOTALS:
                a, b = a/old.duration*1000, b/new.duration*1000
                units = 'per 1000 logged duration units'
            prior_rows.append(dict(metric=metric, phase=phase, units=units,
                                   saved_cdr_n=len(a), ccc_n=len(b), saved_cdr_mean=a.mean(),
                                   ccc_mean=b.mean(), **contrast(a, b)))
    prior = adjusted(prior_rows)
    prior.to_csv(OUT / 'saved_cdr_vs_current_ccc.csv', index=False)
    mean = summary.pivot(index=['metric', 'phase'], columns='dataset', values='mean').reset_index()
    final = changes[changes.phase == 3].set_index('metric')
    def result(metric):
        x = final.loc[metric]
        return (f'{x.difference:+.2f} (95% CI {x.ci95_low:.2f} to '
                f'{x.ci95_high:.2f}; Holm-adjusted p={x.p_holm:.3g})')
    report = f'''# HRL-CDR vs HRL-CCC at 10000

## Input versions matter

The linked `HRL-CDR-10000/` report is an earlier analysis: {saved.trial_id.nunique()}
trials, with durations {sorted(saved.duration.unique().tolist())}. The current CDR
and CCC YAML files each have {frames['CDR'].trial_id.nunique()} trials and duration
{frames['CDR'].duration.iloc[0]} per phase. The primary comparison below uses the
current YAML files. The earlier report is preserved. Its requested comparison
against CCC appears separately below and in `saved_cdr_vs_current_ccc.csv`.

## Interpretation

For the current matched-duration files, decanalization produces a large phase-2
disruption, followed by recovery. CCC also changes over time, so comparing CDR's
final phase only to its own baseline overstates evidence for a persistent deficit
relative to uninterrupted canalization. At phase 3, CDR has a lower positive
signed setpoint mean and higher reward than CCC, while pellet collection is
similar. A lower positive signed mean is not proof of lower absolute error.

The final-phase differences after subtracting each trial's baseline are:

- Signed setpoint mean: {result('setpoint-avg')}.
- Raw reward: {result('reward-total')}.
- Total gain: {result('gain-total')}.
- Pellets harvested: {result('gains-harvested')}.

Intervals are unadjusted; p-values use the separate 12-test change-contrast family.
The pellet comparison does not establish equivalence. The evidence supports a
large temporary disruption without a final-phase deficit relative to CCC on
these measures; it does not establish that decanalization improves learning.
The resource trajectory changes too: phase-2 loss is zero in CDR, and final-phase
loss remains more negative in CDR than CCC.

For the **older saved results explicitly linked in the request**, final signed
setpoint mean is 35.42 versus 20.23 in CCC. Final reward rates are 22.07 versus
27.21, and pellet rates are 19.27 versus 26.22 per 1000 logged duration units.
Those differences coexist with substantial baseline differences and unequal
elapsed-time windows, so they cannot isolate an effect of the CDR sequence.
The updated CDR file and the old report must not be treated as the same dataset.

## Current files: phase means

{table(mean, ['metric', 'phase', 'CDR', 'CCC'])}

## Between-run comparisons

Differences are CDR minus CCC. Primary tests use independent-sample Welch t-tests,
since matching IDs alone do not establish paired experimental provenance. Holm
correction covers these 18 comparisons. The 95% CIs are unadjusted.

{table(comparisons, ['metric', 'phase', 'difference', 'ci95_low', 'ci95_high', 'p_holm'])}

## Changes relative to each run's phase 1

Each trial's phase-1 value is subtracted before comparing runs with Welch tests.
The difference is (CDR phase j minus CDR phase 1) minus (CCC phase j minus CCC
phase 1). Holm correction covers these 12 comparisons separately. These contrasts
account for observed baseline differences, but are not proof of a causal effect.

{table(changes, ['metric', 'phase', 'cdr_change', 'ccc_change', 'difference', 'ci95_low', 'ci95_high', 'p_holm'])}

## Trial alignment sensitivity

The files have the same trial IDs. Baseline alignment is shown below. The project
documents reproducible environment streams, but seed/checkpoint/run provenance is
absent from the YAML. `paired_id_sensitivity.csv` therefore reports conditional
paired tests as a sensitivity analysis (18-test Holm family), not the primary test.

{table(pd.DataFrame(baseline), ['metric', 'exactly_equal_trials', 'n', 'max_absolute_difference'])}

## Requested saved CDR results vs current CCC

These observations have unequal phase durations and different sample sizes.
Additive totals are converted to rates per 1000 logged duration units. Setpoint
averages and reported deviation are left as logged. Rate conversion does not
correct duration-dependent dynamics or align elapsed time: saved CDR phase 1 spans
the same logged duration as all three CCC phases. Tests are exploratory Welch
tests (18-test Holm family); this is not a controlled comparison.

{table(prior, ['metric', 'phase', 'units', 'saved_cdr_mean', 'ccc_mean', 'difference', 'p_holm'])}

## Limits

- Signed setpoint averages cannot recover mean absolute error or time within ±5.
- `setpoint-dev` is used as logged; its formula and sampling count are unspecified.
  Its magnitude should not be compared across durations as a verified standard deviation.
- Trial-level inference assumes independent replicates; trials do not establish
  variation across independently trained policies. Equal first phases do not prove equivalence.
- A final-phase difference cannot distinguish carryover, resource changes, recovery
  transients, or learning. Resource gain/loss totals directly differ between conditions.
- Raw rewards are not min-max normalized. A 10000 filename does not verify the
  checkpoint or reward configuration used to produce the run.

## Provenance and reproduction

'''
    for name, path in paths.items():
        report += f'- {name}: `{path}`; SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`\n'
    report += f'- Saved CDR rows: `{saved_path}`; SHA-256 `{hashlib.sha256(saved_path.read_bytes()).hexdigest()}`\n'
    report += '\nRun `python analysis/compare_cdr_ccc.py` with analysis/requirements.txt installed.\n\n![Current comparison](overview.png)\n'
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    print(mean.to_string(index=False))
    print('\nBaseline-adjusted contrasts:\n'+changes.to_string(index=False))
    print(f'\nSaved to {OUT}')


if __name__ == '__main__':
    main()
