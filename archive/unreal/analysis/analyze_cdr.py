"""Analyze complete CDR trials, preserving raw metrics and within-trial pairing.

Run from any directory: python analysis/analyze_cdr.py
Requires pyyaml, numpy, scipy, pandas and matplotlib (analysis/requirements.txt).
"""
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

ROOT = Path(__file__).resolve().parent
STATES = ['canalized', 'decanalized', 'recanalized']
METRICS = {
    'setpoint-avg': 'Signed mean setpoint distance',
    'setpoint-dev': 'Reported setpoint deviation',
    'reward-total': 'Raw total reward',
    'loss-total': 'Total loss (signed)',
    'gain-total': 'Total gain',
    'gains-harvested': 'Pellets harvested',
}


def load_trials(path):
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    rows, seen = [], set()
    for trial in data['trials']:
        key = (trial['reward-function'], trial['sequence-order'], trial['id'])
        if key in seen:
            raise ValueError(f'Duplicate trial: {key}')
        seen.add(key)
        if trial['sequence-order'] != 'cdr':
            raise ValueError(f'Expected CDR trial: {key}')
        phases = sorted(trial['sequences'], key=lambda s: s['sequence-idx'])
        if [s['state'] for s in phases] != STATES or [s['sequence-idx'] for s in phases] != [1, 2, 3]:
            raise ValueError(f'Incomplete or invalid phases: {key}')
        for phase in phases:
            if phase['duration'] <= 0 or not all(np.isfinite(phase[m]) for m in METRICS):
                raise ValueError(f'Invalid measurement: {key}')
            rows.append(dict(trial_id=trial['id'], reward_function=key[0], **phase))
    frame = pd.DataFrame(rows)
    if len(seen) < 2 or frame['reward_function'].nunique() != 1:
        raise ValueError('Expected at least two trials of a single reward function')
    if frame['duration'].nunique() != 1:
        raise ValueError('Phase totals require equal durations')
    return frame


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        (len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    return adjusted


def summarize(df):
    rows = []
    for metric in METRICS:
        for state in STATES:
            x = df.loc[df.state == state, metric].to_numpy()
            sem = stats.sem(x)
            margin = stats.t.ppf(.975, len(x) - 1) * sem
            rows.append(dict(metric=metric, state=state, n=len(x), mean=x.mean(),
                             sd=x.std(ddof=1), median=np.median(x), min=x.min(), max=x.max(),
                             ci95_low=x.mean()-margin, ci95_high=x.mean()+margin))
    return pd.DataFrame(rows)


def paired_tests(df):
    rows = []
    for metric in METRICS:
        wide = df.pivot(index='trial_id', columns='state', values=metric)
        for before, after in [('canalized', 'decanalized'),
                              ('decanalized', 'recanalized'),
                              ('canalized', 'recanalized')]:
            delta = (wide[after] - wide[before]).to_numpy()
            sd, n = delta.std(ddof=1), len(delta)
            margin = stats.t.ppf(.975, n-1) * stats.sem(delta)
            test = stats.ttest_rel(wide[after], wide[before])
            rows.append(dict(metric=metric, before=before, after=after, n=n,
                             mean_change=delta.mean(), ci95_low=delta.mean()-margin,
                             ci95_high=delta.mean()+margin, t=test.statistic,
                             p=test.pvalue, cohens_dz=delta.mean()/sd if sd else np.nan))
    result = pd.DataFrame(rows)
    result['p_holm'] = holm(result.p.to_numpy())
    return result


def plot(df, out):
    colors = ['#277f91', '#d46a35', '#567c45']
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        wide = df.pivot(index='trial_id', columns='state', values=metric)[STATES]
        ax.plot(range(1, 4), wide.to_numpy().T, color='#9ba6ac', alpha=.17, lw=.7)
        boxes = ax.boxplot([wide[s] for s in STATES], patch_artist=True,
                           widths=.42, showfliers=False)
        for box, color in zip(boxes['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(.65)
        for i, state in enumerate(STATES, 1):
            x = wide[state].to_numpy()
            jitter = np.random.default_rng(42).uniform(-.12, .12, len(x))
            ax.scatter(i+jitter, x, color=colors[i-1], s=10, alpha=.6, zorder=3)
            margin = stats.t.ppf(.975, len(x)-1)*stats.sem(x)
            ax.errorbar(i+.29, x.mean(), yerr=margin, fmt='D', color='black',
                        ms=4, capsize=3, zorder=4)
        ax.set_xticks([1, 2, 3], ['Canalized', 'Decanalized', 'Recanalized'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{out.name}: {df.trial_id.nunique()} paired CDR trials\n'
                 'Lines connect trials; black diamonds show means with 95% confidence intervals', fontsize=14)
    fig.savefig(out / 'overview.png', dpi=180)
    fig.savefig(out / 'overview.pdf')
    plt.close(fig)


def table(frame, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in frame.iterrows():
        values = [f'{row[c]:.5g}' if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/HRL-CDR-10000.yml')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / 'results' / args.input.stem
    out.mkdir(parents=True, exist_ok=True)
    df = load_trials(args.input)
    summary, tests = summarize(df), paired_tests(df)
    df.to_csv(out / 'sequences.csv', index=False)
    summary.to_csv(out / 'phase_summary.csv', index=False)
    tests.to_csv(out / 'paired_tests.csv', index=False)
    plot(df, out)
    historical = []
    for name in ['HRL-CDR', 'NF-CDR']:
        path = ROOT / 'data/Old Results' / f'{name}.yml'
        if path.exists():
            historical.append(summarize(load_trials(path)).assign(dataset=f'Archived {name}'))
    historical.append(summary.assign(dataset=args.input.stem))
    history = pd.concat(historical, ignore_index=True)
    history.to_csv(out / 'historical_summary.csv', index=False)
    means = df.groupby('state')[list(METRICS)].mean()
    c, d, r = (means.loc[s] for s in STATES)
    residual = tests[(tests.metric == 'setpoint-avg') &
                     (tests.before == 'canalized') & (tests.after == 'recanalized')].iloc[0]
    report = f'''# {args.input.stem} analysis

Input: `{args.input.resolve()}`  
SHA-256: `{hashlib.sha256(args.input.read_bytes()).hexdigest()}`  
Validated: {df.trial_id.nunique()} complete trials; {len(df)} phase records; all durations {df.duration.iloc[0]}.

## What the results show

**Strong recovery after disruption, with residual differences from the initial phase.**

- Mean signed setpoint distance rises from {c['setpoint-avg']:.2f} to
  {d['setpoint-avg']:.2f}, then falls to {r['setpoint-avg']:.2f}. Recanalization
  reverses {100*(d['setpoint-avg']-r['setpoint-avg'])/(d['setpoint-avg']-c['setpoint-avg']):.1f}%
  of the increase in the group mean. This describes signed phase averages, not
  a percentage recovery of overall homeostatic control.
- The final signed mean remains {residual.mean_change:.2f} units above the initial
  mean (paired 95% CI {residual.ci95_low:.2f} to {residual.ci95_high:.2f};
  Holm-adjusted p={residual.p_holm:.3g}).
- Pellets harvested: {c['gains-harvested']:.2f} → {d['gains-harvested']:.2f} →
  {r['gains-harvested']:.2f}; the final mean is
  {100*(1-r['gains-harvested']/c['gains-harvested']):.1f}% below the initial mean.
- Raw reward: {c['reward-total']:.2f} → {d['reward-total']:.2f} →
  {r['reward-total']:.2f}; the final mean is
  {100*(1-r['reward-total']/c['reward-total']):.1f}% below the initial mean.
- Total gain: {c['gain-total']:.2f} → {d['gain-total']:.2f} →
  {r['gain-total']:.2f}; the final mean is
  {100*(1-r['gain-total']/c['gain-total']):.1f}% below the initial mean.
- Recorded loss is zero in every decanalized phase. Thus the perturbation changes
  resource balance as well as behavior; phase differences do not isolate learning.
- All 18 phase contrasts except initial-versus-final reported setpoint deviation
  pass the Holm-adjusted 0.05 threshold. That deviation comparison has adjusted
  p=0.081; lack of significance does not demonstrate equivalence.

The trajectory is consistent with regulation returning when canalization returns,
but does not by itself establish adaptation or relearning. The phase averages
cannot distinguish an initial recovery transient from a persistent final offset.

## Phase summaries

{table(summary, ['metric', 'state', 'n', 'mean', 'sd', 'median', 'ci95_low', 'ci95_high'])}

## Paired phase changes

Changes are after minus before. Two-sided paired t-tests match trial IDs.
Holm adjustment covers all 18 metric/contrast tests. Confidence intervals are
unadjusted 95% intervals for mean changes; Cohen's dz uses the SD of paired changes.
Inference assumes trials provide independent replicates; the file contains no
seed or policy provenance to verify that assumption. These trials do not measure
variation across independently trained policies. Phase and time/order are confounded.

{table(tests, ['metric', 'before', 'after', 'mean_change', 'ci95_low', 'ci95_high', 'p_holm', 'cohens_dz'])}

## Interpretation limits

- `setpoint-avg` is signed: averaging near zero does not establish small absolute error.
  The phase summaries cannot recover time within ±5 or mean absolute error.
- `setpoint-dev` is plotted exactly as logged; its formula and sampling count are
  not specified in this YAML. Do not assume it is a conventional standard deviation.
- Loss is signed, so more negative totals indicate a larger recorded loss magnitude.
- Rewards remain raw. The legacy sandbox normalizes each dataset independently,
  which removes absolute reward differences. Reward functions/objectives can differ.
- Historical means are descriptive only. Matching YAML durations do not establish
  identical environment settings, simulation timing, reward code or sampling.
- A 10000 filename does not mean 10000 trials: this file has 100 trials and no
  training-checkpoint metadata. Aggregate phases cannot establish recovery speed.

## Archived context

{table(history, ['dataset', 'metric', 'state', 'n', 'mean'])}

![Paired phase overview](overview.png)

## Reproduce

From the repository root, with `analysis/requirements.txt` installed:

```powershell
python analysis/analyze_cdr.py
python analysis/sandbox.py
```

The sandbox reproduces the original comparison against archived NF-CDR, including
its independent t-tests and per-dataset reward normalization. Its outputs are in
`legacy-comparison/`; its statistical results do not establish a controlled HRL/NF
comparison. Use the paired tests above to assess the new HRL run's phase changes.
'''
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(summary[['metric', 'state', 'n', 'mean', 'median']].to_string(index=False))
    print('\nPaired changes (after minus before):')
    print(tests[['metric', 'before', 'after', 'mean_change', 'p_holm']].to_string(index=False))
    print(f'\nSaved results to {out.resolve()}')


if __name__ == '__main__':
    main()
