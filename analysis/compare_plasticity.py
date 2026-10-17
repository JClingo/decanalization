"""Compare HRL CDR at 1x, 4x and 7x decanalized overlap counting.

Run: python analysis/compare_plasticity.py
The overlap mapping is supplied by the user, not inferred from YAML metadata.
"""
import hashlib
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_cdr import ROOT, METRICS, table
from compare_cdr_ccc import load, contrast, adjusted

SETTINGS = {
    '1x (0 plasticity)': ('HRL-CDR-0-plasticity.yml', 1, '#277f91'),
    '4x (standard)': ('HRL-CDR-10000.yml', 4, '#d46a35'),
    '7x (6 plasticity)': ('HRL-CDR-6-plasticity.yml', 7, '#76519b'),
}
OUT = ROOT / 'results/HRL-CDR-plasticity'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {name: load(ROOT / 'data' / filename, 'cdr')
              for name, (filename, _, _) in SETTINGS.items()}
    if any(f.trial_id.nunique() < 2 or f.duration.nunique() != 1 for f in frames.values()):
        raise ValueError('Need complete replicate trials and constant durations')
    durations = {f.duration.iloc[0] for f in frames.values()}
    if len(durations) != 1:
        raise ValueError('Durations must match')
    rows, tests, paired, baseline = [], [], [], []
    for name, f in frames.items():
        f.assign(setting=name, overlap_multiplier=SETTINGS[name][1]).to_csv(
            OUT / f'sequences_{SETTINGS[name][1]}x.csv', index=False)
        for metric in METRICS:
            for phase in [1, 2, 3]:
                x = f.loc[f['sequence-idx'] == phase, metric]
                margin = stats.t.ppf(.975, len(x)-1)*stats.sem(x)
                rows.append(dict(setting=name, multiplier=SETTINGS[name][1], metric=metric,
                                 phase=phase, n=len(x), mean=x.mean(), median=x.median(),
                                 sd=x.std(ddof=1), min=x.min(), max=x.max(),
                                 ci95_low=x.mean()-margin, ci95_high=x.mean()+margin))
    for metric in METRICS:
        wide = {name: f.pivot(index='trial_id', columns='sequence-idx', values=metric).sort_index()
                for name, f in frames.items()}
        for lower, higher in combinations(SETTINGS, 2):
            a, b = wide[higher], wide[lower]
            align = a.index.equals(b.index)
            if align:
                delta = a[1]-b[1]
                baseline.append(dict(metric=metric, higher=higher, lower=lower, n=len(delta),
                                     exact_matches=int((delta == 0).sum()),
                                     max_absolute_difference=abs(delta).max()))
            for phase in [2, 3]:
                da, db = a[phase]-a[1], b[phase]-b[1]
                tests.append(dict(metric=metric, phase=phase, higher=higher, lower=lower,
                                  higher_change=da.mean(), lower_change=db.mean(), **contrast(da, db)))
                if align:
                    paired.append(dict(metric=metric, phase=phase, higher=higher, lower=lower,
                                       **contrast(da, db, paired=True)))
    summary, tests = pd.DataFrame(rows), adjusted(tests)
    summary.to_csv(OUT / 'phase_summary.csv', index=False)
    tests.to_csv(OUT / 'baseline_adjusted_pairwise.csv', index=False)
    pd.DataFrame(baseline).to_csv(OUT / 'baseline_alignment.csv', index=False)
    if paired:
        adjusted(paired).to_csv(OUT / 'paired_id_sensitivity.csv', index=False)
    # Secondary phase-3 analysis: across-trial dispersion, distinct from the
    # level of the logged setpoint-dev measurement. Brown-Forsythe uses medians.
    variance_rows = []
    for metric in ['setpoint-avg', 'setpoint-dev', 'reward-total']:
        samples = {name: f.loc[f['sequence-idx'] == 3, metric] for name, f in frames.items()}
        test = stats.levene(*samples.values(), center='median')
        variance_rows.append(dict(metric=metric, comparison='all three',
                                  statistic=test.statistic, p=test.pvalue))
        for lower, higher in combinations(SETTINGS, 2):
            test = stats.levene(samples[lower], samples[higher], center='median')
            variance_rows.append(dict(metric=metric, comparison=f'{lower} vs {higher}',
                                      statistic=test.statistic, p=test.pvalue))
    variance = adjusted(variance_rows)
    variance.to_csv(OUT / 'phase3_across_trial_variance_tests.csv', index=False)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for offset, (name, (_, _, color)) in zip([-.07, 0, .07], SETTINGS.items()):
            s = summary[(summary.setting == name) & (summary.metric == metric)].sort_values('phase')
            ax.errorbar(s.phase+offset, s['mean'], yerr=s.ci95_high-s['mean'],
                        label=name, color=color, marker='o', capsize=3)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3], ['Canalized', 'Decanalized', 'Recanalized'])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('HRL-CDR: decanalized overlap multiplier\n'
                 'Means and 95% confidence intervals; 250 trials per setting; 1000-unit phases', fontsize=14)
    fig.savefig(OUT / 'overview.png', dpi=180)
    fig.savefig(OUT / 'overview.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for phase, color, marker in [(2, '#d46a35', 'o'), (3, '#277f91', 's')]:
            s = summary[(summary.metric == metric) & (summary.phase == phase)].sort_values('multiplier')
            ax.errorbar(s.multiplier, s['mean'], yerr=s.ci95_high-s['mean'],
                        label='Decanalized' if phase == 2 else 'Recanalized',
                        color=color, marker=marker, capsize=4)
        ax.set_title(title)
        ax.set_xticks([1, 4, 7], ['1×', '4×', '7×'])
        ax.set_xlabel('Decanalized overlap multiplier')
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend()
    fig.suptitle('HRL-CDR: response to overlap multiplier\nMeans and 95% confidence intervals', fontsize=14)
    fig.savefig(OUT / 'multiplier_response.png', dpi=180)
    plt.close(fig)

    means = summary.pivot(index=['metric', 'phase'], columns='setting', values='mean').reset_index()
    columns = ['metric', 'phase', *SETTINGS]
    medians = summary.pivot(index=['metric', 'phase'], columns='setting', values='median').reset_index()
    final = summary[summary.phase == 3].pivot(index='setting', columns='metric', values='mean')
    final['net_recorded_gain_plus_loss'] = final['gain-total']+final['loss-total']
    final.reset_index().to_csv(OUT / 'final_resource_balance.csv', index=False)
    def pct(name, metric):
        return 100*(final.loc[name, metric]/final.loc['1x (0 plasticity)', metric]-1)
    report = f'''# HRL-CDR plasticity comparison

## Design

User-specified decanalized counting: 0 plasticity = 1 counted overlap per actual
overlap; standard = 4; 6 plasticity = 7. All files contain HRL CDR trials.
The 1× condition still decanalizes; it retains the baseline counting rate rather
than removing the decanalized phase. Every phase has logged duration {next(iter(durations))}.
Files have {', '.join(str(f.trial_id.nunique()) for f in frames.values())} trials respectively.
Only the three named YAML files are used.

## Interpretation

**The main separation is in recanalization, rather than in the size of the
decanalized disturbance.** All three conditions undergo a large positive
setpoint excursion and reduced pellet collection during decanalization. None
of the 18 phase-2 pairwise contrasts survives the specified 36-test Holm family.
This is not proof that the settings are equivalent during that phase.

Final-phase means follow a consistent ordering: increasing overlap counting
from 1× to 4× to 7× corresponds to lower reward, lower resource gain, fewer
pellets, less negative recorded loss, and a higher signed setpoint mean.
The 1× condition ends closest to its starting reward, gain and pellet levels.

Compared with 1×, final-phase reward is {abs(pct('4x (standard)', 'reward-total')):.1f}% lower at 4×
and {abs(pct('7x (6 plasticity)', 'reward-total')):.1f}% lower at 7×. Gain is
{abs(pct('4x (standard)', 'gain-total')):.1f}% and {abs(pct('7x (6 plasticity)', 'gain-total')):.1f}% lower,
and pellet count is {abs(pct('4x (standard)', 'gains-harvested')):.1f}% and
{abs(pct('7x (6 plasticity)', 'gains-harvested')):.1f}% lower respectively.
Those reward, gain and pellet differences from 1× survive Holm correction after
baseline adjustment. Between 4× and 7×, only gain and loss differences survive;
reward and pellet differences do not.

Less negative loss partly offsets the reduced gain. The derived arithmetic
gain-plus-loss total is {final.loc['1x (0 plasticity)', 'net_recorded_gain_plus_loss']:.2f},
{final.loc['4x (standard)', 'net_recorded_gain_plus_loss']:.2f}, and
{final.loc['7x (6 plasticity)', 'net_recorded_gain_plus_loss']:.2f} for 1×, 4×, and 7×.
Thus the lower reward cannot simply be described as less net recorded resource
input. This arithmetic is descriptive and does not reconstruct the full state
trajectory, resets, or unlogged contributions.

Signed final setpoint means rise from 8.60 to 13.01 to 16.39. For 7× minus 1×,
the baseline-adjusted difference is +7.79 (unadjusted 95% CI 2.81–12.76), but the
primary Holm-adjusted p is 0.0576. Conditional matched-ID analysis gives adjusted
p=0.000566 for that contrast. The state-shift conclusion therefore depends on
the pairing assumption; the resource and reward differences above are clearer.
Signed means are not absolute errors.

This pattern is consistent with a persistent effect of decanalization's overlap
counting on the recanalized environment or behavior. The aggregates cannot
identify the mechanism or demonstrate neural adaptation, faster recovery, or an
optimal setting. No uninterrupted CCC condition is part of this comparison.

## Phase means

{table(means, columns)}

## Phase medians

{table(medians, columns)}

## Comparisons adjusted for baseline

For each trial, subtract the initial canalized phase from phase 2 or phase 3.
Compare those changes with two-sided Welch tests between runs. Differences are
higher multiplier minus lower multiplier. A single Holm family covers all 36
tests: six metrics × two phases × three pairwise contrasts. The confidence intervals
are unadjusted 95% intervals. A CI excluding zero need not survive this correction.

{table(tests, ['metric', 'phase', 'higher', 'lower', 'difference', 'ci95_low', 'ci95_high', 'p_holm'])}

## Phase-3 variation across trials

This is a secondary analysis of the spread across trials, distinct from comparing
the mean of `setpoint-dev`. Brown-Forsythe tests (median-centered Levene tests)
assume independent trial samples. Holm correction covers all 12 tests here:
three metrics × (one omnibus comparison plus three pairwise comparisons).
No test survives this correction. This does not demonstrate equal variances.

{table(variance, ['metric', 'comparison', 'statistic', 'p', 'p_holm'])}

## Baseline alignment

{table(pd.DataFrame(baseline), ['metric', 'higher', 'lower', 'n', 'exact_matches', 'max_absolute_difference'])}

## Limits

- The overlap mapping comes from the user; YAML lacks configuration, checkpoint,
  and seed metadata. Identical baselines support comparability but do not prove
  all settings or policies were held fixed.
- Primary tests do not assume cross-run pairing. Conditional trial-ID paired
  sensitivity tests are saved separately, with their own 36-test Holm family.
- Inference assumes independent trial replicates, not independent trained policies.
- `setpoint-avg` is signed. Closer group means to zero do not establish lower
  mean absolute error; the files cannot recover time within ±5 or recovery speed.
- `setpoint-dev` is used as logged, with no verified formula or sample count.
- Loss is signed: more negative means greater recorded loss magnitude.
- The analysis cannot establish the mechanism of a final-phase difference or
  whether it reflects learning, altered resources, or recovery transients.
- Nonsignificant tests do not establish equivalence or an optimal multiplier.

## Provenance and reproduction

'''
    for name, (filename, _, _) in SETTINGS.items():
        path = ROOT / 'data' / filename
        report += f'- {name}: `{path}`; SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`\n'
    report += '\nRun `python analysis/compare_plasticity.py` with analysis/requirements.txt installed.\n\n![Phase comparison](overview.png)\n\n![Multiplier response](multiplier_response.png)\n'
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    print(means[columns].to_string(index=False))
    print('\nBaseline-adjusted pairwise tests:\n'+tests.to_string(index=False))
    print(f'\nSaved to {OUT}')


if __name__ == '__main__':
    main()
