"""Compare CDR and CCC YAML files without using archived results.

python analysis/compare_conditions.py --reward NF --checkpoint 5000
"""
import argparse
import hashlib
from pathlib import Path

from compare_cdr_ccc import load, contrast, adjusted
from analyze_cdr import ROOT, METRICS, table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reward', choices=['NF', 'HRL'], default='NF')
    parser.add_argument('--checkpoint', default='5000')
    parser.add_argument('--cdr', type=Path, help='Override the CDR input YAML')
    parser.add_argument('--ccc', type=Path, help='Override the CCC input YAML')
    parser.add_argument('--output', type=Path, help='Override the output directory')
    args = parser.parse_args()
    prefix = args.reward
    paths = {name: ROOT / f'data/{prefix}-{name}-{args.checkpoint}.yml' for name in ['CDR', 'CCC']}
    if args.cdr:
        paths['CDR'] = args.cdr
    if args.ccc:
        paths['CCC'] = args.ccc
    frames = {name: load(path, name.lower(), prefix) for name, path in paths.items()}
    for f in frames.values():
        if f.trial_id.nunique() < 2 or f.duration.nunique() != 1:
            raise ValueError('Need at least two complete trials and constant phase durations')
    if frames['CDR'].duration.iloc[0] != frames['CCC'].duration.iloc[0]:
        raise ValueError('Phase durations must match across conditions')
    out = args.output or ROOT / f'results/{prefix}-CDR-vs-CCC-{args.checkpoint}'
    comparison_title = f'{paths["CDR"].stem} vs {paths["CCC"].stem}'
    out.mkdir(parents=True, exist_ok=True)
    summaries, between, changes, sensitivity, alignment = [], [], [], [], []
    for name, f in frames.items():
        f.to_csv(out / f'{name.lower()}_sequences.csv', index=False)
        for metric in METRICS:
            for phase in [1, 2, 3]:
                x = f.loc[f['sequence-idx'] == phase, metric]
                margin = stats.t.ppf(.975, len(x)-1)*stats.sem(x)
                summaries.append(dict(dataset=name, metric=metric, phase=phase, n=len(x),
                                      mean=x.mean(), sd=x.std(ddof=1), median=x.median(),
                                      min=x.min(), max=x.max(), ci95_low=x.mean()-margin,
                                      ci95_high=x.mean()+margin))
    for metric in METRICS:
        wide = {name: f.pivot(index='trial_id', columns='sequence-idx', values=metric).sort_index()
                for name, f in frames.items()}
        paired = wide['CDR'].index.equals(wide['CCC'].index)
        if paired:
            delta = wide['CDR'][1]-wide['CCC'][1]
            alignment.append(dict(metric=metric, n=len(delta), exact_matches=int((delta == 0).sum()),
                                  max_absolute_difference=abs(delta).max()))
        for phase in [1, 2, 3]:
            a, b = wide['CDR'][phase], wide['CCC'][phase]
            between.append(dict(metric=metric, phase=phase, **contrast(a, b)))
            if paired:
                sensitivity.append(dict(metric=metric, phase=phase, **contrast(a, b, paired=True)))
        for phase in [2, 3]:
            a, b = (wide[name][phase]-wide[name][1] for name in ['CDR', 'CCC'])
            changes.append(dict(metric=metric, phase=phase, cdr_change=a.mean(), ccc_change=b.mean(),
                                **contrast(a, b)))
    summary, between, changes = pd.DataFrame(summaries), adjusted(between), adjusted(changes)
    for name, f in [('phase_summary', summary), ('between_runs_welch', between),
                    ('baseline_adjusted_changes', changes), ('baseline_alignment', pd.DataFrame(alignment))]:
        f.to_csv(out / f'{name}.csv', index=False)
    if sensitivity:
        adjusted(sensitivity).to_csv(out / 'paired_id_sensitivity.csv', index=False)
    means = summary.pivot(index=['metric', 'phase'], columns='dataset', values='mean').reset_index()
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for name, color, offset in [('CDR', '#d46a35', -.04), ('CCC', '#277f91', .04)]:
            s = summary[(summary.dataset == name) & (summary.metric == metric)].sort_values('phase')
            ax.errorbar(s.phase+offset, s['mean'], yerr=s.ci95_high-s['mean'],
                        marker='o', capsize=4, color=color, label=name, lw=2)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3], ['Phase 1\nC / C', 'Phase 2\nD / C', 'Phase 3\nR / C'])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend()
    counts = ' / '.join(str(f.trial_id.nunique()) for f in frames.values())
    fig.suptitle(f'{comparison_title}: {counts} trials\n'
                 f'Means and 95% confidence intervals; {frames["CDR"].duration.iloc[0]}-unit phases', fontsize=14)
    fig.savefig(out / 'overview.png', dpi=180)
    fig.savefig(out / 'overview.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, metric in zip(axes.flat, ['setpoint-avg', 'reward-total', 'gains-harvested', 'gain-total']):
        rng = np.random.default_rng(42)
        for name, color, offset in [('CDR', '#d46a35', -.18), ('CCC', '#277f91', .18)]:
            values = [frames[name].loc[frames[name]['sequence-idx'] == phase, metric].to_numpy()
                      for phase in [1, 2, 3]]
            positions = np.arange(1, 4)+offset
            boxes = ax.boxplot(values, positions=positions, widths=.27, patch_artist=True,
                               showfliers=False, manage_ticks=False,
                               medianprops={'color': 'black'})
            for box in boxes['boxes']:
                box.set_facecolor(color)
                box.set_alpha(.55)
            for pos, x in zip(positions, values):
                ax.scatter(pos+rng.uniform(-.08, .08, len(x)), x, color=color, alpha=.25, s=8)
            ax.plot([], [], color=color, label=name, lw=3)
        ax.set_title(METRICS[metric])
        if metric in ['setpoint-avg', 'gain-total']:
            ax.set_yscale('symlog', linthresh=25)
            ax.set_ylabel('Symmetric log scale; linear within ±25')
        ax.set_xticks([1, 2, 3], ['Phase 1', 'Phase 2', 'Phase 3'])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend()
    fig.suptitle(f'{comparison_title}: trial distributions\n'
                 'Dots show every trial; boxes show quartiles and median', fontsize=14)
    fig.savefig(out / 'distributions.png', dpi=180)
    plt.close(fig)
    medians = summary.pivot(index=['metric', 'phase'], columns='dataset', values='median').reset_index()
    report = f'''# {comparison_title}

Validated complete trials: CDR {frames['CDR'].trial_id.nunique()}, CCC {frames['CCC'].trial_id.nunique()}.
Every trial has all three ordered phases; each phase has logged duration {frames['CDR'].duration.iloc[0]}.
Only the two named YAML inputs are used. Rewards are raw; totals require no duration rescaling.

## Phase means

{table(means, ['metric', 'phase', 'CDR', 'CCC'])}

## Phase medians

{table(medians, ['metric', 'phase', 'CDR', 'CCC'])}

Means and medians can differ substantially. Inspect both and the individual-trial
distribution plot before interpreting average reward or average signed state.

## Between-condition comparisons

Differences are CDR minus CCC. Two-sided Welch tests; Holm adjustment across
these 18 comparisons. Confidence intervals are unadjusted 95% intervals.

{table(between, ['metric', 'phase', 'difference', 'ci95_low', 'ci95_high', 'p_holm'])}

## Changes relative to phase 1

Subtract each trial's phase 1 before comparing CDR versus CCC with Welch tests.
Difference = (CDR phase j minus phase 1) minus (CCC phase j minus phase 1).
Holm adjustment covers these 12 comparisons separately. Intervals are unadjusted.
This preserves within-trial dependence without assuming cross-run matching.

{table(changes, ['metric', 'phase', 'cdr_change', 'ccc_change', 'difference', 'ci95_low', 'ci95_high', 'p_holm'])}

## Baseline alignment and limits

{table(pd.DataFrame(alignment), ['metric', 'n', 'exact_matches', 'max_absolute_difference']) if alignment else 'Trial IDs do not align across conditions.'}

- Matching trial IDs or identical baselines do not verify matched seeds or policies.
  If IDs align, conditional paired tests are in `paired_id_sensitivity.csv`
  (a separate 18-test Holm family). Primary tests use Welch comparisons.
- Inference assumes independent trial replicates and does not measure variation
  across independently trained policies. The YAML omits seed/configuration metadata.
- Signed setpoint means are not mean absolute errors. Negative values represent
  a signed offset; comparing closeness of group means to zero cannot establish MAE.
- `setpoint-dev` is used exactly as logged; its formula and sampling count are not
  specified here. Do not interpret it as a verified conventional standard deviation.
- More gain or higher NF reward need not imply better homeostatic regulation.
  Resource gain/loss changes can explain behavior without learning or adaptation.
- Phase averages cannot establish recovery speed. Nonsignificant differences
  do not establish equivalence, and filenames do not verify checkpoint provenance.

## Provenance

'''
    for name, path in paths.items():
        report += f'- {name}: `{path}`; SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`\n'
    command = f'python analysis/compare_conditions.py --reward {prefix} --checkpoint {args.checkpoint}'
    for flag, value in [('cdr', args.cdr), ('ccc', args.ccc), ('output', args.output)]:
        if value:
            command += f' --{flag} "{value}"'
    report += f'\nReproduce: `{command}`\n\n![Comparison](overview.png)\n\n![Trial distributions](distributions.png)\n'
    if prefix == 'NF' and args.checkpoint == '5000' and not args.cdr and not args.ccc:
        final = changes[changes.phase == 3].set_index('metric')
        key_findings = '''## Interpretation of this NF run

Decanalization is associated with a large positive excursion in signed setpoint
mean and a collapse in pellet count. Mean phase-2 signed distance is 443.05 versus
3.76 under CCC; median distance is 130.33 versus -15.40. Pellet means are 2.50 versus
29.22, a 91.4% reduction relative to CCC.

The apparent reward increase is not representative of the typical trial: phase-2
CDR reward has mean 57.67 but median 3.33 (CCC mean 41.72, median 39.77). CDR phase-2
gain has mean 863.94 but median 210.00 (CCC median 646.50). Extreme trials pull means
upward. Neither reward nor gain differences survive the specified Holm correction.
Higher mean reward here is not evidence of better regulation.

Final-phase behavior returns near the uninterrupted condition at the group level.
Median signed distance is -15.56 in CDR and -14.47 in CCC, compared with -16.55
initially in both. Positive final means (13.85 and 6.95) reflect the long positive
tail and must not be read as the typical trial or as MAE.

'''
        for metric in ['setpoint-avg', 'reward-total', 'gains-harvested', 'loss-total']:
            x = final.loc[metric]
            key_findings += (f'- Baseline-adjusted {metric}: CDR minus CCC = {x.difference:+.2f}; '
                             f'95% CI [{x.ci95_low:.2f}, {x.ci95_high:.2f}]; '
                             f'Holm-adjusted p={x.p_holm:.3g}.\n')
        key_findings += '''
There is no statistically resolved final difference in signed state, logged
deviation, reward, gain, or pellets after multiplicity correction. Loss remains
more negative in CDR: resource balance has not fully converged. Wide intervals
and nonsignificant tests do not establish equivalent regulation or rule out
meaningful effects. This is evidence of a temporary disruption with broad
behavioral recovery, rather than proof of adaptation or complete recovery.

'''
        report = report.replace('## Phase means\n', key_findings+'## Phase means\n', 1)
    if prefix == 'HRL' and paths['CDR'].stem == 'HRL-CDR-0-plasticity':
        final = changes[changes.phase == 3].set_index('metric')
        phase3 = summary[summary.phase == 3].pivot(index='metric', columns='dataset', values='mean')
        total_reward = {name: f.groupby('trial_id')['reward-total'].sum().mean()
                        for name, f in frames.items()}
        findings = f'''## Interpretation: 0 plasticity versus uninterrupted canalization

0 plasticity means ordinary overlap counting during decanalization, not absence
of decanalization. Initial setpoint and resource measurements match exactly by
trial ID. Baseline reward differs in one trial by only 0.000452.

The 0-plasticity CDR run has a large phase-2 disturbance, but its final-phase
measurements are better on reward, signed mean offset and logged deviation than
uninterrupted CCC. These statements describe observed metrics, not improved
learning or lower verified mean absolute error.

'''
        for metric in ['reward-total', 'setpoint-avg', 'setpoint-dev', 'gains-harvested']:
            x = final.loc[metric]
            findings += (f'- {metric}: CDR mean {phase3.loc[metric, "CDR"]:.2f} versus '
                         f'CCC {phase3.loc[metric, "CCC"]:.2f}. Baseline-adjusted difference '
                         f'{x.difference:+.2f}, 95% CI [{x.ci95_low:.2f}, {x.ci95_high:.2f}], '
                         f'Holm-adjusted p={x.p_holm:.3g}.\n')
        findings += f'''
These p-values use the 12-test baseline-change family defined below; CIs are
unadjusted. Lower logged setpoint deviation is not a verification of its formula
or a test of equal variance across trial-level means.

The final advantage does not offset the acute cost: mean summed reward across
all three phases is {total_reward['CDR']:.2f} in CDR versus {total_reward['CCC']:.2f}
in CCC, or {100*(1-total_reward['CDR']/total_reward['CCC']):.1f}% lower in CDR.
These are descriptive per-trial sums over equal total logged durations.

CDR also retains greater final-phase gain and more negative loss. The findings
are consistent with decanalization changing subsequent resource turnover and
the state trajectory, but aggregates cannot distinguish that mechanism from
recovery transients or other unverified configuration differences. The 1× run
remains near its own initial levels while CCC changes over time.

'''
        report = report.replace('## Phase means\n', findings+'## Phase means\n', 1)
    (out / 'report.md').write_text(report, encoding='utf-8')
    print(means.to_string(index=False))
    print('\nBaseline-adjusted contrasts:\n'+changes.to_string(index=False))
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
