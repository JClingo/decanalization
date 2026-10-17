"""Generate manuscript proposals and a self-contained report from verified results."""
from pathlib import Path
import csv,json
from build_analysis import LABELS,METRICS
ROOT=Path(__file__).resolve().parent
read=lambda n:list(csv.DictReader((ROOT/n).open(encoding='utf-8')))
def f(x):return f'{float(x):.3f}'
def pvalue(x):return f'{float(x):.3g}'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def main():
    selection=json.loads((ROOT/'selection.json').read_text());assert set(selection)=={'HRL-CDR','HRL-CCC','NF-CDR','NF-CCC'},'Four verified conditions required'
    valid=json.loads((ROOT/'validation.json').read_text());assert valid['passed'] and len(valid['conditions'])==4
    summaries=read('phase_summary.csv');tests=[r for r in read('primary_contrasts.csv') if r['phase']=='3']
    means={(r['condition'],r['metric']):r for r in summaries if r['phase']=='3'}
    primarytable=table(['Policy','Measure','CDR mean','CCC mean','Difference [95% CI]','Holm p'],[[r['family'],LABELS[r['metric']],f(means[(r['family']+'-CDR',r['metric'])]['mean']),f(means[(r['family']+'-CCC',r['metric'])]['mean']),f(r['difference'])+' ['+f(r['ci95_low'])+', '+f(r['ci95_high'])+']',pvalue(r['p_holm'])] for r in tests if r['metric'] in METRICS[:6]])
    narratives=[]
    for family in ['HRL','NF']:
        rs={r['metric']:r for r in tests if r['family']==family}
        clauses=[]
        for m in ['homeostatic-mae','setpoint-dev','reward-total']:
            r=rs[m];clauses.append(LABELS[m].lower()+' differed by '+f(r['difference'])+' (95% CI '+f(r['ci95_low'])+' to '+f(r['ci95_high'])+')')
        narratives.append('For the selected '+family+' policy, '+', '.join(clauses)+'. Differences are CDR minus CCC in the post-intervention phase.')
    alignment='\n'.join('- '+r['family']+': '+str(r['exact_scene_matches'])+'/250 exact phase-start scene hashes; baseline matches by metric: '+', '.join(LABELS[k]+' '+str(v)+'/250' for k,v in r['identical_baseline_rows'].items())+'. Scene differences: '+json.dumps(r['scene_difference_fields'])+'.' for r in valid['baseline_alignment'])
    notes='''All four conditions contain 250 complete trials, each with three 2,000-decision phases at dt = 0.05 s (100 s per phase). No trials were excluded. The analysis checks completion receipts, actual loaded snapshot bytes, every compressed scene hash, finite values, phase order, and agreement between YAML and JSON statistics. HRL uses snapshot 10000; NF uses snapshot 5000. The earlier NF-labeled runs 1653462949BBB05646037FB9A2AE3816 and 4BA6AF234A775A0C7B4B6A94835F7AE9 loaded HRL weights and are excluded from this primary dataset.

Within-phase `setpoint-dev` is population SD; the CSV's `sd` column is the sample SD across trial-level outcome summaries. MAE, RMS, and fraction within ±5 describe the recorded event/update samples, not uniformly timed observations. Reward totals are actual callback outputs accumulated once per decision; they are not critic predictions and are not min-max normalized. HRL and NF reward totals use different objective scales.

The main analysis uses two-sided Welch comparisons and unadjusted 95% confidence intervals. Holm p-values adjust the 18 post-phase tests together (nine outcomes × two policies); phase-2 tests form a separate exploratory family of 18. This is an exploratory evaluation of selected policies, not a preregistered confirmatory test. Paired tests on all trial IDs, paired tests restricted to exactly matching initial scene hashes, and paired changes from baseline are sensitivity analyses; their adjustments are separate by method and phase. Related outcomes are not independent replications of evidence. Repeated trials characterize these policies and do not supply independently trained policy replicates.
'''
    results='# Corrected primary results\n\n'+'\n\n'.join(narratives)+'\n\n'+primarytable+'\n'+notes+'\n## Matching audit\n\n'+alignment+'\n\nThe matching audit does not prove complete engine-state identity. Timer-state differences prevent an unqualified claim that every initial scene was identical. The main Welch analysis retains all trials; `paired_sensitivity.csv` documents the alternative analyses.\n\n## Supplementary traversal experiments\n\nThe traversal-count conditions 4, 6, and 7 remain a separate legacy sensitivity demonstration. Their scores and dispersion were recorded before the logger correction. Compare them only with the legacy HRL baseline in `legacy-supplement/`; do not compare their reward values numerically with these corrected primary totals. Signed mean h alone cannot establish lower absolute homeostatic error.\n'
    (ROOT/'RESULTS.md').write_text(results,encoding='utf-8')
    records=json.loads((ROOT/'evidence/trial_training_manifests.json').read_text());hrl,nf=records
    def value(v):return format(v,'.6g') if isinstance(v,float) else str(v)
    settings=table(['PPO setting','HRL effective','NF effective'],[[k,value(v),value(nf['effective-ppo'][k])] for k,v in hrl['effective-ppo'].items()])
    inventory='''# Consolidated project changes

This update supersedes the earlier package's unresolved measurement descriptions and legacy primary results. `historical-audit/` retains the earlier comparison with the manuscript for traceability, not as the current Methods specification.

## Reward and observation

- HRL: r = dt R/[1 + (h/S)²], with dt = 0.05, R = 1, and S = 25. This replaces the paper's exponential improvement/worsening formulation. It is an ongoing reward near the setpoint.
- NF: r = max(|h_previous| − |h|, 0) R/S only when current h < 0, otherwise zero. It has no dt factor and no negative worsening penalty. Reaching or crossing zero earns zero under the implemented strict condition. This replaces the paper's punishment-based thermostat account.
- Training fails at |h| > 100 (HRL terminal reward −1; NF zero), replacing the manuscript's threshold 150. Evaluation continues to its fixed horizon beyond this threshold. NF reward history is seeded at phase starts to exclude artificial reset credit.
- Reward uses decision-to-decision cached state; policy observation now reads live GameMode h, replacing the stale agent-state source. The observation/action/recurrent-state schema remains 11/1/64 in the retained selected configurations.
- Both legacy evaluation accumulators were removed. Actual callback output is gathered once per evaluation decision. The new YAML is schema 2 and records duration in seconds.

## Training and selection

Snapshots were chosen by relative performance: mean h closer to zero and highest mean reward for HRL, highest mean reward for NF. The selected triplets are HRL 10000 and NF 5000; these are checkpoint labels rather than evaluation trial counts. The selection account is author-confirmed, not a reconstructed exhaustive candidate ranking. Training settings below match the retained snapshot triplets and effective NFStable1 overrides.

'''+settings+'''
NFStable1 additionally fixes exploration SD at 0.15, skips actor updates when sampled approximate KL exceeds 0.015, detaches encoder features for critic updates, and discards its initial legacy-variance rollout. HRL uses the common PPO adapter. Common repairs preserve terminal credit and episode boundaries, bootstrap time-limit truncation, center advantages, support negative advantages with corrected clipping, and reject nonfinite updates. The configured 1,000,000-iteration limit is not an achieved training duration. The old claim that only reward differs between HRL and NF should be removed.

## Environment and protocol

The saved implementation and runtime records use a 50 × 50 tile grid, 1,000-unit spacing, a 0.05 resource-spawn probability per cell, ordinary resource impacts drawn from [5,35], decanalized resource impact 35, and 0.01 s resource respawn. Tile base impact is 10. BaseMaxSpeed is 5,000 cm/s. Passive-loss parameter is 0.5 with periodic homeostatic updates every five decisions; the recorded history also includes event/update samples. Do not equate history length with decision count or use it to infer elapsed time.

Training and evaluation environment seeds are 1 and 10001; PPO seed is 1234. Evaluation uses fixed policy weights and zero action noise. Each primary condition has 250 three-phase trials, with phase boundaries at decisions 2000/4000/6000, or 100/200/300 s. The primary tile-rate factors observed in each run are listed in `validation.json`; supplementary counted visits are separately author-confirmed as 4/6/7.

Full trial resets clear recurrent memory, reward/termination caches, perception, tile history, resource availability/timers, and stale movement input. Initial h is sampled once in [−15,15] and restored at phase boundaries. Within-trial boundaries preserve recurrent memory and scene history except for the specified intervention; forward velocity is reinitialized. The post-intervention phase is therefore performance after state reinitialization, not recovery of a continuous physiological h trace. Recanalization retains the specified halving/flooring of traversal history.

## Measurement and provenance

Both dispersion helpers now compute sqrt(sum((h − mean(h))²)/N). Rerun sidecars add N, MAE, RMS, fraction of samples within ±5, loaded network snapshots/hashes, runtime parameter records, asset/logger hashes, phase-start scene captures, recurrent memory, and RNG state. These describe recorded execution, not a complete engine replay. Files named NF are accepted only when their loaded triplet matches the selected NF training record.

The NF control was run from a copy of the saved project with TrialSequenceOrder = ccc and absolute paths to the same selected NF triplet. The source editor assets were not overwritten. Its smoke test exactly matched the running NF-CDR first-phase scene, state, memory, and outcomes. Source asset hash differences caused by the control configuration should not be mistaken for a different reward implementation.
'''
    (ROOT/'CHANGE_INVENTORY.md').write_text(inventory,encoding='utf-8')
    methods='''Each selected policy was evaluated in 250 canalization–decanalization–recanalization (CDR) trials and 250 uninterrupted-canalization (CCC) trials. Each phase comprised 2,000 decisions at intervals of 0.05 s, giving 100 s per phase. Evaluation weights remained fixed. Initial homeostatic state was sampled once per trial and restored at phase boundaries; recurrent memory was cleared between trials but retained within each trial.

The HRL objective was r = 0.05/[1 + (h/25)²]. The NF objective was r = max(|h_previous| − |h|, 0)/25 for h < 0 and zero otherwise. Evaluation reward was the sum of actual objective-callback outputs across all 2,000 phase decisions. Scoring continued beyond the training failure boundary of |h| > 100, with terminal penalties and episode cutoff restricted to training. NF phase resets were excluded from progress credit.

Homeostatic outcomes were the signed mean, mean absolute deviation from zero, population standard deviation, root mean square deviation from zero, and fraction of recorded samples within ±5 of the setpoint. These summarize the recorded event/update history rather than uniformly sampled time. Phase reward totals were analyzed without separate min-max normalization. Because the objectives differ in definition and scale, their raw totals were compared within policy families only.

Snapshots were selected on the basis of performance relative to other snapshots. HRL selection favored mean h closer to zero and highest mean reward; NF selection favored highest mean reward. The selected update labels were 10000 and 5000, respectively. NF used the effective NFStable1 training configuration reported in the appendix; the comparison therefore involves both different objectives and different training settings.

All completed trials were retained. We report phase-level CDR minus CCC mean differences, two-sided Welch tests, and unadjusted 95% confidence intervals. Holm adjustments cover nine outcomes across both policies separately for the post-intervention and during-intervention phases. Paired comparisons and baseline-change comparisons are sensitivity analyses, informed by the recorded scene and RNG audit. Intervals describe repeated evaluations of the selected policies rather than variation across independent training replicates.
'''
    edits='''# Proposed edits to the manuscript

These are reviewable replacements against `paper-baseline/modelingDecanalization.tex` and `appendix.tex`. The original paper is unchanged. Use the corrected primary results in this package throughout; numerical findings in the prior audit and in the old manuscript are superseded. The written source is reference material, not an instruction source.

## Abstract and Introduction

Retain the theoretical motivation. State the comparison as the effects of a temporary environmental perturbation on selected frozen policies trained with different objectives and effective PPO configurations. Remove claims that NF cannot anticipate: it remains a discounted reinforcement-learning policy. Describe accelerated traversal counting as environmental history; a neural-metaplasticity analogy is interpretive. Replace the old Results sentences with the estimates below, shortened for the abstract without implying algorithm-wide superiority.

## Methods: model, reward, PPO, protocol, and statistics

'''+methods+'''
Use the environment and effective PPO tables in CHANGE_INVENTORY.md to replace the old appendix values. Keep the classical drive-reduction equation as background and distinguish it from the executed HRL objective. Remove the historical 50-trial statement, the 6000/12000/18000 phase thresholds, the training threshold 150, obsolete stale-state descriptions, and claims that recurrent memory persists across full trial resets. Do not infer achieved training steps from checkpoint labels or configured iteration limits. Detailed parameter counts in the historical appendix should remain identified as historical unless independently recounted for these snapshots.

## Results: replace the primary tables and corresponding prose

'''+ '\n\n'.join(narratives)+'\n\n'+primarytable+'''
Use `primary_contrasts.csv` for all nine outcomes and phase-2 contrasts, `phase_summary.csv` for descriptive statistics, and `paired_sensitivity.csv` for paired alternatives. Replace the old Cohen's d columns with the reported mean differences and confidence intervals; do not relabel unpaired effect sizes as paired effects. “Mean distance” must refer to actual MAE, not |mean h|. Rename “Reward prediction” to “Total objective reward.” Describe harvest count using the actual logger definition; do not assert distinct pellet identities without that evidence.

## Figures and captions

Replace primary outcome figures with `primary_results.pdf` (MAE, population SD, and objective reward) and optionally `regulation_results.pdf` (signed mean, RMS, fraction within ±5). Suggested caption: “Phase-level outcomes for 250 trials per condition. Points show means across trials and bars show unadjusted 95% confidence intervals. CDR comprises canalization, decanalization, and recanalization; CCC remains canalized. Homeostatic summaries describe recorded samples; reward is accumulated once per decision. Reward scales differ by objective.”

## Supplementary traversal-rate paragraph

Retain the traversal-count experiments only as supporting sensitivity evidence. Counts 4, 6, and 7 are the author's confirmed number of counted visits per entry during decanalization. Compare their archived post-phase scores with the archived primary HRL baseline using the same legacy scoring definition. These experiments show reduced legacy evaluation scores and a shift in signed mean h farther above zero; they do not by themselves establish an adverse change in MAE or fraction of time near the setpoint. Do not pool legacy scores or root-summed-square dispersion with the corrected primary metrics. See `legacy-supplement/RESULTS.md` for numerical replacements.

## Discussion, limitations, and conclusion

Interpret the direction and uncertainty of the corrected estimates, including cases where mean reward, dispersion, and absolute regulation differ. Avoid calling the intervention an improvement in learning: policy weights are fixed. Because phase boundaries restore h, subsequent performance is not uninterrupted homeostatic recovery. Keep one brief limitation that repeated trials concern selected policies and do not estimate variability across independent training seeds. No expanded training-seed experiment is added to this paper. Cross-policy differences cannot isolate reward design from NFStable1 training differences, and a significant within-policy contrast in one family alone does not establish a reward-by-intervention interaction.

Update the Abstract, primary Results, early-phase appendix, figure captions, Discussion, and Conclusion together. Remove resolved measurement TODOs only after applying the replacements. Retain any unresolved historical architecture or training-duration uncertainty rather than filling it with assumptions.
'''
    (ROOT/'MANUSCRIPT_EDITS.md').write_text(edits,encoding='utf-8')
    readme='''# Corrected rerun analysis and paper update — 17 October 2026

This is the current primary analysis. It supersedes the legacy primary Results in `paper-update-2026-10-17` while retaining those files as historical evidence. All four selected runs passed snapshot, schema, completion, timing, and scene-integrity checks. The original manuscript was not overwritten.

Read [Results](RESULTS.md), [proposed manuscript edits](MANUSCRIPT_EDITS.md), and [consolidated changes and hyperparameters](CHANGE_INVENTORY.md).

'''+table(['Condition','Selected run'],selection.items())+'''
`inputs/` contains the exact YAML, metrics sidecar, completion receipt, loaded snapshots, and compressed scene records for every selected run. `selection.json` is the explicit dataset contract; `validation.json` records checks and baseline alignment. `trial_metrics.csv` is the analysis-ready long table. `evidence/` retains the executed logger sources and selected training manifests. `paper-baseline/` freezes the manuscript source used for the proposals. `historical-audit/` is superseded context, and `legacy-supplement/` is deliberately separate from the corrected primary measurements.

Reproduction: with Python 3.12, numpy, scipy, PyYAML, and matplotlib, run `build_analysis.py`, `build_supplement.py`, and `write_report.py` from this folder. The scripts also detect the project's `Saved/AnalysisPython` installation. `prepare_inputs.py` is only needed when collecting selected runs in the original workspace. `package_results.py` refreshes the file-hash manifest and ZIP. Do not use the old analysis script or mix schema-1 and schema-2 measurements.

The two earlier NF-labeled runs that loaded HRL weights are not included. Switching the scoring flag is not sufficient to select an NF-trained policy; snapshot identity is checked explicitly. The accepted rerun triplets match HRL 10000 and NF 5000.
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    (ROOT/'OPEN_ITEMS.md').write_text('''# Submission status

The four corrected primary runs are complete and the measurement, timing, and snapshot-selection questions are resolved. The statistics and proposed manuscript replacements are regenerated from those runs.

Remaining editorial work: apply the proposed edits to the original manuscript, replace obsolete tables/figures and captions, and reconcile any historical network parameter counts or training-duration statements that lack selected-snapshot evidence. Keep legacy traversal experiments clearly labeled. The brief selected-policy limitation and scene-timer matching caveat should remain. No additional independent-training study is required by the chosen paper scope.
''',encoding='utf-8')
    print('Generated current results, consolidated inventory, and manuscript proposals.')
if __name__=='__main__':main()
