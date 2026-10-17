"""Legacy traversal sensitivity only; never compare with schema-2 reward totals."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'Saved/AnalysisPython'))
import numpy as np
from scipy import stats
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_analysis import compare,holm

def main():
    folder=ROOT/'legacy-supplement';arrays={};rows=[];contrasts=[]
    for label,name in [('reference','HRL-CDR-10000.yml'),('4','HRL-CDR-4-plasticity.yml'),('6','HRL-CDR-6-plasticity.yml'),('7','HRL-CDR-7-plasticity.yml')]:
        d=yaml.safe_load((folder/'inputs'/name).read_text());assert d.get('schema-version',1)==1
        t=sorted(d['trials'],key=lambda x:x['id']);assert [x['id'] for x in t]==list(range(250))
        a=np.array([[x['sequences'][2][k] for k in ['setpoint-avg','reward-total']] for x in t]);assert np.isfinite(a).all();arrays[label]=a
        rows.append(dict(label=label,file=name,n=250,signed_mean_h=float(a[:,0].mean()),legacy_reward=float(a[:,1].mean())))
    for label in ['4','6','7']:
        for j,k in enumerate(['signed_mean_h','legacy_reward']):contrasts.append(dict(count=label,metric=k,**compare(arrays[label][:,j],arrays['reference'][:,j])))
    holm(contrasts)
    for name,data in [('summary.csv',rows),('contrasts.csv',contrasts)]:
        with (folder/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Legacy supplementary traversal-rate results','','These archived files were recorded before the reward and dispersion fixes. They are supporting sensitivity evidence, not additional corrected primary conditions. Counts 4, 6, and 7 were confirmed by the author. The comparison below uses the legacy primary HRL-CDR reference and identical measurement definitions; no schema-2 primary values enter this analysis.','','| Count label | Post signed mean h | Post legacy reward | Reward change vs legacy reference |','|---|---:|---:|---:|']
    baseline=rows[0]['legacy_reward']
    for row in rows:lines.append(f"| {row['label']} | {row['signed_mean_h']:.3f} | {row['legacy_reward']:.3f} | {100*(row['legacy_reward']/baseline-1):.2f}% |")
    lines+=['','Increasing counted visits was associated with lower legacy post-phase reward and signed mean h farther above zero. This is consistent with an adverse effect in these recorded outcomes, but signed mean h is not MAE, and the archived dispersion field is not population SD. Do not claim a demonstrated deterioration in time-weighted regulation or compare these reward totals directly with corrected primary rewards.','', 'The accompanying contrasts use two-sided Welch tests and unadjusted 95% intervals, with Holm correction across six exploratory comparisons.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for j,label in enumerate(['Signed mean h','Legacy evaluation score']):
        x=[arrays[k][:,j] for k in ['reference','4','6','7']]
        axes[j].errorbar(range(4),[v.mean() for v in x],yerr=[stats.t.ppf(.975,249)*stats.sem(v) for v in x],marker='o',capsize=4,color='#1b7089')
        axes[j].set_xticks(range(4),['Legacy reference','4','6','7']);axes[j].set_title(label);axes[j].grid(alpha=.2)
    fig.suptitle('Supplement only: legacy post-phase outcomes, 250 trials per condition')
    fig.savefig(folder/'supplement_results.png',dpi=180);fig.savefig(folder/'supplement_results.pdf');plt.close(fig)
if __name__=='__main__':main()
