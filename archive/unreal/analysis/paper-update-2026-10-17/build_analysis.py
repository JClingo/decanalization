"""Rebuild the paper evidence from frozen input copies. No project assets modified."""
from pathlib import Path
import sys, json, hashlib, csv, shutil
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
if (PROJECT/'Saved/AnalysisPython').exists():
    sys.path.insert(0, str(PROJECT/'Saved/AnalysisPython'))
import numpy as np
from scipy import stats
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ['setpoint-avg','setpoint-dev','reward-total','loss-total','gain-total','gains-harvested']
PRIMARY = ['HRL-CCC-10000.yml','HRL-CDR-10000.yml','NF-CCC-5000.yml','NF-CDR-5000.yml']
SUPPLEMENT = ['HRL-CDR-4-plasticity.yml','HRL-CDR-6-plasticity.yml','HRL-CDR-7-plasticity.yml']
def save(name, rows):
    with (ROOT/name).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def contrast(a,b):
    v=[np.var(x,ddof=1)/len(x) for x in [a,b]]
    se=np.sqrt(sum(v)); diff=np.mean(a)-np.mean(b)
    df=sum(v)**2/sum(z*z/(len(x)-1) for z,x in zip(v,[a,b])) if se else np.inf
    margin=stats.t.ppf(.975,df)*se
    return dict(difference=diff,ci95_low=diff-margin,ci95_high=diff+margin,
                p=float(2*stats.t.sf(abs(diff/se),df)) if se else float(diff==0))
def holm(rows):
    order=np.argsort([r['p'] for r in rows]); prev=0
    for rank,i in enumerate(order):
        prev=max(prev,min(1,(len(rows)-rank)*rows[i]['p'])); rows[i]['p_holm']=prev
    return rows
def main():
    (ROOT/'inputs').mkdir(exist_ok=True)
    arrays={}; summary=[]; manifest=[]
    for name in PRIMARY+SUPPLEMENT:
        path=ROOT/'inputs'/name
        if not path.exists(): shutil.copy2(PROJECT/'analysis/data'/name,path)
        data=yaml.safe_load(path.read_text())['trials']; seen=set(); rows=[]
        for t in data:
            assert t['id'] not in seen; seen.add(t['id'])
            assert t['reward-function']==name.split('-')[0]
            assert t['sequence-order']==name.split('-')[1].lower()
            phases=sorted(t['sequences'],key=lambda x:x['sequence-idx'])
            assert [p['sequence-idx'] for p in phases]==[1,2,3]
            assert [p['state'] for p in phases]==(['canalized']*3 if '-CCC-' in name else ['canalized','decanalized','recanalized'])
            rows.append((t['id'],phases))
        rows.sort(key=lambda x:x[0])
        durations={p['duration'] for _,ps in rows for p in ps}; assert durations=={1000}
        a=np.array([[[p[m] for m in METRICS] for p in ps] for _,ps in rows],float)
        assert np.isfinite(a).all(); arrays[name]=a
        manifest.append(dict(file=name,role='primary' if name in PRIMARY else 'supplement',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trials=len(rows),phases=len(rows)*3,durations=sorted(durations),ids=[r[0] for r in rows]))
        for phase in range(3):
            for k,m in enumerate(METRICS):
                x=a[:,phase,k]; margin=stats.t.ppf(.975,len(x)-1)*stats.sem(x)
                summary.append(dict(file=name,phase=phase+1,metric=m,n=len(x),mean=x.mean(),sd=x.std(ddof=1),median=np.median(x),ci95_low=x.mean()-margin,ci95_high=x.mean()+margin))
    save('phase_summary.csv',summary)
    (ROOT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    tests=[]; alignment=[]
    for family,suffix in [('HRL','10000'),('NF','5000')]:
        a,b=[arrays[f'{family}-{c}-{suffix}.yml'] for c in ['CDR','CCC']]
        familytests=[]
        for k,m in enumerate(METRICS):
            alignment.append(dict(family=family,metric=m,identical_baseline_rows=int(np.sum(a[:,0,k]==b[:,0,k])),n=len(a)))
            for phase in [1,2]:
                familytests.append(dict(family=family,phase=phase+1,metric=m,estimand='phase difference CDR minus CCC',**contrast(a[:,phase,k],b[:,phase,k])))
                familytests.append(dict(family=family,phase=phase+1,metric=m,estimand='change from phase 1 CDR minus CCC',**contrast(a[:,phase,k]-a[:,0,k],b[:,phase,k]-b[:,0,k])))
        tests.extend(holm(familytests))
    save('primary_contrasts.csv',tests); save('baseline_alignment.csv',alignment)
    supp=[]; ref=arrays['HRL-CDR-10000.yml']
    for name in SUPPLEMENT:
        a=arrays[name]
        for phase in [1,2]:
            for k,m in enumerate(METRICS):
                supp.append(dict(file=name,reference='HRL-CDR-10000.yml',phase=phase+1,metric=m,**contrast(a[:,phase,k]-a[:,0,k],ref[:,phase,k]-ref[:,0,k])))
    save('supplement_contrasts.csv',holm(supp))
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    labels=['Signed mean h','Logged deviation (not standard deviation)','Logged evaluation score']
    for row,(family,suffix) in enumerate([('HRL','10000'),('NF','5000')]):
        for k in range(3):
            ax=axes[row,k]
            for condition,color,offset in [('CDR','#c66328',-.035),('CCC','#247a98',.035)]:
                a=arrays[f'{family}-{condition}-{suffix}.yml'][:,:,k]
                ax.errorbar(np.arange(1,4)+offset,a.mean(0),yerr=stats.t.ppf(.975,len(a)-1)*stats.sem(a,axis=0),marker='o',capsize=3,label=condition,color=color)
            ax.set_title(f'{family}: {labels[k]}',fontsize=10);ax.set_xticks([1,2,3],['Pre','During','Post']);ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Primary files: trial means and unadjusted 95% intervals\nSeparate selected policies; scores are not comparable across reward families',fontsize=12)
    fig.savefig(ROOT/'primary_results.png',dpi=160);fig.savefig(ROOT/'primary_results.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    names=['HRL-CDR-10000.yml']+SUPPLEMENT
    for k,ax in enumerate(axes):
        samples=[arrays[name][:,2,k] for name in names]
        ax.errorbar(range(4),[x.mean() for x in samples],yerr=[stats.t.ppf(.975,len(x)-1)*stats.sem(x) for x in samples],marker='o',capsize=4,color='#247a98')
        ax.set_xticks(range(4),['Primary','4','6','7']);ax.set_xlabel('Decanalized counted visits per entry');ax.set_title(labels[k],fontsize=10);ax.grid(alpha=.2)
    fig.suptitle('Supplement only: post-intervention HRL outcomes\nRates 4, 6, and 7 confirmed by the author; unadjusted trial-level 95% intervals',fontsize=12)
    fig.savefig(ROOT/'supplement_results.png',dpi=160);fig.savefig(ROOT/'supplement_results.pdf');plt.close(fig)
    print('Validated:',[(m['file'],m['trials']) for m in manifest])
    for r in tests:
        if r['phase']==3 and r['metric'] in METRICS[:3] and r['estimand'].startswith('change'): print(r)
    for name,a in arrays.items(): print(name,'post means',dict(zip(METRICS,np.round(a[:,2,:].mean(0),4))))
if __name__=='__main__': main()
