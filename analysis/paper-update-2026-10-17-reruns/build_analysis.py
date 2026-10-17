"""Reproducible schema-2 primary analysis; reject mislabeled policy snapshots.

Uses only frozen inputs selected in selection.json. Scene/snapshot files and
completion receipts must accompany every input. Legacy sensitivity data are
packaged separately and never pooled with these measurements.
"""
from pathlib import Path
import csv,gzip,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'Saved/AnalysisPython'))
import numpy as np
from scipy import stats
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METRICS=['homeostatic-mae','setpoint-dev','reward-total','homeostatic-fraction-within-5','homeostatic-rms','setpoint-avg','loss-total','gain-total','gains-harvested']
LABELS={'homeostatic-mae':'Mean absolute h','setpoint-dev':'Population SD of h','reward-total':'Actual objective reward','homeostatic-fraction-within-5':'Fraction of samples within ±5','homeostatic-rms':'RMS h','setpoint-avg':'Signed mean h','loss-total':'Total h loss','gain-total':'Total h gain','gains-harvested':'Pellet harvest count'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,rows):
    if not rows:return
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(run_id,family,order):
    base=ROOT/'inputs';p=base/(run_id+'.metrics.json');run=json.loads(p.read_text());y=base/(run_id+'.yml');data=yaml.safe_load(y.read_text())
    assert run['status']=='complete' and run['schema-version']==data['schema-version']==2
    assert data['duration-unit']=='simulation-seconds' and data['reward-source']=='TrainingEnv.GatherAgentReward'
    assert data['setpoint-dev-definition']=='population-standard-deviation'
    v=run['provenance'];record=v['matched-training-record']
    for name,h in v['logger-sha256'].items():
        legacy=ROOT/'evidence'/name;version=ROOT/'evidence/logger_versions'/h/name
        assert (legacy.exists() and sha(legacy)==h) or (version.exists() and sha(version)==h),'Logger evidence mismatch'
    assert record and v['training-record-match-count']==1 and record['reward-family']==family,'Loaded snapshot family mismatch'
    assert record['snapshot-update-label']==(10000 if family=='HRL' else 5000)
    assert v['environment']['IsHRL']==(family=='HRL') and v['environment']['TrialSequenceOrder']==order
    assert v['simulation-dt-seconds']==.05 and v['phase-decision-ticks']==2000
    receipt=json.loads((base/(run_id+'.completed.json')).read_text());assert receipt['results_saved'] and receipt['trials_completed']==receipt['trials_requested']==250
    for kind,s in v['snapshots'].items():
        assert sha(base/s['loaded-snapshot'])==s['loaded-sha256']==record['snapshot-sha256'][kind]
    trials=sorted(run['trials'],key=lambda t:t['id']);logged=sorted(data['trials'],key=lambda t:t['id'])
    assert [t['id'] for t in trials]==[t['id'] for t in logged]==list(range(250))
    rows=[];flat=[];rates=set();samples=[]
    for t,l in zip(trials,logged):
        assert t['reward-function']==l['reward-function']==family and t['sequence-order']==l['sequence-order']==order
        ps=sorted(t['phases'],key=lambda p:p['sequence-idx']);ls=sorted(l['sequences'],key=lambda p:p['sequence-idx'])
        assert [p['sequence-idx'] for p in ps]==[p['sequence-idx'] for p in ls]==[1,2,3]
        assert [p['state'] for p in ls]==(['canalized']*3 if order=='ccc' else ['canalized','decanalized','recanalized'])
        row=[]
        for phase,q in zip(ps,ls):
            assert phase['complete'] and phase['decision-ticks']==2000 and phase['duration-seconds']==q['duration']==100
            for k in ['reward-total','setpoint-avg','setpoint-dev']:
                assert math.isclose(phase[k],q[k],rel_tol=1e-6,abs_tol=1e-6),(run_id,t['id'],k)
            assert phase['homeostatic-sample-count']>0 and phase['setpoint-dev']>=0 and phase['homeostatic-mae']>=0
            assert 0<=phase['homeostatic-fraction-within-5']<=1 and phase['reward-total']>=0
            if family=='HRL':assert phase['reward-total']<=100.001
            assert math.isclose(phase['homeostatic-rms']**2,phase['setpoint-dev']**2+phase['setpoint-avg']**2,rel_tol=1e-8,abs_tol=1e-8)
            blob=gzip.decompress((base/phase['scene-file']).read_bytes());assert hashlib.sha256(blob).hexdigest()==phase['scene-sha256']
            scene=json.loads(blob);assert len(scene['tiles'])==phase['environment-tile-count']==2500
            assert len(scene['resources'])==phase['environment-pellet-count']
            rates.update(tile['DecanalizationRateFactor'] for tile in scene['tiles'])
            samples.append(phase['homeostatic-sample-count'])
            vals=[phase[k] if k in phase else q[k] for k in METRICS];assert np.isfinite(vals).all();row.append(vals)
            flat.append(dict(run=run_id,family=family,order=order,trial=t['id'],phase=phase['sequence-idx'],sample_count=phase['homeostatic-sample-count'],**dict(zip(METRICS,vals))))
        rows.append(row)
    return np.array(rows),run,flat,dict(run=run_id,family=family,order=order,trials=250,phases=750,loaded_snapshot_family=record['reward-family'],snapshot_update=record['snapshot-update-label'],tile_rate_factors=sorted(rates),sample_count_min=min(samples),sample_count_max=max(samples),yaml_sha256=sha(y),metrics_sha256=sha(p),all_scene_hashes_verified=True)
def interval(x):
    mean=float(np.mean(x));se=float(stats.sem(x));margin=float(stats.t.ppf(.975,len(x)-1))*se
    return dict(n=len(x),mean=mean,sd=float(np.std(x,ddof=1)),median=float(np.median(x)),ci95_low=mean-margin,ci95_high=mean+margin)
def compare(a,b,paired=False):
    delta=float(np.mean(a)-np.mean(b))
    if paired:
        d=a-b;se=float(stats.sem(d));df=len(d)-1
    else:
        va=float(np.var(a,ddof=1)/len(a));vb=float(np.var(b,ddof=1)/len(b));se=math.sqrt(va+vb)
        df=(va+vb)**2/(va*va/(len(a)-1)+vb*vb/(len(b)-1)) if se else math.inf
    margin=float(stats.t.ppf(.975,df))*se if se else 0.
    p=float(2*stats.t.sf(abs(delta/se),df)) if se else (1. if delta==0 else 0.)
    return dict(n_cdr=len(a),n_ccc=len(b),difference=delta,ci95_low=delta-margin,ci95_high=delta+margin,p=p)
def holm(rows):
    running=0.
    for rank,i in enumerate(np.argsort([r['p'] for r in rows])):
        running=max(running,min(1.,(len(rows)-rank)*rows[i]['p']));rows[i]['p_holm']=running
    return rows
def scene_diffs(a,b,path=''):
    if isinstance(a,dict):
        return [p for k in a for p in scene_diffs(a[k],b[k],path+'/'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [path+'/length']
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in scene_diffs(x,y,path+'/'+str(i))]
    return [] if a==b else [path]
def main():
    selected=json.loads((ROOT/'selection.json').read_text())
    assert set(selected)=={'HRL-CDR','HRL-CCC','NF-CDR','NF-CCC'} or '--allow-partial' in sys.argv,'Four selected primary conditions required'
    arrays={};runs={};summaries=[];records=[];flat=[]
    for condition,run_id in selected.items():
        family,order=condition.split('-');a,r,f,record=load(run_id,family,order.lower());arrays[condition]=a;runs[condition]=r;flat+=f;records.append(record)
        for phase in range(3):
            for k,m in enumerate(METRICS):summaries.append(dict(condition=condition,phase=phase+1,metric=m,**interval(a[:,phase,k])))
    save('phase_summary.csv',summaries);save('trial_metrics.csv',flat)
    contrasts=[];sensitivity=[];alignment=[]
    for family in ['HRL','NF']:
        if family+'-CDR' not in arrays or family+'-CCC' not in arrays:continue
        a,b=[arrays[family+'-'+c] for c in ['CDR','CCC']];ra,rb=[runs[family+'-'+c] for c in ['CDR','CCC']]
        exact=[];diffcounts={};startmatch={k:0 for k in ['rng-start','policy-memory-start','initial-homeostatic-state','observed-start-h']}
        for t,u in zip(ra['trials'],rb['trials']):
            p,q=t['phases'][0],u['phases'][0];same=p['scene-sha256']==q['scene-sha256'];exact.append(same)
            for k in startmatch:startmatch[k]+=int(p[k]==q[k])
            if not same:
                x,y=[json.loads(gzip.decompress((ROOT/'inputs'/z['scene-file']).read_bytes())) for z in [p,q]]
                for key in scene_diffs(x,y):diffcounts[key.rsplit('/',1)[-1]]=diffcounts.get(key.rsplit('/',1)[-1],0)+1
        exact=np.array(exact)
        alignment.append(dict(family=family,n=250,exact_scene_matches=int(exact.sum()),other_start_matches=startmatch,scene_difference_fields=diffcounts,identical_baseline_rows={m:int(np.sum(a[:,0,k]==b[:,0,k])) for k,m in enumerate(METRICS)}))
        for phase in [1,2]:
            for k,m in enumerate(METRICS):
                contrasts.append(dict(family=family,phase=phase+1,metric=m,method='Welch',estimand='CDR minus CCC',**compare(a[:,phase,k],b[:,phase,k])))
                for label,mask,change in [('paired-all',np.ones(250,dtype=bool),False),('paired-exact-scene',exact,False),('paired-change-from-baseline',np.ones(250,dtype=bool),True)]:
                    if mask.sum()<2:continue
                    x=a[mask,phase,k]-(a[mask,0,k] if change else 0);y=b[mask,phase,k]-(b[mask,0,k] if change else 0)
                    sensitivity.append(dict(family=family,phase=phase+1,metric=m,method=label,**compare(x,y,True)))
    for phase in [2,3]:holm([r for r in contrasts if r['phase']==phase])
    for method in ['paired-all','paired-exact-scene','paired-change-from-baseline']:
        for phase in [2,3]:holm([r for r in sensitivity if r['phase']==phase and r['method']==method])
    save('primary_contrasts.csv',contrasts);save('paired_sensitivity.csv',sensitivity)
    (ROOT/'validation.json').write_text(json.dumps(dict(passed=True,conditions=records,baseline_alignment=alignment),indent=2))
    for filename,metrics in [('primary_results',['homeostatic-mae','setpoint-dev','reward-total']),('regulation_results',['setpoint-avg','homeostatic-rms','homeostatic-fraction-within-5'])]:
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for row,family in enumerate(['HRL','NF']):
            for col,m in enumerate(metrics):
                ax=axes[row,col]
                for order,color,off in [('CDR','#b54f24',-.035),('CCC','#1b7089',.035)]:
                    key=family+'-'+order
                    if key not in arrays:continue
                    x=arrays[key][:,:,METRICS.index(m)]
                    ax.errorbar(np.arange(1,4)+off,x.mean(0),yerr=stats.t.ppf(.975,249)*stats.sem(x,axis=0),color=color,marker='o',capsize=3,label=order)
                ax.set_title(family+': '+LABELS[m],fontsize=10);ax.set_xticks([1,2,3],['Pre','During','Post']);ax.grid(alpha=.2)
        axes[0,0].legend();fig.suptitle('Corrected primary trials: means and unadjusted 95% intervals\nReward scales differ by objective; homeostatic summaries describe recorded samples',fontsize=11)
        fig.savefig(ROOT/(filename+'.png'),dpi=180);fig.savefig(ROOT/(filename+'.pdf'));plt.close(fig)
    for r in contrasts:
        if r['phase']==3 and r['metric'] in METRICS[:4]:print(r)
if __name__=='__main__':main()
