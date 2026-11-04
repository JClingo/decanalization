"""Per-phase evaluation behind the paper's Table 2 and phase figure (Figure 5).

CCC vs CDR (d = 1, boost 1, 100 s) for the four stable-trained policies per family, 64 paired trials each
(evaluation seeds 5000-5063), changing world. Per trial and phase: mean |h| (mae), mean h, trapped
(h <= -100 during the phase), harvests, and harvests from the two new patches (rich). Trials are collapsed
to per-training-seed means, the unit of inference.

Usage: python eval_phases.py [OUT.json]     (default runs/phases_by_seed.json; published copy
       ../results/evaluations/phases_by_seed.json). About 10 minutes on 8 cores.
"""
import json,os,statistics as st,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from settings import WORLD
HERE=Path(__file__).resolve().parent
TAGS=['_stable','_stable_s1','_stable_s2','_stable_s3']
METRICS=('mae','trapped','mean_h','rich','harvests')
DESCRIPTION=('Per-training-seed means over 64 paired trials (evaluation seeds 5000-5063), stable-trained policies in the '
             'changing world. CDR: dose 1, entropy boost 1, 100 s. Phases 0-300, 300-400, 400-700 s. Metrics per phase: '
             'mae = mean |h|; trapped = fraction of trials with h <= -100 during the phase; mean_h = mean h; rich = harvests '
             'from the two new patches.')

def policy(family,tag):
    local=HERE/'runs'/f'policy_{family}{tag}.npz'
    return local if local.exists() else HERE.parent/'results'/'policies'/f'policy_{family}{tag}.npz'

def run(args):
    family,tag,seed,arm=args
    w=list(np.load(policy(family,tag)).values())
    p=MapParams(**WORLD)
    dose=phase_dose(p,'DOSE',1.,100.) if arm=='CDR' else phase_dose(p,'CCC')
    t=OptionTrial(p,seed,dose,family,w,temp_boost=1. if arm=='CDR' else 0.).run()
    a,b,c=p.phases;bounds=[(0,a),(a,a+b),(a+b,a+b+c)];S=t.log['samples'];H=t.log['harvests']
    ph=lambda lo,hi:[s for s in S if lo<s[0]<=hi]
    return family,tag,seed,arm,dict(
        mae=[float(st.mean(abs(s[3]) for s in ph(*bd))) for bd in bounds],
        mean_h=[float(st.mean(s[3] for s in ph(*bd))) for bd in bounds],
        trapped=[bool(min(s[3] for s in ph(*bd))<=-100) for bd in bounds],
        harvests=[sum(1 for h in H if lo<h[0]<=hi) for lo,hi in bounds],
        rich=[sum(1 for h in H if h[1] in t.rich and lo<h[0]<=hi) for lo,hi in bounds])

def collapse(done):
    seeds=[]
    for i,tag in enumerate(TAGS):
        entry=dict(training_seed=i,policy_tag=tag)
        for f in ('HRL','NF'):
            entry[f]={}
            for arm in ('CCC','CDR'):
                rs=[r for ff,tg,_,a,r in done if ff==f and tg==tag and a==arm]
                entry[f][arm]={m:[float(st.mean(float(r[m][k]) for r in rs)) for k in range(3)] for m in METRICS}
        seeds.append(entry)
    return dict(description=DESCRIPTION,seeds=seeds)

if __name__=='__main__':
    os.chdir(HERE)
    out=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'runs'/'phases_by_seed.json';out.parent.mkdir(parents=True,exist_ok=True)
    jobs=[(f,tg,s,arm) for f in ('HRL','NF') for tg in TAGS for s in range(5000,5064) for arm in ('CCC','CDR')]
    with Pool() as pool:done=pool.map(run,jobs)
    out.write_text(json.dumps(collapse(done),indent=1));print('wrote',out)
