"""When agents first harvest from the new patches, behind the Results and Limitations statements on late discovery.

CCC vs CDR (d = 1, boost 1, 100 s) for the four stable-trained policies per family, 64 paired trials each
(evaluation seeds 5000-5063), changing world; the same trials as eval_phases.py. Per trial: time of the first
harvest from either new patch (None if none by 700 s), h at the start of the recanalization phase, and h at the
end of the trial. Summary per family and arm, pooled over trials: fraction that never harvested from the new
patches, median first-harvest time among those that did, and fraction ending at h <= -100.

Usage: python eval_first_harvest.py [OUT.json]     (default runs/first_harvest.json; published copy
       ../results/evaluations/first_harvest.json). About 3 minutes on 8 cores.
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

def policy(family,tag):
    local=HERE/'runs'/f'policy_{family}{tag}.npz'
    return local if local.exists() else HERE.parent/'results'/'policies'/f'policy_{family}{tag}.npz'

def run(args):
    family,tag,seed,arm=args
    w=list(np.load(policy(family,tag)).values())
    p=MapParams(**WORLD)
    dose=phase_dose(p,'DOSE',1.,100.) if arm=='CDR' else phase_dose(p,'CCC')
    t=OptionTrial(p,seed,dose,family,w,temp_boost=1. if arm=='CDR' else 0.).run()
    a,b,c=p.phases;rich=[h[0] for h in t.log['harvests'] if h[1] in t.rich]
    h3=[s[3] for s in t.log['samples'] if a+b<s[0]<=a+b+c]
    return dict(family=family,policy_tag=tag,seed=seed,arm=arm,first_new=min(rich) if rich else None,h_start3=float(h3[0]),h_end=float(h3[-1]))

def summarize(rows):
    found=[r['first_new'] for r in rows if r['first_new'] is not None]
    return dict(n=len(rows),never=sum(r['first_new'] is None for r in rows)/len(rows),median_first=st.median(found) if found else None,
                after550=sum(x>550 for x in found)/len(rows),median_h_start3=st.median(r['h_start3'] for r in rows),
                median_h_end=st.median(r['h_end'] for r in rows),end_trapped=sum(r['h_end']<=-100 for r in rows)/len(rows))

if __name__=='__main__':
    os.chdir(HERE)
    out=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'runs'/'first_harvest.json';out.parent.mkdir(parents=True,exist_ok=True)
    jobs=[(f,tg,s,arm) for f in ('HRL','NF') for tg in TAGS for s in range(5000,5064) for arm in ('CCC','CDR')]
    with Pool() as pool:done=pool.map(run,jobs)
    summary={f:{arm:summarize([r for r in done if r['family']==f and r['arm']==arm]) for arm in ('CCC','CDR')} for f in ('HRL','NF')}
    out.write_text(json.dumps(dict(description=__doc__.split('\n\n')[1].replace('\n',' '),summary=summary,trials=done),indent=1))
    for f,v in summary.items():
        for arm,s in v.items():print(f,arm,{k:(round(x,3) if isinstance(x,float) else x) for k,x in s.items()})
    print('wrote',out)
