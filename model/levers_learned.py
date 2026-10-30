"""Lever decomposition for learned policies (distribution-shift test, 4 training seeds, fresh seeds 5000+).
Temperature-only cells (prec_lever 0) at temp_boost {1, 3}, d = 1, compared with the undisturbed agent;
precision-only (temp 0, d 1) and both levers come from the existing eval_policies runs.
Pre-set question: which lever carries the HRL benefit? Reported per lever with seed-level t-intervals."""
import json,statistics as st,math
from multiprocessing import Pool
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from gates import summarize
from settings import WORLD
TAGS=['_stable','_stable_s1','_stable_s2','_stable_s3'];T3=3.182
def run(args):
    f,tag,seed,b=args;w=list(np.load(f'runs/policy_{f}{tag}.npz').values());p=MapParams(**WORLD,prec_lever=0.)
    t=OptionTrial(p,seed,phase_dose(p,'DOSE',1.,100.),f,w,temp_boost=b).run();return f,tag,seed,b,summarize(p,t)['mae'][2]
if __name__=='__main__':
    with Pool() as pool:done=pool.map(run,[(f,t,s,b) for f in ('HRL','NF') for t in TAGS for s in range(5000,5064) for b in (1.,3.)])
    R={}
    for f,t,s,b,m in done:R.setdefault((f,t,b),{})[s]=m
    out={}
    for f in ('HRL','NF'):
        out[f]={}
        for t in TAGS:
            E=json.load(open(f'runs/eval_policies{t}.json'))[f];U=E['U']['mae'][2]
            for b in (1.,3.):out[f].setdefault(f'temp-only {b}',[]).append(st.mean(R[(f,t,b)].values())-U)
            out[f].setdefault('precision-only',[]).append(E['cells']['0.0,1.0']['mae_vs_U']['mean'])
            out[f].setdefault('both temp 1',[]).append(E['cells']['1.0,1.0']['mae_vs_U']['mean'])
            out[f].setdefault('both temp 3',[]).append(E['cells']['3.0,1.0']['mae_vs_U']['mean'])
        for k,v in out[f].items():
            m=st.mean(v);h=T3*st.stdev(v)/2;print(f,k,f'{m:+.1f} [{m-h:+.1f}, {m+h:+.1f}]',[round(x,1) for x in v])
    json.dump(out,open('runs/levers_learned.json','w'),indent=1)
