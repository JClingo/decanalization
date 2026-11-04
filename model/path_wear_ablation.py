"""Path-wear ablation: does path wear contribute to trapping?

The stable-trained policies (four training seeds per family) are evaluated in the changing world with
the usual worn-path costs, and again with traversal cost made flat (the same on every tile, so wear has
no effect). Flat levels are given on the command line: the mean cost per second agents actually pay
with wear (so the energy budget is roughly matched), and c = 1, the cost of a fully worn path.
CCC and CDR (d = 1, boost 1), 64 paired trials per seed (evaluation seeds 5000-5063). Policy weights
are unchanged; the policy's tile-cost input simply becomes constant.

The worn-path baseline is read from ../results/evaluations/phases_by_seed.json (same trials); pass --wear
to rerun it here as well. Set N_TRIALS for a quick test.

Usage: python path_wear_ablation.py OUT.json [--wear] [flat cost ...]      e.g. 2.6
       python path_wear_ablation.py --pilot                       mean cost paid with wear (HRL seed 0, CCC)
"""
import json,math,os,statistics as st,sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from settings import WORLD
POL=Path(__file__).resolve().parent.parent/'results'/'policies'
TAGS=['_stable','_stable_s1','_stable_s2','_stable_s3'];SEEDS=range(5000,5000+int(os.environ.get('N_TRIALS',64)));T3=3.182

class Costed(OptionTrial):
    def step(self,action=None):
        super().step(action);self.paid.append(self.rate)

def run(args):
    family,tag,seed,arm,flat=args
    w=list(np.load(POL/f'policy_{family}{tag}.npz').values())
    p=MapParams(**WORLD,**({} if flat is None else dict(c0=flat,c_min=flat)))
    dose=phase_dose(p,'DOSE',1.,100.) if arm=='CDR' else phase_dose(p,'CCC')
    t=Costed(p,seed,dose,family,w,temp_boost=1. if arm=='CDR' else 0.);t.paid=[];t.run()
    a,b,c=p.phases;B=[(0,a),(a,a+b),(a+b,a+b+c)];S=t.log['samples'];H=t.log['harvests']
    ph=lambda lo,hi:[s[3] for s in S if lo<s[0]<=hi]
    return dict(family=family,tag=tag,seed=seed,arm=arm,flat=flat,mean_cost=float(np.mean(t.paid)),
                mae=[float(np.mean(np.abs(ph(*x)))) for x in B],mean_h=[float(np.mean(ph(*x))) for x in B],
                trapped=[bool(min(ph(*x))<=-100) for x in B],
                rich=[sum(1 for h in H if h[1] in t.rich and lo<h[0]<=hi) for lo,hi in B])

def ci(v):
    m=st.mean(v);return m,T3*st.stdev(v)/math.sqrt(len(v))

def summarize(rows):
    """Per-seed means, then mean and 95% t-interval over the four training seeds."""
    out={}
    for f in ('HRL','NF'):
        for flat in sorted({r['flat'] for r in rows},key=lambda x:-1 if x is None else x):
            key=f"{f} {'wear' if flat is None else f'flat {flat:g}'}";out[key]={}
            per={tg:{arm:[r for r in rows if r['family']==f and r['tag']==tg and r['flat']==flat and r['arm']==arm]
                     for arm in ('CCC','CDR')} for tg in TAGS}
            m=lambda tg,arm,fn:st.mean(fn(r) for r in per[tg][arm])
            for name,fn in (('trapped_p3',lambda r:float(r['trapped'][2])),('trapped_p1',lambda r:float(r['trapped'][0])),
                            ('mae_p3',lambda r:r['mae'][2]),('mean_cost',lambda r:r['mean_cost'])):
                out[key][name+' CCC']=ci([m(tg,'CCC',fn) for tg in TAGS])
            out[key]['CDR-CCC mae_p3']=ci([m(tg,'CDR',lambda r:r['mae'][2])-m(tg,'CCC',lambda r:r['mae'][2]) for tg in TAGS])
            out[key]['CDR-CCC trapped_p3']=ci([m(tg,'CDR',lambda r:float(r['trapped'][2]))-m(tg,'CCC',lambda r:float(r['trapped'][2])) for tg in TAGS])
    return out

if __name__=='__main__':
    if sys.argv[1]=='--pilot':
        with Pool() as pool:rows=pool.map(run,[('HRL','_stable',s,'CCC',None) for s in SEEDS])
        print('mean cost paid with wear (HRL seed 0, CCC):',round(st.mean(r['mean_cost'] for r in rows),3));sys.exit()
    # With wear, the same trials are already in results/evaluations/phases_by_seed.json (from eval_phases.py),
    # so only flat-cost runs are made here unless --wear is given.
    args=sys.argv[1:];wear='--wear' in args;args=[a for a in args if a!='--wear']
    out=args[0];flats=([None] if wear else [])+[float(x) for x in args[1:]]
    jobs=[(f,tg,s,arm,fl) for fl in flats for f in ('HRL','NF') for tg in TAGS for s in SEEDS for arm in ('CCC','CDR')]
    with Pool() as pool:rows=pool.map(run,jobs,chunksize=4)
    S=summarize(rows)
    ref=json.loads((POL.parent/'evaluations'/'phases_by_seed.json').read_text())['seeds']
    for f in ('HRL','NF'):
        g=lambda arm,m,k:[s[f][arm][m][k] for s in ref]
        S[f+' wear (phases_by_seed.json)']={'trapped_p3 CCC':ci(g('CCC','trapped',2)),'trapped_p1 CCC':ci(g('CCC','trapped',0)),
            'mae_p3 CCC':ci(g('CCC','mae',2)),
            'CDR-CCC mae_p3':ci([b-a for a,b in zip(g('CCC','mae',2),g('CDR','mae',2))]),
            'CDR-CCC trapped_p3':ci([b-a for a,b in zip(g('CCC','trapped',2),g('CDR','trapped',2))])}
    json.dump(dict(description=__doc__,summary=S,trials=rows),open(out,'w'))
    for k,v in S.items():
        print(k);[print(f'  {n}: {a:.3f} ± {b:.3f}') for n,(a,b) in v.items()]
