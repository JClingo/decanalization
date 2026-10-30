"""Decanalization levers for the map agent (seeds 1000+, 32 paired trials, degrading world, v_explore 4).
Levers: precision reduction only (prec_lever 1, boost 0, already tested), exploration boost only
(prec_lever 0), and both. Values are never flattened (flatten_values 0). Criteria fixed before running:
  G2  both levers at d = 0.5: phase-3 trap rate or MAE below the undisturbed agent (paired CI upper < 0).
  G3U both levers over d in {0.25, 0.5, 0.75, 1}: the best dose is 0.5 or 0.75, beats the undisturbed
      agent (paired CI upper < 0 on MAE or trap), and its phase-3 MAE is below both d = 0.25 and d = 1.
Reported for each explore_boost in {1, 2, 4}; also the explore-only decomposition at d = 0.5 and 1.
"""
import json,statistics as st,sys
from multiprocessing import Pool
from mapagent import MapParams,MapTrial
from world import phase_dose
from gates import summarize
from confirm_dose import paired
BASE=dict(q_lo=3,q_hi=15,h_seek=-3,h_avoid=8,v_explore=4.,flatten_values=0.)
DOSES=(0.25,0.5,0.75,1.)
def run(args):
    seed,boost,lever,d=args;p=MapParams(**BASE,explore_boost=boost,prec_lever=lever)
    t=MapTrial(p,seed,phase_dose(p,'DOSE',d,100.) if d else phase_dose(p,'CCC'));return seed,(boost,lever,d),summarize(p,t.run())
if __name__=='__main__':
    seeds=range(1000,1032);jobs=[(s,0.,1.,None) for s in seeds]
    for b in (1.,2.,4.):jobs+=[(s,b,1.,d) for s in seeds for d in DOSES]+[(s,b,0.,d) for s in seeds for d in (0.5,1.)]
    with Pool() as pool:done=pool.map(run,jobs)
    R={}
    for s,k,r in done:R.setdefault(k,{})[s]=r
    U=R[(0.,1.,None)];mae3=lambda r:r['mae'][2];trap3=lambda r:float(r['trapped'][2])
    print('undisturbed: mae3',round(st.mean(mae3(r) for r in U.values()),1),'trap3',round(st.mean(trap3(r) for r in U.values()),2))
    out=dict(undisturbed=dict(mae3=st.mean(mae3(r) for r in U.values()),trap3=st.mean(trap3(r) for r in U.values())),boosts={})
    for b in (1.,2.,4.):
        rows={}
        for lever,ds in ((1.,DOSES),(0.,(0.5,1.))):
            for d in ds:
                X=R[(b,lever,d)];rows[f"{'both' if lever else 'explore-only'} d={d}"]=dict(mae3=st.mean(mae3(r) for r in X.values()),trap3=st.mean(trap3(r) for r in X.values()),
                    mae_vs_U=paired(X,U,mae3),trap_vs_U=paired(X,U,trap3),mae2=st.mean(r['mae'][1] for r in X.values()))
        both={d:rows[f'both d={d}'] for d in DOSES};best=min(DOSES,key=lambda d:both[d]['mae3'])
        beats=lambda v:v['mae_vs_U']['ci'][1]<0 or v['trap_vs_U']['ci'][1]<0
        g2=beats(both[0.5]);g3=best in (0.5,0.75) and beats(both[best]) and both[best]['mae3']<min(both[0.25]['mae3'],both[1.]['mae3'])
        out['boosts'][b]=dict(rows=rows,G2=g2,G3U=g3,best=best)
        print(f'boost {b}: G2 {g2} G3U {g3} best {best}')
        for k,v in rows.items():print('  ',k,'mae3',round(v['mae3'],1),'trap3',round(v['trap3'],2),'| vs undisturbed mae',round(v['mae_vs_U']['mean'],1),[round(x,1) for x in v['mae_vs_U']['ci']],'trap',round(v['trap_vs_U']['mean'],2),[round(x,2) for x in v['trap_vs_U']['ci']],'| mae2',round(v['mae2'],1))
    json.dump(out,open('runs/map_levers.json','w'),indent=1,default=str)
