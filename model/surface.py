"""Dose x exploration-boost surface on fresh seeds (4000+, 64 paired trials), reliable-opportunity
world, map agent with directed exploration, both levers, 100 s window.
Grid: boost {0.5, 1, 2, 3} x dose {0.25, 0.5, 0.75, 1}. Intensity = d * (1 + boost).
Criteria fixed before running:
  Benefit   some cell beats the undisturbed agent on phase-3 MAE (paired CI upper < 0).
  Harm      the highest-intensity cell (boost 3, d = 1) is worse than undisturbed (paired CI lower > 0).
  Interior  the best cell (lowest mean phase-3 MAE) has intensity strictly inside the grid's range,
            and both the lowest- and highest-intensity cells are worse than it (paired CI of
            best - cell upper < 0).
"""
import json,statistics as st,sys
from multiprocessing import Pool
from map_levers2 import run
from confirm_dose import paired
OVER={"rich_q_lo":8,"rich_q_hi":12,"rich_degrade":0,"rich_delay":0.5}
BOOSTS=(0.5,1.,2.,3.);DOSES=(0.25,0.5,0.75,1.)
if __name__=='__main__':
    seeds=range(4000,4064)
    jobs=[(s,1.,0.,None,'MAP',OVER) for s in seeds]+[(s,1.,b,d,'MAP',OVER) for s in seeds for b in BOOSTS for d in DOSES]
    with Pool() as pool:done=pool.map(run,jobs)
    R={}
    for s,k,r in done:R.setdefault(k,{})[s]=r
    U=R[(1.,0.,None,'MAP')];mae3=lambda r:r['mae'][2];trap3=lambda r:float(r['trapped'][2]);rich=lambda r:r['rich23']
    cells={(b,d):R[(1.,b,d,'MAP')] for b in BOOSTS for d in DOSES}
    table={f'{b},{d}':dict(intensity=d*(1+b),mae3=st.mean(mae3(r) for r in X.values()),median=st.median(mae3(r) for r in X.values()),
           mae_vs_U=paired(X,U,mae3),trap_vs_U=paired(X,U,trap3),rich_vs_U=paired(X,U,rich),mae2=st.mean(r['mae'][1] for r in X.values()))
           for (b,d),X in cells.items()}
    best=min(cells,key=lambda k:table[f'{k[0]},{k[1]}']['mae3'])
    inten={k:k[1]*(1+k[0]) for k in cells};lo=min(cells,key=lambda k:inten[k]);hi=max(cells,key=lambda k:inten[k])
    benefit=any(v['mae_vs_U']['ci'][1]<0 for v in table.values())
    harm=table['3.0,1.0']['mae_vs_U']['ci'][0]>0
    vs_lo=paired(cells[best],cells[lo],mae3);vs_hi=paired(cells[best],cells[hi],mae3)
    interior=inten[lo]<inten[best]<inten[hi] and vs_lo['ci'][1]<0 and vs_hi['ci'][1]<0
    out=dict(undisturbed=dict(mae3=st.mean(mae3(r) for r in U.values()),median=st.median(mae3(r) for r in U.values()),trap3=st.mean(trap3(r) for r in U.values())),
             table=table,best=best,best_vs_lowest=vs_lo,best_vs_highest=vs_hi,criteria=dict(benefit=benefit,harm=harm,interior=interior))
    json.dump(out,open('runs/surface.json','w'),indent=1,default=str)
    print('undisturbed',{k:round(v,2) for k,v in out['undisturbed'].items()})
    print('phase-3 MAE vs undisturbed (rows: boost; cols: dose 0.25 0.5 0.75 1)')
    for b in BOOSTS:
        print(f' boost {b}:','  '.join(f"{table[f'{b},{d}']['mae_vs_U']['mean']:+6.1f} [{table[f'{b},{d}']['mae_vs_U']['ci'][0]:+.0f},{table[f'{b},{d}']['mae_vs_U']['ci'][1]:+.0f}]" for d in DOSES))
    print('trap vs undisturbed')
    for b in BOOSTS:print(f' boost {b}:','  '.join(f"{table[f'{b},{d}']['trap_vs_U']['mean']:+.2f}" for d in DOSES))
    print('best',best,'intensity',inten[best],'| best-lowest',round(vs_lo['mean'],1),[round(x,1) for x in vs_lo['ci']],'| best-highest',round(vs_hi['mean'],1),[round(x,1) for x in vs_hi['ci']])
    print('CRITERIA',out['criteria'])
