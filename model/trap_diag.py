"""Diagnose map-agent trapping: what is the agent pursuing while trapped (h <= -100)?"""
import json,statistics as st,sys
from multiprocessing import Pool
from mapagent import MapParams,MapTrial
from world import phase_dose
BASE=dict(q_lo=3,q_hi=15,h_seek=-3,h_avoid=8,v_explore=4.)
def run(args):
    seed,agent=args;p=MapParams(**BASE,**(dict(P_reward=0.,tau_P=20.) if agent=='LOWPREC' else {}))
    t=MapTrial(p,seed,phase_dose(p,'CCC'));t.diag=True;t.run();D=t.log['diag']
    trapped=[d for d in D if d[1]<=-100];hungry=[d for d in D if d[2]=='seek' and d[1]>-100]
    def profile(rows):
        if not rows:return None
        tg=[d for d in rows if d[3] is not None]
        return dict(samples=len(rows),pursuing_memory=len(tg)/len(rows),
            target_empty=st.mean(not d[7] for d in tg) if tg else None,target_degraded=st.mean((d[9] is not None and d[9]<0.5) for d in tg) if tg else None,
            target_precision=st.median(d[5] for d in tg) if tg else None,believed_worth=st.median(d[4] for d in tg) if tg else None,
            true_value=st.median(d[10]*d[7] for d in tg) if tg else None,distinct_targets=len({d[3] for d in tg}),
            distinct_patches=len({d[8] for d in tg}),target_distance=st.median(d[6] for d in tg) if tg else None)
    return seed,agent,dict(trapped=profile(trapped),hungry_untrapped=profile(hungry),ever_trapped=bool(trapped))
if __name__=='__main__':
    with Pool() as pool:out=pool.map(run,[(s,a) for s in range(1000,1032) for a in ('MAP','LOWPREC')])
    res={}
    for s,a,r in out:res.setdefault(a,{})[s]=r
    def agg(rows,k):
        vals=[r[k] for r in rows if r and r.get(k) is not None];return round(st.mean(vals),3) if vals else None
    for a in res:
        tr=[r['trapped'] for r in res[a].values() if r['trapped']];hu=[r['hungry_untrapped'] for r in res[a].values() if r['hungry_untrapped']]
        print(a,'trials ever trapped',sum(r['ever_trapped'] for r in res[a].values()),'/',len(res[a]))
        for label,rows in (('  trapped',tr),('  hungry, not trapped',hu)):
            print(label,{k:agg(rows,k) for k in ('pursuing_memory','target_empty','target_degraded','target_precision','believed_worth','true_value','distinct_targets','distinct_patches','target_distance')})
    json.dump(res,open('runs/trap_diag.json','w'),indent=1,default=str)
