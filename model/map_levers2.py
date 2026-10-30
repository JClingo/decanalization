"""Lever test in the opportunity world: 2 rich patches (quality 25-35) appear at the phase-2 start,
>= 20 tiles from the agent. Map agent (v_explore 4, precision-only flattening), degrading world,
32 paired trials, seeds 1000+. Grid: directed_explore {0, 1} x explore_boost {0.5, 1, 2}; doses
d in {0.25, 0.5, 0.75, 1} for 100 s with both levers. Criteria fixed before running, per config:
  G2  d = 0.5 beats the undisturbed agent on phase-3 MAE or trap rate (paired CI upper < 0).
  G3U the best dose by phase-3 MAE is 0.5 or 0.75, beats the undisturbed agent (as in G2), and
      its MAE is below both d = 0.25 and d = 1.
  Escape endpoint (reported): rich-patch harvests in phases 2-3, dose minus undisturbed.
  G1  (reported) undisturbed MAP vs LOWPREC phase-3 trap rate in this world.
"""
import json,statistics as st,sys
from multiprocessing import Pool
from mapagent import MapParams,MapTrial
from world import phase_dose
from gates import summarize
from confirm_dose import paired
BASE=dict(q_lo=3,q_hi=15,h_seek=-3,h_avoid=8,v_explore=4.,flatten_values=0.,rich_patches=2,rich_appear=300.)
DOSES=(0.25,0.5,0.75,1.)
def run(args):
    seed,de,boost,d,agent,over=args   # overrides travel with the job (spawned workers re-import BASE)
    p=MapParams(**dict(BASE,**over),directed_explore=de,explore_boost=boost,**(dict(P_reward=0.,tau_P=20.) if agent=='LOWPREC' else {}))
    t=MapTrial(p,seed,phase_dose(p,'DOSE',d,100.) if d else phase_dose(p,'CCC')).run()
    r=summarize(p,t);a=p.phases[0];r['rich23']=sum(1 for h in t.log['harvests'] if h[1] in t.rich and h[0]>a);return seed,(de,boost,d,agent),r
if __name__=='__main__':
    over=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
    seeds=range(1000,1032);jobs=[]
    for de in (0.,1.):
        jobs+=[(s,de,0.,None,a,over) for s in seeds for a in ('MAP','LOWPREC')]
        for b in (0.5,1.,2.):jobs+=[(s,de,b,d,'MAP',over) for s in seeds for d in DOSES]
    with Pool() as pool:done=pool.map(run,jobs)
    R={}
    for s,k,r in done:R.setdefault(k,{})[s]=r
    mae3=lambda r:r['mae'][2];trap3=lambda r:float(r['trapped'][2]);rich=lambda r:r['rich23']
    beats=lambda X,U:paired(X,U,mae3)['ci'][1]<0 or paired(X,U,trap3)['ci'][1]<0
    out={}
    for de in (0.,1.):
        U,L=R[(de,0.,None,'MAP')],R[(de,0.,None,'LOWPREC')]
        g1=paired(U,L,trap3);print(f'directed {de}: undisturbed mae3 {st.mean(mae3(r) for r in U.values()):.1f} trap3 {st.mean(trap3(r) for r in U.values()):.2f} rich23 {st.mean(rich(r) for r in U.values()):.1f}; G1 MAP-LOWPREC trap {g1["mean"]:.2f} {[round(x,2) for x in g1["ci"]]}')
        out[de]=dict(G1=g1,boosts={})
        for b in (0.5,1.,2.):
            X={d:R[(de,b,d,'MAP')] for d in DOSES};by={d:st.mean(mae3(r) for r in X[d].values()) for d in DOSES};best=min(DOSES,key=lambda d:by[d])
            g2=beats(X[0.5],U);g3=best in (0.5,0.75) and beats(X[best],U) and by[best]<min(by[0.25],by[1.])
            out[de]['boosts'][b]=dict(G2=g2,G3U=g3,best=best,doses={d:dict(mae3=by[d],mae_vs_U=paired(X[d],U,mae3),trap_vs_U=paired(X[d],U,trap3),rich_vs_U=paired(X[d],U,rich)) for d in DOSES})
            print(f'  boost {b}: G2 {g2} G3U {g3} best {best}')
            for d in DOSES:
                v=out[de]['boosts'][b]['doses'][d]
                print(f"    d={d} mae3 {v['mae3']:.1f} | dMAE {v['mae_vs_U']['mean']:.1f} {[round(x,1) for x in v['mae_vs_U']['ci']]} | dtrap {v['trap_vs_U']['mean']:.2f} {[round(x,2) for x in v['trap_vs_U']['ci']]} | drich {v['rich_vs_U']['mean']:.1f} {[round(x,1) for x in v['rich_vs_U']['ci']]}")
    json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else 'runs/map_levers2.json','w'),indent=1,default=str)
