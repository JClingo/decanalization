"""Stage-A cost sweep: two-sided costs so canalization can be adaptive and decanalization dosed.

Base: the run-2 CHOSEN setting. Grid: breadth_cost {0.5, 1, 2} x dec_noise {0.5, 1, 2}.
Criteria, fixed before running (seeds 1000+, 32 paired trials):
  S1  stable world (degrade = 0): CCC phase-3 mean MAE below WIDE (paired CI upper < 0),
      i.e. canalization pays when ruts do not degrade.
  G1' degrading world: CCC phase-3 trap rate above WIDE (paired CI lower > 0).
  G2' degrading world: d = 1 for 100 s lowers phase-3 trap rate vs CCC (paired CI upper < 0).
  G3U inverted U over d in {0.25, 0.5, 0.75, 1} (100 s): the lowest phase-3 trap rate is at
      d = 0.5 or 0.75 and is at least 0.08 below both d = 0.25 and d = 1.
Choice: among configs passing all four, the deepest U (min gap to the two ends).
"""
import itertools,json,statistics as st,sys
from multiprocessing import Pool
from world import Params,Trial,phase_dose
from gates import summarize
from confirm_dose import CHOSEN,paired

DOSES=(0.25,0.5,0.75,1.)
def run(args):
    p,seed,arm,d=args
    dose=phase_dose(p,'DOSE',d,100.) if arm=='DOSE' else phase_dose(p,arm)
    return seed,(arm,d),summarize(p,Trial(p,seed,dose).run())

def evaluate(cfg,seeds,pool):
    deg,stable=Params(**cfg),Params(**dict(cfg,degrade=0.))
    jobs=[(stable,s,a,None) for s in seeds for a in ('CCC','WIDE')]
    done_s=pool.map(run,jobs)
    jobs=[(deg,s,a,None) for s in seeds for a in ('CCC','WIDE')]+[(deg,s,'DOSE',d) for s in seeds for d in DOSES]
    done_d=pool.map(run,jobs)
    S,D={},{}
    for seed,key,r in done_s:S.setdefault(key,{})[seed]=r
    for seed,key,r in done_d:D.setdefault(key,{})[seed]=r
    mae3=lambda r:r['mae'][2];trap3=lambda r:float(r['trapped'][2])
    s1=paired(S[('CCC',None)],S[('WIDE',None)],mae3);g1=paired(D[('CCC',None)],D[('WIDE',None)],trap3)
    g2=paired(D[('DOSE',1.)],D[('CCC',None)],trap3)
    trap={d:st.mean(trap3(r) for r in D[('DOSE',d)].values()) for d in DOSES}
    best=min(DOSES,key=lambda d:trap[d]);gap=min(trap[0.25],trap[1.])-trap[best]
    out=dict(cfg=cfg,S1=s1,G1=g1,G2=g2,trap_by_dose=trap,ccc_trap3=st.mean(trap3(r) for r in D[('CCC',None)].values()),
             wide_trap3=st.mean(trap3(r) for r in D[('WIDE',None)].values()),
             stable_mae3=dict(CCC=st.mean(mae3(r) for r in S[('CCC',None)].values()),WIDE=st.mean(mae3(r) for r in S[('WIDE',None)].values())),
             mae2_by_dose={d:st.mean(r['mae'][1] for r in D[('DOSE',d)].values()) for d in DOSES},
             mae3_by_dose={d:st.mean(mae3(r) for r in D[('DOSE',d)].values()) for d in DOSES},best_dose=best,u_gap=gap)
    out['pass']=dict(S1=s1['ci'][1]<0,G1=g1['ci'][0]>0,G2=g2['ci'][1]<0,G3U=best in (0.5,0.75) and gap>=0.08)
    return out

if __name__=='__main__':
    trials=int(sys.argv[1]) if len(sys.argv)>1 else 32;seeds=range(1000,1000+trials);rows=[]
    with Pool() as pool:
        grid=json.loads(sys.argv[2]) if len(sys.argv)>2 else dict(bc=[0.5,1.,2.],dn=[0.5,1.,2.],extra={})
        for bc,dn in itertools.product(grid['bc'],grid['dn']):
            r=evaluate(dict(CHOSEN,breadth_cost=bc,dec_noise=dn,**grid.get('extra',{})),seeds,pool);rows.append(r)
            print(json.dumps(dict(bc=bc,dn=dn,passes=r['pass'],S1=[round(r['S1']['mean'],1)]+[round(x,1) for x in r['S1']['ci']],
                stable=r['stable_mae3'],trap_ccc=r['ccc_trap3'],trap_wide=r['wide_trap3'],trap_dose=r['trap_by_dose'],mae2=r['mae2_by_dose']),default=lambda o:round(o,2)),flush=True)
    ok=[r for r in rows if all(r['pass'].values())];choice=max(ok,key=lambda r:r['u_gap']) if ok else None
    json.dump(dict(trials=trials,criterion=__doc__,rows=rows,choice=choice),open(sys.argv[3] if len(sys.argv)>3 else 'runs/cost_sweep.json','w'),indent=1,default=str)
    print('CHOICE',json.dumps(choice['cfg'] if choice else None))
