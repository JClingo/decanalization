"""Gates for the cognitive-map agent (seeds 1000+, paired trials). Criteria fixed before running:
  S1  stable world (degrade = 0): MAP (precision learning) phase-3 time-MAE below LOWPREC
      (P_reward = 0, tau_P = 20 s; beliefs track the senses), paired CI upper < 0.
  G1  degrading world: MAP phase-3 trap rate above LOWPREC, paired CI lower > 0
      (or, if neither traps, MAP phase-3 MAE above LOWPREC).
  G2  degrading world: flattening at d = 0.5 for 100 s lowers MAP's phase-3 trap rate, or MAE,
      vs undisturbed MAP (paired CI upper < 0).
  G3U over d in {0.25, 0.5, 0.75, 1}: the best phase-3 MAE is at d = 0.5 or 0.75 and is lower
      than both d = 0.25 and d = 1 (point estimates).
NOMAP (reactive heuristic with full perception) is reported as a reference.
"""
import json,statistics as st,sys
from multiprocessing import Pool
from dataclasses import asdict
from world import Params,Trial,phase_dose
from mapagent import MapParams,MapTrial
from gates import summarize
from confirm_dose import paired

BASE=dict(q_lo=3,q_hi=15,h_seek=-3,h_avoid=8)
LOWPREC=dict(P_reward=0.,tau_P=20.);DOSES=(0.25,0.5,0.75,1.)

def run(args):
    world,seed,agent,d,over=args
    cfg=dict(BASE,**over,**({} if world=='degrading' else dict(degrade=0.)))
    if agent=='NOMAP':p=Params(**{k:v for k,v in cfg.items() if k in Params.__dataclass_fields__});t=Trial(p,seed,phase_dose(p,'WIDE'))
    else:
        p=MapParams(**cfg,**(LOWPREC if agent=='LOWPREC' else {}))
        t=MapTrial(p,seed,phase_dose(p,'DOSE',d,100.) if d else phase_dose(p,'CCC'))
    return seed,(world,agent,d),summarize(p,t.run())

if __name__=='__main__':
    trials=int(sys.argv[1]) if len(sys.argv)>1 else 32;over=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
    seeds=range(1000,1000+trials)
    jobs=[(w,s,a,None,over) for w in ('stable','degrading') for s in seeds for a in ('MAP','LOWPREC','NOMAP')]
    jobs+=[('degrading',s,'MAP',d,over) for s in seeds for d in DOSES]
    with Pool() as pool:done=pool.map(run,jobs)
    R={}
    for seed,key,r in done:R.setdefault(key,{})[seed]=r
    mae3=lambda r:r['mae'][2];trap3=lambda r:float(r['trapped'][2])
    desc=lambda rows:dict(mae=[st.mean(r['mae'][i] for r in rows) for i in range(3)],median_mae3=st.median(r['mae'][2] for r in rows),
                          trap=[st.mean(float(r['trapped'][i]) for r in rows) for i in range(3)],new_patches=st.mean(r['new_patches'] for r in rows))
    out=dict(over=over,trials=trials,arms={'/'.join(str(x) for x in k):desc(list(v.values())) for k,v in R.items()})
    M,L=R[('degrading','MAP',None)],R[('degrading','LOWPREC',None)]
    out['S1']=paired(R[('stable','MAP',None)],R[('stable','LOWPREC',None)],mae3);out['S1']['passed']=out['S1']['ci'][1]<0
    g1t,g1m=paired(M,L,trap3),paired(M,L,mae3)
    out['G1']=dict(trap=g1t,mae=g1m,passed=g1t['ci'][0]>0 or (g1t['mean']==0 and g1m['ci'][0]>0))
    g2t,g2m=paired(R[('degrading','MAP',0.5)],M,trap3),paired(R[('degrading','MAP',0.5)],M,mae3)
    out['G2']=dict(trap=g2t,mae=g2m,passed=g2t['ci'][1]<0 or g2m['ci'][1]<0)
    by={d:st.mean(mae3(r) for r in R[('degrading','MAP',d)].values()) for d in DOSES};best=min(DOSES,key=lambda d:by[d])
    out['G3U']=dict(mae3_by_dose=by,undisturbed=st.mean(mae3(r) for r in M.values()),best=best,passed=best in (0.5,0.75) and by[best]<min(by[0.25],by[1.]))
    out['dose_vs_MAP']={d:dict(mae=paired(R[('degrading','MAP',d)],M,mae3),trap=paired(R[('degrading','MAP',d)],M,trap3)) for d in DOSES}
    name=sys.argv[3] if len(sys.argv)>3 else 'runs/map_gates.json';json.dump(out,open(name,'w'),indent=1,default=str)
    show=lambda d:{k:(round(v,2) if isinstance(v,float) else [round(x,2) for x in v] if isinstance(v,list) else v) for k,v in d.items()}
    for k,v in out['arms'].items():print(k,show(v))
    for g in ('S1','G1','G2','G3U'):print(g,json.dumps(out[g],default=lambda o:round(float(o),2)))
    for d,v in out['dose_vs_MAP'].items():print('dose',d,'mae',round(v['mae']['mean'],1),[round(x,1) for x in v['mae']['ci']],'trap',round(v['trap']['mean'],2),[round(x,2) for x in v['trap']['ci']])
