"""Evaluate the trained HRL and NF option policies (fresh seeds 5000+, 64 paired trials, sampled actions).
World: reliable-opportunity degrading world (as in training). Arms per family:
  U        undisturbed; LOWPREC same policy, beliefs track the senses (P_reward 0, tau_P 20)
  cells    temp_boost {0, 1, 3} x dose {0.5, 1}, 100 s window, precision lever on
Criteria fixed before running, per family:
  G1       MAP - LOWPREC phase-3 trap rate CI lower > 0 (or MAE CI lower > 0)
  Benefit  some cell beats U on phase-3 MAE (paired CI upper < 0)
  Harm     (boost 3, d 1) worse than U on phase-3 MAE (paired CI lower > 0)
Contrast   HRL vs NF at (boost 1, d 1): [HRL cell - HRL U] - [NF cell - NF U] on phase-3 MAE, paired by
           seed; the paper's hypothesis predicts < 0 (HRL benefits more). Reported with CI.
"""
import json,statistics as st,sys
from multiprocessing import Pool
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from gates import summarize,boot
from confirm_dose import paired
from settings import WORLD
CELLS=[(b,d) for b in (0.,1.,3.) for d in (0.5,1.)]+[(10.,1.)]   # (10, 1) added for the seed replication

def run(args):
    family,seed,arm,b,d,tag=args;w=list(np.load(f'runs/policy_{family}{tag}.npz').values())
    p=MapParams(**WORLD,**(dict(P_reward=0.,tau_P=20.) if arm=='LOWPREC' else {}))
    dose=phase_dose(p,'DOSE',d,100.) if arm=='CELL' else phase_dose(p,'CCC')
    t=OptionTrial(p,seed,dose,family,w,temp_boost=b).run();r=summarize(p,t)
    r['rich23']=sum(1 for h in t.log['harvests'] if h[1] in t.rich and h[0]>p.phases[0])
    opts=[o for tt,o in t.log['options'] if p.phases[0]<=tt<p.phases[0]+p.phases[1]];r['explore_share2']=float(np.mean([o==1 for o in opts])) if opts else None
    return family,seed,(arm,b,d),r

if __name__=='__main__':
    tag=sys.argv[1] if len(sys.argv)>1 else '';seeds=range(5000,5064);jobs=[]
    for f in ('HRL','NF'):
        jobs+=[(f,s,'U',0.,None,tag) for s in seeds]+[(f,s,'LOWPREC',0.,None,tag) for s in seeds]+[(f,s,'CELL',b,d,tag) for s in seeds for b,d in CELLS]
    with Pool() as pool:done=pool.map(run,jobs)
    R={}
    for f,s,k,r in done:R.setdefault(f,{}).setdefault(k,{})[s]=r
    mae3=lambda r:r['mae'][2];trap3=lambda r:float(r['trapped'][2]);rich=lambda r:r['rich23']
    out={}
    for f in ('HRL','NF'):
        U=R[f][('U',0.,None)];L=R[f][('LOWPREC',0.,None)]
        o=dict(U=dict(mae=[st.mean(r['mae'][i] for r in U.values()) for i in range(3)],median3=st.median(mae3(r) for r in U.values()),
                      trap3=st.mean(trap3(r) for r in U.values()),mean_h3=None),
               G1=dict(trap=paired(U,L,trap3),mae=paired(U,L,mae3)),cells={})
        for b,d in CELLS:
            X=R[f][('CELL',b,d)]
            o['cells'][f'{b},{d}']=dict(mae3=st.mean(mae3(r) for r in X.values()),median3=st.median(mae3(r) for r in X.values()),mae_vs_U=paired(X,U,mae3),
                trap_vs_U=paired(X,U,trap3),rich_vs_U=paired(X,U,rich),explore2=st.mean(r['explore_share2'] for r in X.values() if r['explore_share2'] is not None),
                mae2=st.mean(r['mae'][1] for r in X.values()))
        o['criteria']=dict(G1=o['G1']['trap']['ci'][0]>0 or o['G1']['mae']['ci'][0]>0,
            benefit=any(v['mae_vs_U']['ci'][1]<0 for v in o['cells'].values()),harm=o['cells']['3.0,1.0']['mae_vs_U']['ci'][0]>0)
        out[f]=o
    dH={s:mae3(R['HRL'][('CELL',1.,1.)][s])-mae3(R['HRL'][('U',0.,None)][s]) for s in seeds}
    dN={s:mae3(R['NF'][('CELL',1.,1.)][s])-mae3(R['NF'][('U',0.,None)][s]) for s in seeds}
    diff=[dH[s]-dN[s] for s in seeds];out['contrast']=dict(mean=st.mean(diff),ci=boot(diff),n=len(diff))
    json.dump(out,open(f'runs/eval_policies{tag}.json','w'),indent=1,default=str)
    for f in ('HRL','NF'):
        o=out[f];print(f,'U mae',[round(x,1) for x in o['U']['mae']],'median3',round(o['U']['median3'],1),'trap3',round(o['U']['trap3'],2))
        print('  G1 trap',round(o['G1']['trap']['mean'],2),[round(x,2) for x in o['G1']['trap']['ci']],'mae',round(o['G1']['mae']['mean'],1),[round(x,1) for x in o['G1']['mae']['ci']])
        for k,v in o['cells'].items():print(f"  cell {k}: mae3 {v['mae3']:.1f} (med {v['median3']:.1f}) dMAE {v['mae_vs_U']['mean']:+.1f} {[round(x,1) for x in v['mae_vs_U']['ci']]} dtrap {v['trap_vs_U']['mean']:+.2f} drich {v['rich_vs_U']['mean']:+.1f} explore2 {v['explore2']:.2f} mae2 {v['mae2']:.1f}")
        print('  CRITERIA',o['criteria'])
    print('CONTRAST HRL-NF at (1,1):',round(out['contrast']['mean'],1),[round(x,1) for x in out['contrast']['ci']])
