"""Stage-A confirmation and dose-response for the chosen setting (fresh seeds 2000+).

Trap endpoint (adopted after the tuning sweep showed bistable outcomes; exploratory):
a phase is 'trapped' if h falls to -100 or below in it. Trap rates are compared
paired by seed with bootstrap CIs; medians and means of phase time-MAE are reported.
Confirmation gates:
  G1' CCC phase-3 trap rate exceeds WIDE's (paired CI above 0).
  G2' CDR phase-3 trap rate is below CCC's (paired CI below 0).
Dose grid: d in {0.25, 0.5, 0.75, 1} x duration in {25, 50, 100} s, starting at phase 2.
"""
import json,statistics as st,sys
from multiprocessing import Pool
from dataclasses import asdict
from world import Params,Trial,phase_dose
from gates import summarize,boot

CHOSEN=dict(q_lo=3,q_hi=15,h_seek=-3,h_avoid=8,prior_gain=0.8,fov_narrow=0.6,range_narrow=0.5,beta=1/120)

def run(args):
    p,seed,arm,d,dur=args
    dose=phase_dose(p,'DOSE',d,dur) if arm=='DOSE' else phase_dose(p,arm)
    return seed,(arm,d,dur),summarize(p,Trial(p,seed,dose).run())

def paired(a,b,f):
    d=[float(f(a[s]))-float(f(b[s])) for s in a if s in b];return dict(mean=st.mean(d),ci=boot(d),n=len(d))

def describe(rows):
    return dict(trap=[st.mean(float(r['trapped'][i]) for r in rows) for i in range(3)],
                median_mae=[st.median(r['mae'][i] for r in rows) for i in range(3)],mean_mae=[st.mean(r['mae'][i] for r in rows) for i in range(3)],
                new_patches=st.mean(r['new_patches'] for r in rows))

if __name__=='__main__':
    trials=int(sys.argv[1]) if len(sys.argv)>1 else 64;p=Params(**CHOSEN);seeds=range(2000,2000+trials)
    jobs=[(p,s,a,None,None) for s in seeds for a in ('CCC','CDR','WIDE')]
    jobs+=[(p,s,'DOSE',d,dur) for s in seeds for d in (0.25,0.5,0.75,1.) for dur in (25.,50.,100.)]
    with Pool() as pool:done=pool.map(run,jobs)
    res={}
    for seed,key,r in done:res.setdefault(key,{})[seed]=r
    ccc,cdr,wide=res[('CCC',None,None)],res[('CDR',None,None)],res[('WIDE',None,None)]
    trap3=lambda r:r['trapped'][2]
    out=dict(params=asdict(p),trials=trials,
        arms={k[0]:describe(list(v.values())) for k,v in res.items() if k[0]!='DOSE'},
        G1=paired(ccc,wide,trap3),G2=paired(cdr,ccc,trap3),
        G2_median_mae3=dict(CDR=st.median(r['mae'][2] for r in cdr.values()),CCC=st.median(r['mae'][2] for r in ccc.values())),
        G2_mae3=paired(cdr,ccc,lambda r:r['mae'][2]),
        dose={f'd={k[1]},dur={k[2]}':dict(describe(list(v.values())),trap3_vs_CCC=paired(v,ccc,trap3),mae2_vs_CCC=paired(v,ccc,lambda r:r['mae'][1]))
              for k,v in sorted(res.items(),key=lambda kv:(str(kv[0][0]),kv[0][1] or 0,kv[0][2] or 0)) if k[0]=='DOSE'})
    out['G1']['passed']=out['G1']['ci'][0]>0;out['G2']['passed']=out['G2']['ci'][1]<0
    json.dump(out,open('runs/confirm_dose.json','w'),indent=1,default=str)
    print(json.dumps({k:v for k,v in out.items() if k not in ('params','dose')},indent=1,default=str))
    for k,v in out['dose'].items():
        print(k,'trap',[round(x,2) for x in v['trap']],'median',[round(x) for x in v['median_mae']],'trap3-CCC',round(v['trap3_vs_CCC']['mean'],2),[round(x,2) for x in v['trap3_vs_CCC']['ci']],
              'mae2-CCC',round(v['mae2_vs_CCC']['mean']),'new',round(v['new_patches'],2))
