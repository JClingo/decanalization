"""Stage-A gates G0-G2 with the heuristic controller (SPEC.md). Paired trials across arms."""
import json,math,random,statistics as st,sys
from multiprocessing import Pool
from dataclasses import asdict
from world import Params,Trial,phase_dose

def run_one(args):
    p,seed,arm=args;t=Trial(p,seed,phase_dose(p,arm)).run();return seed,arm,summarize(p,t)

def window_sets(entries,a,b,w=50.):
    out=[]
    for s in range(int(a),int(b),int(w)):out.append({(x,y) for (tt,x,y) in entries if s<=tt<s+w})
    return out

def summarize(p,t):
    a,b,c=p.phases;bounds=[(0,a),(a,a+b),(a+b,a+b+c)];S=t.log['samples'];E=t.log['entries'];H=t.log['harvests']
    ph=lambda lo,hi:[s for s in S if lo<s[0]<=hi]
    mae=[st.mean(abs(s[3]) for s in ph(*bd)) for bd in bounds]
    tiles=[{(x,y) for (tt,x,y) in E if lo<tt<=hi} for lo,hi in bounds]
    p1patch={k for (tt,k,*_) in H if tt<=a}
    new=[(tt,k) for (tt,k,*_) in H if tt>a and k not in p1patch]
    trap=[min(s[3] for s in ph(*bd))<=-100 for bd in bounds]   # exploratory: captured by a starving rut
    return dict(mae=mae,trapped=trap,windows=[sorted(w) for w in window_sets(E,0,a)],tiles=[sorted(x) for x in tiles],
        kappa_late=st.mean(s[4] for s in ph(a-100,a)),kappa=[st.mean(s[4] for s in ph(*bd)) for bd in bounds],
        degraded=[st.mean(s[7] for s in ph(*bd)) for bd in bounds],harvests=[sum(1 for h in H if lo<h[0]<=hi) for lo,hi in bounds],
        patches=[sorted({h[1] for h in H if lo<h[0]<=hi}) for lo,hi in bounds],new_patches=len({k for _,k in new}),
        new_latency=(new[0][0]-a) if new else None,fov=[st.mean(s[5] for s in ph(*bd)) for bd in bounds])

def jac(a,b):a,b=set(map(tuple,a)),set(map(tuple,b));return len(a&b)/len(a|b) if a|b else float('nan')
def boot(d,n=4000,seed=1):
    r=random.Random(seed);m=sorted(st.mean(r.choice(d) for _ in d) for _ in range(n));return [m[int(.025*n)],m[int(.975*n)]]
def paired(x,y,f):
    d=[f(x[s])-f(y[s]) for s in x if s in y];return dict(mean=st.mean(d),ci=boot(d),n=len(d))

def analyze(res):
    out={}
    ccc=res['CCC'];seeds=sorted(ccc)
    own=[st.mean(jac(w1,w2) for w1,w2 in zip(ccc[s]['windows'],ccc[s]['windows'][1:])) for s in seeds]
    chance=[st.mean(jac(ccc[s]['windows'][i],ccc[o]['windows'][i+1]) for o in seeds if o!=s for i in range(len(ccc[s]['windows'])-1)) for s in seeds]
    out['G0']=dict(window_jaccard=st.mean(own),chance=st.mean(chance),ratio=st.mean(own)/st.mean(chance),kappa_late=st.mean(ccc[s]['kappa_late'] for s in seeds))
    out['G0']['passed']=out['G0']['ratio']>=2 and out['G0']['kappa_late']>=0.5
    g1=paired(res['CCC'],res['WIDE'],lambda r:r['mae'][2]);g1['degraded_CCC_minus_WIDE']=paired(res['CCC'],res['WIDE'],lambda r:r['degraded'][2])
    g1['passed']=g1['ci'][0]>0 and g1['degraded_CCC_minus_WIDE']['mean']>0;out['G1']=g1
    g2=paired(res['CDR'],res['CCC'],lambda r:r['mae'][2]);g2['new_patches_CDR_minus_CCC']=paired(res['CDR'],res['CCC'],lambda r:r['new_patches'])
    g2['p1p3_jaccard_CDR_minus_CCC']=paired(res['CDR'],res['CCC'],lambda r:jac(r['tiles'][0],r['tiles'][2]))
    g2['passed']=g2['ci'][1]<0 and g2['new_patches_CDR_minus_CCC']['mean']>0;out['G2']=g2
    out['trap']={arm:[st.mean(float(r['trapped'][i]) for r in res[arm].values()) for i in range(3)] for arm in res}
    out['median_mae3']={arm:st.median(r['mae'][2] for r in res[arm].values()) for arm in res}
    out['arms']={arm:{k:[st.mean(r[k][i] for r in res[arm].values()) for i in range(3)] for k in ('mae','kappa','degraded','harvests','fov')} for arm in res}
    return out

if __name__=='__main__':
    trials=int(sys.argv[1]) if len(sys.argv)>1 else 32;over=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
    p=Params(**over);jobs=[(p,1000+i,arm) for i in range(trials) for arm in ('CCC','CDR','WIDE')]
    with Pool() as pool:done=pool.map(run_one,jobs)
    res={}
    for seed,arm,r in done:res.setdefault(arm,{})[seed]=r
    report=dict(params=asdict(p),trials=trials,**analyze(res))
    print(json.dumps({k:v for k,v in report.items() if k!='params'},indent=1,default=lambda o:o.item() if hasattr(o,"item") else str(o)))
    out=sys.argv[3] if len(sys.argv)>3 else None
    if out:open(out,'w').write(json.dumps(report,indent=1,default=lambda o:o.item() if hasattr(o,"item") else str(o)))
