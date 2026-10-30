"""Stage-A tuning sweep: find a survivable hypercanalization setting.

Criterion, fixed before running: among configs passing G0, G1 and G2 whose CCC phase-3
time-MAE is at most 2.5x WIDE's (costly but survivable), choose the one with the lowest
CCC phase-3 MAE, i.e. the mildest mechanism that still shows the dynamics.
"""
import itertools,json,sys
from multiprocessing import Pool
from dataclasses import asdict
from world import Params
from gates import run_one,analyze

BASE=dict(q_lo=3,q_hi=15,h_seek=-3,h_avoid=8)
GRID=dict(prior_gain=[0.6,0.8,1.0],narrow=[(0.75,0.7),(0.6,0.5)],beta=[1/120,1/60])

def evaluate(cfg,trials,pool):
    p=Params(**cfg);done=pool.map(run_one,[(p,1000+i,arm) for i in range(trials) for arm in ('CCC','CDR','WIDE')])
    res={}
    for seed,arm,r in done:res.setdefault(arm,{})[seed]=r
    return analyze(res)

if __name__=='__main__':
    trials=int(sys.argv[1]) if len(sys.argv)>1 else 32;rows=[]
    with Pool() as pool:
        for pg,(fn,rn),beta in itertools.product(GRID['prior_gain'],GRID['narrow'],GRID['beta']):
            cfg=dict(BASE,prior_gain=pg,fov_narrow=fn,range_narrow=rn,beta=beta);a=evaluate(cfg,trials,pool)
            mae={arm:a['arms'][arm]['mae'][2] for arm in a['arms']}
            row=dict(cfg=cfg,G0=a['G0']['passed'],G0_ratio=a['G0']['ratio'],G1=a['G1']['passed'],G1_diff=a['G1']['mean'],G1_ci=a['G1']['ci'],
                     G2=a['G2']['passed'],G2_diff=a['G2']['mean'],G2_ci=a['G2']['ci'],new_patches=a['G2']['new_patches_CDR_minus_CCC']['mean'],mae3=mae,
                     survivable=mae['CCC']<=2.5*mae['WIDE'],trap=a['trap'],median_mae3=a['median_mae3'])
            rows.append(row);print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items()},default=str),flush=True)
    ok=[r for r in rows if r['G0'] and r['G1'] and r['G2'] and r['survivable']]
    choice=min(ok,key=lambda r:r['mae3']['CCC']) if ok else None
    json.dump(dict(trials=trials,criterion=__doc__,rows=rows,choice=choice),open('runs/sweep.json','w'),indent=1,default=str)
    print('CHOICE',json.dumps(choice,default=str))
