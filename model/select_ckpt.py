"""Pick each family's checkpoint on held-out validation trials (seeds 20000+, 8 trials, no intervention),
by its own objective: HRL minimum mean |h|, NF minimum mean deficit max(-h, 0). Candidates: every 5th
iteration from 20, plus the last. Writes runs/policy_{family}.npz and runs/selection_{family}.json.
"""
import glob,json,shutil,statistics as st,sys
from multiprocessing import Pool
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from settings import WORLD
def score(args):
    family,path,seed,world=args;w=list(np.load(path).values());p=MapParams(**world)
    t=OptionTrial(p,seed,phase_dose(p,'CCC'),family,w).run();H=[s[3] for s in t.log['samples']]
    return path,st.mean(abs(h) for h in H) if family=='HRL' else st.mean(max(-h,0) for h in H),st.mean(abs(h) for h in H)
if __name__=='__main__':
    family=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else '';world=dict(WORLD,**(json.loads(sys.argv[3]) if len(sys.argv)>3 else {}))
    paths=sorted(glob.glob(f'runs/ckpt_{family}{tag}/it*.npz'))
    cand=[x for x in paths if int(x[-7:-4])>=20 and int(x[-7:-4])%5==0]+[paths[-1]]
    with Pool() as pool:res=pool.map(score,[(family,c,20000+i,world) for c in dict.fromkeys(cand) for i in range(8)])
    by={}
    for path,obj,mae in res:by.setdefault(path,[]).append((obj,mae))
    table={k:dict(objective=st.mean(o for o,_ in v),mae=st.mean(m for _,m in v)) for k,v in by.items()}
    best=min(table,key=lambda k:table[k]['objective']);shutil.copy(best,f'runs/policy_{family}{tag}.npz')
    json.dump(dict(rule=__doc__,table=table,selected=best),open(f'runs/selection_{family}{tag}.json','w'),indent=1)
    for k,v in table.items():print(k,round(v['objective'],2),round(v['mae'],2))
    print('SELECTED',best)
