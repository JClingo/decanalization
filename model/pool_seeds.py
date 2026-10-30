"""Pool the distribution-shift evaluations across training seeds (seed = unit of inference).
Per-seed effects are paired-trial means; pooled mean and a t-interval over seeds (df = n - 1)."""
import json,statistics as st,math,sys
T={2:12.706,3:4.303,4:3.182,5:2.776}
tags=['_stable','_stable_s1','_stable_s2','_stable_s3']
E=[json.load(open(f'runs/eval_policies{t}.json')) for t in tags]
def pooled(vals):
    n=len(vals);m=st.mean(vals);h=T[n]*st.stdev(vals)/math.sqrt(n) if n>1 else float('nan');return dict(mean=m,ci=[m-h,m+h],per_seed=vals,n=n)
out={}
for f in ('HRL','NF'):
    o=dict(undisturbed_trap3=pooled([e[f]['U']['trap3'] for e in E]),undisturbed_mae3=pooled([e[f]['U']['mae'][2] for e in E]),
           G1_trap=pooled([e[f]['G1']['trap']['mean'] for e in E]),G1_mae=pooled([e[f]['G1']['mae']['mean'] for e in E]),cells={})
    cells=set.intersection(*[set(e[f]['cells']) for e in E])
    for c in sorted(cells,key=lambda k:tuple(map(float,k.split(',')))):
        o['cells'][c]=dict(mae=pooled([e[f]['cells'][c]['mae_vs_U']['mean'] for e in E]),trap=pooled([e[f]['cells'][c]['trap_vs_U']['mean'] for e in E]),
                           mae2=pooled([e[f]['cells'][c]['mae2'] for e in E]),rich=pooled([e[f]['cells'][c]['rich_vs_U']['mean'] for e in E]))
    if all('10.0,1.0' in e[f]['cells'] for e in E[1:]):
        o['temp10_seeds123']=dict(mae=pooled([e[f]['cells']['10.0,1.0']['mae_vs_U']['mean'] for e in E[1:]]),mae2=pooled([e[f]['cells']['10.0,1.0']['mae2'] for e in E[1:]]),
                                  vs_temp3_mae=pooled([e[f]['cells']['10.0,1.0']['mae3']-e[f]['cells']['3.0,1.0']['mae3'] for e in E[1:]]))
    out[f]=o
out['contrast_1_1']=pooled([e['contrast']['mean'] for e in E])
json.dump(out,open('runs/pooled_seeds.json','w'),indent=1)
r=lambda d:f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] seeds {[round(x,2) for x in d['per_seed']]}"
for f in ('HRL','NF'):
    o=out[f];print(f);print('  undisturbed trap3',r(o['undisturbed_trap3']));print('  undisturbed mae3',r(o['undisturbed_mae3']))
    print('  G1 trap',r(o['G1_trap']));print('  G1 mae',r(o['G1_mae']))
    for c,v in o['cells'].items():print(f'  cell {c}: dMAE',r(v['mae']),'| dtrap',r(v['trap']),'| mae2',round(v['mae2']['mean'],1))
    if 'temp10_seeds123' in o:print('  temp10 (3 seeds): dMAE',r(o['temp10_seeds123']['mae']),'| mae2',r(o['temp10_seeds123']['mae2']),'| minus temp3',r(o['temp10_seeds123']['vs_temp3_mae']))
print('CONTRAST HRL-NF (1,1)',r(out['contrast_1_1']))
