"""Confirm the best map-agent decanalization setting on fresh seeds (3000+, 64 paired trials).

Selection rule, fixed before the reliable-world lever results were seen: from the lever-test
JSON, among (directed, boost, dose) cells that beat the undisturbed agent (paired CI upper < 0
on phase-3 MAE or trap rate), take the most negative mean MAE difference (ties: lower dose).
If none beats it, take the most negative mean MAE difference.
Confirmation criterion: the chosen cell beats the undisturbed agent on phase-3 MAE
(paired CI upper < 0); trap rate, rich-patch harvests and the dose curve at the chosen
boost are reported.
"""
import json,statistics as st,sys
from multiprocessing import Pool
from map_levers2 import run,DOSES
from confirm_dose import paired

def choose(levers):
    cells=[]
    for de,v in levers.items():
        for b,w in v['boosts'].items():
            for d,x in w['doses'].items():
                beats=x['mae_vs_U']['ci'][1]<0 or x['trap_vs_U']['ci'][1]<0
                cells.append((not beats,x['mae_vs_U']['mean'],float(d),float(de),float(b)))
    cells.sort();_,m,d,de,b=cells[0];return dict(directed=de,boost=b,dose=d,lever_mae_diff=m,beat_in_lever_test=not cells[0][0])

if __name__=='__main__':
    levers=json.load(open(sys.argv[1]));over=json.loads(sys.argv[2]);c=choose(levers);print('chosen',c)
    seeds=range(3000,3064);de,b=c['directed'],c['boost']
    jobs=[(s,de,0.,None,'MAP',over) for s in seeds]+[(s,de,b,d,'MAP',over) for s in seeds for d in DOSES]
    with Pool() as pool:done=pool.map(run,jobs)
    R={}
    for s,k,r in done:R.setdefault(k,{})[s]=r
    U=R[(de,0.,None,'MAP')];mae3=lambda r:r['mae'][2];trap3=lambda r:float(r['trapped'][2]);rich=lambda r:r['rich23']
    out=dict(chosen=c,over=over,undisturbed=dict(mae3=st.mean(mae3(r) for r in U.values()),median_mae3=st.median(mae3(r) for r in U.values()),
             trap3=st.mean(trap3(r) for r in U.values()),rich23=st.mean(rich(r) for r in U.values())),doses={})
    for d in DOSES:
        X=R[(de,b,d,'MAP')]
        out['doses'][d]=dict(mae3=st.mean(mae3(r) for r in X.values()),median_mae3=st.median(mae3(r) for r in X.values()),
            mae_vs_U=paired(X,U,mae3),trap_vs_U=paired(X,U,trap3),rich_vs_U=paired(X,U,rich),mae2=st.mean(r['mae'][1] for r in X.values()))
    out['confirmed']=out['doses'][c['dose']]['mae_vs_U']['ci'][1]<0
    json.dump(out,open('runs/confirm_map.json','w'),indent=1,default=str)
    print('undisturbed',{k:round(v,2) for k,v in out['undisturbed'].items()})
    for d,v in out['doses'].items():
        print(f"d={d} mae3 {v['mae3']:.1f} (median {v['median_mae3']:.1f}) | dMAE {v['mae_vs_U']['mean']:.1f} {[round(x,1) for x in v['mae_vs_U']['ci']]} | dtrap {v['trap_vs_U']['mean']:.2f} {[round(x,2) for x in v['trap_vs_U']['ci']]} | drich {v['rich_vs_U']['mean']:.1f} {[round(x,1) for x in v['rich_vs_U']['ci']]} | mae2 {v['mae2']:.1f}")
    print('CONFIRMED',out['confirmed'])
