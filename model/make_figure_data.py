"""Regenerate the paper's figure data (results/figures/*.dat) and, if pdflatex is available, the figure PDFs.

trajectory.dat  the first evaluation trial (seeds 5000+) in which the undisturbed stable-trained HRL policy
                (training seed 0) is trapped in phase 3 and the decanalized one (boost 1, d = 1) is not;
                h every 0.5 s, every 4th sample (every 2 s)
levers.dat      lever effects on phase-3 MAE (mean and 95% t half-width over the four training seeds), from
                levers_learned.json
Inputs come from runs/ if present, otherwise from the published ../results/.
Usage: python make_figure_data.py [output_dir]   (default runs/figures; compare with ../results/figures)
"""
import json,shutil,statistics as st,subprocess,sys
from pathlib import Path
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from settings import WORLD
HERE=Path(__file__).resolve().parent;RESULTS=HERE.parent/'results'
def find(local,published):
    p=HERE/'runs'/local;return p if p.exists() else RESULTS/published
out=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'runs/figures';out.mkdir(parents=True,exist_ok=True)
# Figure 1: one paired trial
w=list(np.load(find('policy_HRL_stable.npz','policies/policy_HRL_stable.npz')).values())
for s in range(5000,5064):
    p=MapParams(**WORLD)
    u=OptionTrial(p,s,phase_dose(p,'CCC'),'HRL',w).run();d=OptionTrial(p,s,phase_dose(p,'DOSE',1.,100.),'HRL',w,temp_boost=1.).run()
    hu=[x[3] for x in u.log['samples']];hd=[x[3] for x in d.log['samples']]
    if min(hu[800:])<=-100 and min(hd[800:])>-100:break
else:raise SystemExit('no example trial found')
rows=list(zip([x[0] for x in u.log['samples']],hu,hd))[::4]
(out/'trajectory.dat').write_text('t undisturbed decanalized\n'+''.join(f'{round(t,2)} {a:.2f} {b:.2f}\n' for t,a,b in rows))
# Figure 2: lever effects
L=json.load(open(find('levers_learned.json','evaluations/levers_learned.json')));T3=3.182
order=[('both temp 3','Both (boost 3)'),('both temp 1','Both (boost 1)'),('temp-only 3.0','Entropy only (boost 3)'),
       ('temp-only 1.0','Entropy only (boost 1)'),('precision-only','Precision only')]
lines=['y label HRLm HRLlo HRLhi NFm NFlo NFhi']
for i,(k,lab) in enumerate(order,1):
    cells=[str(i),'{'+lab+'}']
    for fam in ('HRL','NF'):
        v=L[fam][k];m=st.mean(v);h=T3*st.stdev(v)/2;cells+=[f'{m:.2f}',f'{h:.2f}',f'{h:.2f}']
    lines.append(' '.join(cells))
(out/'levers.dat').write_text('\n'.join(lines)+'\n')
print('example trial seed',s,'->',out)
# Figure PDFs from the pgfplots sources
if shutil.which('pdflatex'):
    for name in ('trajectory','levers'):
        shutil.copy(RESULTS/'figures'/f'{name}.tex',out/f'{name}.tex')
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'{name}.tex'],cwd=out,check=True,stdout=subprocess.DEVNULL)
    print('figures built')
