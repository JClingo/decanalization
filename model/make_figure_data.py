"""Regenerate the data for the paper's data-driven figures, print Table 2, and build the figure PDFs.

movement-ccc.dat, movement-cdr.dat, movement-stats.tex   Figure 2: one paired trial (positions, patches, summaries)
trajectory.dat                                          Figure 4: h every 2 s in the same trial
phases.dat                                              Figure 5: per-phase CCC vs CDR, mean and 95% t half-width
                                                        over the four training seeds, from phases_by_seed.json
levers.dat                                              Figure 6: component effects on phase-3 MAE, from
                                                        levers_learned.json

The example trial is evaluation seed 5020 of the stable-trained HRL policy (training seed 0), CCC vs CDR
(d = 1, boost 1): the trial whose per-phase mean h and new-patch harvests are closest to the averages
(see example-trial-scan.py). Table 2 values (CDR minus CCC, 95% t-interval and two-sided p over training
seeds) are printed. Inputs come from runs/ if present, otherwise from the published ../results/.
Figure sources (pgfplots) are in ../results/figures; PDFs are built in the output folder if pdflatex is available.

Usage: python make_figure_data.py [output_dir]   (default runs/figures; compare with ../results/figures)
"""
import json,math,shutil,statistics as st,subprocess,sys
from pathlib import Path
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from settings import WORLD
HERE=Path(__file__).resolve().parent;RESULTS=HERE.parent/'results'
SEED=5020;BOUNDS=(0.,300.,400.,700.);T3=3.182   # t(3): four training seeds

def find(local,published):
    p=HERE/'runs'/local;return p if p.exists() else RESULTS/published

def ci(v):
    return st.mean(v),T3*st.stdev(v)/math.sqrt(len(v))

def pval(v):
    """Two-sided one-sample t-test on per-seed differences, df = 3 (closed form)."""
    t=abs(st.mean(v)/(st.stdev(v)/math.sqrt(len(v))));r=math.sqrt(3)
    return 2*(0.5-(t/(r*(1+t*t/3))+math.atan(t/r))/math.pi)

class Recorded(OptionTrial):
    """Keeps a snapshot of patch fertility at each phase boundary."""
    def step(self,action=None):
        before=self.t;super().step(action)
        for b in BOUNDS[1:]:
            if before<b<=self.t+1e-9:self.snap[b]=[self.fertility(k) for k in range(len(self.q))]

def example(out):
    w=list(np.load(find('policy_HRL_stable.npz','policies/policy_HRL_stable.npz')).values())
    def run(arm):
        p=MapParams(**WORLD)
        dose=phase_dose(p,'DOSE',1.,100.) if arm=='CDR' else phase_dose(p,'CCC')
        t=Recorded(p,SEED,dose,'HRL',w,temp_boost=1. if arm=='CDR' else 0.);t.snap={};return t.run()
    ccc,cdr=run('CCC'),run('CDR');grid=ccc.p.grid;stats=[]
    for name,trial in (('ccc',ccc),('cdr',cdr)):
        # Positions every 0.5 s; a nan row breaks the line where the agent wraps around the torus.
        lines=['t x y h'];prev=None
        for t,x,y,h,*_ in trial.log['samples']:
            if prev and (abs(x-prev[0])>grid/2 or abs(y-prev[1])>grid/2):lines.append('nan nan nan nan')
            lines.append(f'{t:.1f} {x:.2f} {y:.2f} {h:.2f}');prev=(x,y)
        (out/f'movement-{name}.dat').write_text('\n'.join(lines)+'\n')
        # Patches as TikZ commands: familiar patches fade with fertility at the end of each phase;
        # new patches are outlined dashed before they appear at 300 s, solid afterwards.
        corners=[np.argwhere(trial.patch_of==k).min(0) for k in range(len(trial.q))]
        for phase,(lo,hi) in enumerate(zip(BOUNDS,BOUNDS[1:])):
            inside=[s for s in trial.log['samples'] if lo<round(s[0],3)<=hi];hs=[s[3] for s in inside]
            new=sum(1 for hv in trial.log['harvests'] if lo<=round(hv[0],3)<hi and hv[1] in trial.rich)
            old=sum(1 for hv in trial.log['harvests'] if lo<=round(hv[0],3)<hi and hv[1] not in trial.rich)
            print(f'{name} {lo:.0f}-{hi:.0f}s: mean|h| {np.mean(np.abs(hs)):.1f}  min h {min(hs):.1f}  harvests familiar {old}  new {new}')
            tag=name+'ABC'[phase]
            stats+=[f'\\def\\mae{tag}{{{np.mean(np.abs(hs)):.1f}}}',f'\\def\\old{tag}{{{old}}}',
                    f'\\def\\new{tag}{{{new}}}',f'\\def\\minh{tag}{{{min(hs):.0f}}}']
            draws=[]
            for k,(x0,y0) in enumerate(corners):
                box=f'(axis cs:{x0},{y0}) rectangle (axis cs:{x0+3},{y0+3});'
                if k not in trial.rich:draws.append(f'\\fill[food,fill opacity={max(0.12,trial.snap[hi][k]):.2f}] {box}')
                elif phase==0:draws.append(f'\\filldraw[fill=newp!15,draw=newp,dashed,line width=1pt] '
                                           f'(axis cs:{x0-0.6},{y0-0.6}) rectangle (axis cs:{x0+3.6},{y0+3.6});')
                else:draws.append(f'\\filldraw[fill=newp,draw=newp!70!black,line width=0.7pt] {box}')
            stats.append(f'\\def\\patches{tag}{{'+''.join(draws)+'}')
            stats+=[f'\\def\\start{tag}{{{inside[0][1]:.2f},{inside[0][2]:.2f}}}',
                    f'\\def\\stop{tag}{{{inside[-1][1]:.2f},{inside[-1][2]:.2f}}}']
    (out/'movement-stats.tex').write_text(f'% Generated by make_figure_data.py (evaluation seed {SEED})\n'+'\n'.join(stats)+'\n')
    # h every 2 s (samples are every 0.5 s and coincide across conditions).
    hc,hd=ccc.log['samples'],cdr.log['samples']
    rows=['t undisturbed decanalized']+[f'{a[0]:.1f} {a[3]:.2f} {b[3]:.2f}' for i,(a,b) in enumerate(zip(hc,hd)) if i%4==0]
    (out/'trajectory.dat').write_text('\n'.join(rows)+'\n')
    for name,trial in (('ccc',ccc),('cdr',cdr)):
        times=[hv[0] for hv in trial.log['harvests'] if hv[1] in trial.rich]
        print(name,'first new-patch harvest at',f'{min(times):.1f} s' if times else 'never')

def phases(out):
    data=json.loads(find('phases_by_seed.json','evaluations/phases_by_seed.json').read_text())['seeds']
    metrics=('mae','mean_h','trapped','rich')
    cols=['phase']+[f'{f}{m}{a}{s}' for f in ('HRL','NF') for m in metrics for a in ('ccc','cdr') for s in ('','CI')]
    lines=[' '.join(cols)]
    for k in range(3):
        row=[str(k+1)]
        for f in ('HRL','NF'):
            for m in metrics:
                for arm in ('CCC','CDR'):
                    mean,half=ci([s[f][arm][m][k] for s in data]);row+=[f'{mean:.6f}',f'{half:.6f}']
        lines.append(' '.join(row))
    (out/'phases.dat').write_text('\n'.join(lines)+'\n')
    print('Table 2 (CDR minus CCC)')
    for f in ('HRL','NF'):
        print(f)
        for m,k in (('mae',2),('trapped',2),('mean_h',2),('rich',1),('rich',2)):
            c=[s[f]['CCC'][m][k] for s in data];x=[s[f]['CDR'][m][k] for s in data];diffs=[b-a for a,b in zip(c,x)];d,h=ci(diffs)
            print(f'  {m} phase {k+1}: CCC {st.mean(c):.2f}  CDR {st.mean(x):.2f}  diff {d:.2f} [{d-h:.2f}, {d+h:.2f}]  p {pval(diffs):.4f}')
    eff={f:[s[f]['CDR']['mae'][2]-s[f]['CCC']['mae'][2] for s in data] for f in ('HRL','NF')}
    dd=[a-b for a,b in zip(eff['HRL'],eff['NF'])];d,h=ci(dd)
    print(f'HRL minus NF, mae phase 3: {d:.2f} [{d-h:.2f}, {d+h:.2f}]  p {pval(dd):.4f}')

def levers(out):
    L=json.load(open(find('levers_learned.json','evaluations/levers_learned.json')))
    order=[('both temp 3','Both (boost 3)'),('both temp 1','Both (boost 1)'),('temp-only 3.0','Entropy only (boost 3)'),
           ('temp-only 1.0','Entropy only (boost 1)'),('precision-only','Precision only')]
    lines=['y label HRLm HRLlo HRLhi NFm NFlo NFhi']
    for i,(k,lab) in enumerate(order,1):
        cells=[str(i),'{'+lab+'}']
        for fam in ('HRL','NF'):
            v=L[fam][k];m=st.mean(v);h=T3*st.stdev(v)/2;cells+=[f'{m:.2f}',f'{h:.2f}',f'{h:.2f}']
        lines.append(' '.join(cells))
    (out/'levers.dat').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    out=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'runs/figures';out.mkdir(parents=True,exist_ok=True)
    example(out);phases(out);levers(out);print('figure data ->',out)
    if shutil.which('pdflatex'):
        for name in ('movement','trajectory','phases','levers'):
            if out.resolve()!=(RESULTS/'figures').resolve():shutil.copy(RESULTS/'figures'/f'{name}.tex',out/f'{name}.tex')
            subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'{name}.tex'],cwd=out,check=True,stdout=subprocess.DEVNULL)
        print('figures built')
