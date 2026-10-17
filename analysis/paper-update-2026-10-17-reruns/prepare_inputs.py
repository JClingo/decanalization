"""Freeze complete selected reruns and retain manuscript/legacy evidence."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,d):
    d.parent.mkdir(parents=True,exist_ok=True)
    if d.exists():assert sha(p)==sha(d),'Frozen input changed: '+str(d)
    else:shutil.copy2(p,d)
for run in json.loads((ROOT/'selection.json').read_text()).values():
    for suffix in ['.yml','.metrics.json','.completed.json']:
        p=PROJECT/'analysis/data'/(run+suffix);copy(p,ROOT/'inputs'/p.name)
    p=PROJECT/'analysis/data'/(run+'.provenance')
    for f in p.iterdir():copy(f,ROOT/'inputs'/p.name/f.name)
for name in ['modelingDecanalization.tex','appendix.tex']:
    copy(Path('D:/Dev/Modeling-Decanalization')/name,ROOT/'paper-baseline'/name)
old=PROJECT/'analysis/paper-update-2026-10-17'
for name in ['HRL-CDR-10000.yml','HRL-CDR-4-plasticity.yml','HRL-CDR-6-plasticity.yml','HRL-CDR-7-plasticity.yml']:
    copy(old/'inputs'/name,ROOT/'legacy-supplement/inputs'/name)
for name in ['CHANGE_INVENTORY.md','MANUSCRIPT_EDITS.md']:
    copy(old/name,ROOT/'historical-audit'/name)
copy(PROJECT/'Content/Python/trial_training_manifests.json',ROOT/'evidence/trial_training_manifests.json')
for name in ['trial_metrics.py','trial_notifications.py']:
    copy(PROJECT/'Content/Python'/name,ROOT/'evidence'/name)
print('Frozen selected inputs, paper baseline, and separately labeled legacy supplement.')
