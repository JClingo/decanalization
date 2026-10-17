"""Capture local provenance and package this review without altering source files."""
from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[PROJECT/'README.md',PROJECT/'Config/DefaultEngine.ini',PROJECT/'Content/Python/nf_training.py']
for folder in ['analysis/homeostatic','analysis/timing']:
    sources += list((PROJECT/folder).glob('*.py'))+list((PROJECT/folder).glob('*.json'))+[PROJECT/folder/'README.md']
sources += [PROJECT/'Intermediate/LearningAgents/Configs'/name for name in [
    'Training_homeostatic_ppo_SharedMemory_2026-10-14_16-11-38.json',
    'Training_nf_ppo_SharedMemory_2026-10-16_12-08-21.json']]
sources += [Path('D:/Dev/Modeling-Decanalization')/name for name in ['modelingDecanalization.tex','appendix.tex']]
records=[]
for p in sources:
    if not p.exists(): continue
    rel=p.relative_to(PROJECT) if p.is_relative_to(PROJECT) else Path('paper_baseline')/p.name
    dest=ROOT/'evidence/source_records'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    records.append(dict(source=str(p),copy=str(dest.relative_to(ROOT)),sha256=sha(p)))
for p in (ROOT/'evidence/assets').glob('*.uasset'):
    original=next(x for x in (PROJECT/'Content').rglob(p.name) if '/Character/' not in x.as_posix() and '/FirstPerson/' not in x.as_posix())
    records.append(dict(source=str(original),copy=str(p.relative_to(ROOT)),sha256=sha(p),source_unchanged=sha(original)==sha(p)))
manifest=dict(date='2026-10-17',project_commit=subprocess.check_output(['git','-c','core.fsmonitor=false','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),sources=records,
    supplementary_rate_mapping={'HRL-CDR-4-plasticity.yml':4,'HRL-CDR-6-plasticity.yml':6,'HRL-CDR-7-plasticity.yml':7},mapping_source='Explicit user clarification in this chat on 2026-10-17')
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
items=[dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='package_manifest.json' and '__pycache__' not in p.parts]
(ROOT/'package_manifest.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
target=ROOT.with_suffix('.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
with zipfile.ZipFile(target) as z: assert z.testzip() is None
for item in items: assert sha(ROOT/item['file'])==item['sha256']
assert all(r.get('source_unchanged',True) for r in records)
print(f'Validated {len(items)} package files; ZIP {target} ({target.stat().st_size:,} bytes).')
