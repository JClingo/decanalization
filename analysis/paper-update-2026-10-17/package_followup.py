"""Package authorized fixes while retaining the original historical evidence."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1];OUT=ROOT/'rerun-update'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# All copied historical evidence remains byte-identical to the original audit.
for item in json.loads((ROOT/'source_manifest.json').read_text())['sources']:
    assert sha(ROOT/item['copy'])==item['sha256'],item['copy']
for p in (PROJECT/'analysis/trial_metrics').glob('*'):
    if p.is_file():shutil.copy2(p,OUT/p.name)
for n in ['trial_metrics.py','trial_notifications.py','trial_training_manifests.json']:
    shutil.copy2(PROJECT/'Content/Python'/n,OUT/n)
shutil.copy2(PROJECT/'analysis/trial_metrics/README.md',OUT/'IMPLEMENTATION.md')
for name in ['BFL_Common.uasset','BP_AgentTrainingEnv.uasset','BP_GameMode.uasset']:
    dest=OUT/'validated-assets'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(PROJECT/'Saved/TrialMetricsFix/compiled'/name,dest)
for test in json.loads((PROJECT/'analysis/trial_metrics/runtime_validation.json').read_text()):
    p=PROJECT/test['file'];dest=OUT/'smoke-tests';dest.mkdir(exist_ok=True)
    for source in [p,p.with_suffix('').with_suffix('.yml'),p.with_suffix('').with_suffix('.completed.json')]:
        shutil.copy2(source,dest/source.name)
    source=p.with_suffix('').with_suffix('.provenance');shutil.copytree(source,dest/source.name,dirs_exist_ok=True)
manifest=[dict(file=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='package_manifest.json' and '__pycache__' not in p.parts]
(ROOT/'package_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(ROOT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
with zipfile.ZipFile(ROOT.with_suffix('.zip')) as z:assert z.testzip() is None
print(f'Packaged {len(manifest)} files; historical evidence preserved.')
