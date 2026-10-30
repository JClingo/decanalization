"""Inventory and archive the complete, validated four-condition paper update."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
v=json.loads((ROOT/'validation.json').read_text());assert v['passed'] and len(v['conditions'])==4
assert set(json.loads((ROOT/'selection.json').read_text()))=={'HRL-CDR','HRL-CCC','NF-CDR','NF-CCC'}
for name in ['README.md','RESULTS.md','MANUSCRIPT_EDITS.md','CHANGE_INVENTORY.md','primary_results.pdf','regulation_results.pdf','legacy-supplement/RESULTS.md']:assert (ROOT/name).is_file(),name
items=[dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='package_manifest.json' and '__pycache__' not in p.parts]
(ROOT/'package_manifest.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
with zipfile.ZipFile(ROOT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
with zipfile.ZipFile(ROOT.with_suffix('.zip')) as z:assert z.testzip() is None
assert ROOT.with_suffix('.zip').stat().st_size<100_000_000,'Check Git hosting file-size limit before pushing'
print('Validated package:',ROOT.with_suffix('.zip'),ROOT.with_suffix('.zip').stat().st_size,'bytes')
