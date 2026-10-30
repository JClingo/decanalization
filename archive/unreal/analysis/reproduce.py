"""Verify frozen evidence and regenerate paper outputs without editing the archive."""
from pathlib import Path
import argparse, csv, hashlib, importlib.metadata, json, math, os, shutil, subprocess, sys
REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / 'analysis/paper-update-2026-10-17-reruns'
TABLES = ['trial_metrics.csv', 'phase_summary.csv', 'primary_contrasts.csv', 'paired_sensitivity.csv',
          'legacy-supplement/summary.csv', 'legacy-supplement/contrasts.csv']

def verify_manifest():
    records = json.loads((PACKAGE / 'package_manifest.json').read_text())
    for r in records:
        rel = Path(r['file'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe manifest path: ' + str(rel))
        path = PACKAGE / rel
        if not path.is_file() or path.stat().st_size != r['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Frozen package mismatch: ' + str(rel))
    print(f'Verified {len(records)} frozen package files.', flush=True)

def compare_csv(a, b):
    with a.open(newline='', encoding='utf-8') as f: expected = list(csv.reader(f))
    with b.open(newline='', encoding='utf-8') as f: actual = list(csv.reader(f))
    if len(expected) != len(actual): raise ValueError('Row count mismatch: ' + str(b))
    for i, (x, y) in enumerate(zip(expected, actual)):
        if len(x) != len(y): raise ValueError('Column count mismatch: ' + str(b))
        for old, new in zip(x, y):
            if old == new: continue
            try: same = math.isclose(float(old), float(new), rel_tol=2e-7, abs_tol=1e-10)
            except ValueError: same = False
            if not same: raise ValueError(f'Output mismatch: {b}, row {i}: {old!r} vs {new!r}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='stdlib only: verify raw evidence and all primary statistics')
    parser.add_argument('--paper-figures', action='store_true', help='also run base R to recreate the manuscript PNGs')
    parser.add_argument('--output', type=Path, default=REPO/'analysis/generated/paper', help='new output directory; must not exist')
    args = parser.parse_args()
    if not __debug__: parser.error('Run without -O: the scientific validators use assertions.')
    verify_manifest()
    subprocess.run([sys.executable, str(REPO/'analysis/verify_paper.py'), str(PACKAGE)], check=True)
    if args.verify_only:
        if args.paper_figures: parser.error('--paper-figures requires a full run')
        return
    versions = {}
    for name in ['numpy','scipy','PyYAML','matplotlib']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            parser.error('Missing ' + name + '; install analysis/requirements-paper.txt first')
    rscript = shutil.which('Rscript') if args.paper_figures else None
    if args.paper_figures and not rscript: parser.error('Rscript is required for --paper-figures')
    out = args.output.resolve()
    # Never copy recursively into, or overwrite, the frozen evidence.
    if out == PACKAGE or PACKAGE in out.parents or out in PACKAGE.parents:
        parser.error('Output must be separate from the frozen package')
    if out.exists(): parser.error('Output already exists; choose a new --output directory')
    shutil.copytree(PACKAGE, out)
    # Original sidecars use Windows separators. Normalize only code in the output
    # copy so raw evidence and its manifest remain byte-identical on every OS.
    path_expressions = ["base/s['loaded-snapshot']", "base/phase['scene-file']", "ROOT/'inputs'/z['scene-file']"]
    script = out/'build_analysis.py'
    source = script.read_text(encoding='utf-8')
    for expression in path_expressions:
        source = source.replace(expression, expression + ".replace(chr(92), '/')")
    # Do not silently import an old Saved/AnalysisPython installation.
    for name in ['build_analysis.py','build_supplement.py']:
        p = out/name
        text = source if name == 'build_analysis.py' else p.read_text(encoding='utf-8')
        text = '\n'.join(line for line in text.splitlines() if not line.startswith('sys.path.insert(')) + '\n'
        p.write_text(text, encoding='utf-8')
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(out/'.matplotlib')
    for name in ['build_analysis.py','build_supplement.py','write_report.py']:
        subprocess.run([sys.executable, str(out/name)], cwd=out, env=env, check=True)
    for name in TABLES: compare_csv(PACKAGE/name, out/name)
    subprocess.run([sys.executable, str(REPO/'analysis/verify_paper.py'), str(out)], check=True)
    if rscript:
        subprocess.run([rscript,'--vanilla',str(REPO/'analysis/plot_paper.R'),str(out)], check=True)
    # The copied manifest covers the original archive, not regenerated outputs.
    (out/'package_manifest.json').unlink()
    (out/'reproduction.json').write_text(json.dumps(dict(passed=True, python=sys.version,
        dependencies=versions, source=str(PACKAGE), tables_compared=TABLES,
        numeric_relative_tolerance=2e-7, numeric_absolute_tolerance=1e-10,
        paper_figures=bool(rscript)), indent=2), encoding='utf-8')
    print('Reproduced outputs: ' + str(out))
if __name__ == '__main__': main()
