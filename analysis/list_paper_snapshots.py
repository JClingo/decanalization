"""Print verified absolute snapshot paths to paste into Unreal manager settings."""
import argparse, hashlib, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['HRL', 'NF'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent/'paper-update-2026-10-17-reruns'
    selection = json.loads((root/'selection.json').read_text(encoding='utf-8'))
    # CDR and CCC load the same selected policy triplet for each family.
    run = selection[args.family+'-CDR']
    metadata = json.loads((root/'inputs'/(run+'.metrics.json')).read_text(encoding='utf-8'))
    paths = {}
    for kind, field in [('encoder','EncoderSnapshot'),('policy','PolicySnapshot'),('decoder','Decoder Snapshot')]:
        item = metadata['provenance']['snapshots'][kind]
        path = root/'inputs'/item['loaded-snapshot'].replace('\\', '/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['loaded-sha256']:
            raise ValueError('Snapshot hash mismatch: ' + str(path))
        paths[field] = str(path.resolve())
    print(json.dumps(dict(family=args.family, selected_run=run, manager_settings=paths), indent=2))
if __name__ == '__main__': main()
