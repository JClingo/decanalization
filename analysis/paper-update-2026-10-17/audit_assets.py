from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[0]/'timing'))
from graph_asset import pins
defaults={}
for path in (ROOT/'evidence').glob('BP_*.json'):
    a=json.loads(path.read_text(encoding='utf-8-sig')); result=[]
    for i,e in enumerate(a['Exports'],1):
        if e['ObjectName'].startswith('Default__'):
            defaults[path.stem]={p['Name']:p.get('Value') for p in e.get('Data',[])}
        if not e['ObjectName'].startswith('K2Node'): continue
        parent=a['Exports'][e['OuterIndex']-1]['ObjectName'] if e['OuterIndex']>0 else ''
        refs=[q['Value'] for p in e.get('Data',[]) if p['Name'] in ['FunctionReference','VariableReference'] for q in p['Value'] if q['Name']=='MemberName']
        try: ps=[(p['name'],p['default'],[x[0] for x in p['links']]) for p in pins(a,e)]
        except Exception: continue
        result.append(repr((i,parent,e['ObjectName'],refs,ps)))
    (ROOT/'evidence'/f'{path.stem}.nodes.txt').write_text('\n'.join(result),encoding='utf-8')
(ROOT/'evidence/current_defaults.json').write_text(json.dumps(defaults,indent=2))
for name,props in defaults.items():
    print(name, {k:v for k,v in props.items() if any(s in k.lower() for s in ['speed','loss','impact','plastic','duration','trial','seed','reward','setpoint','time','grid','spawn','snapshot'])})
