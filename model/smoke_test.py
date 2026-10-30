"""Smoke test: short trials (30 s / 10 s / 30 s, rich patches at 30 s) with each saved stable-trained policy,
undisturbed and decanalized. Checks that the world, map and policy run and produce finite states."""
import glob,math
from pathlib import Path
import numpy as np
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial
from settings import WORLD
paths=sorted(glob.glob(str(Path(__file__).resolve().parent.parent/'results/policies/policy_*_stable*.npz')))
assert paths,'no saved policies found in results/policies'
for path in paths:
    family='HRL' if 'HRL' in Path(path).name else 'NF';w=list(np.load(path).values())
    p=MapParams(**dict(WORLD,phases=(30.,10.,30.),rich_appear=30.))
    for arm,dose,boost in (('CCC',phase_dose(p,'CCC'),0.),('CDR',phase_dose(p,'DOSE',1.,10.),1.)):
        t=OptionTrial(p,1,dose,family,w,temp_boost=boost).run()
        assert all(math.isfinite(s[3]) for s in t.log['samples']) and t.log['options'],(path,arm)
    print('ok',Path(path).name)
print('SMOKE TEST PASSED',len(paths),'policies')
