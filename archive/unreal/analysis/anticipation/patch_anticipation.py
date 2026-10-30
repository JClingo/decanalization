"""Repair anticipation graph wiring; compile with Unreal before installing.

Inputs are fresh UAssetAPI exports. Never edit compiled bytecode. Tile path
lengths intersect the trace with each hit component's world AABB (the project
uses axis-aligned cube tiles). Passive loss is counted once across the horizon.
"""
import argparse,base64,copy,json,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('workdir',type=Path,help='Directory containing original BP_Agent.json and BP_GameMode.json UAssetAPI exports')
parser.add_argument('--helpers',type=Path,default=Path(__file__).resolve().parents[1]/'timing')
args=parser.parse_args()
sys.path.insert(0,str(args.helpers))
from graph_asset import pins
from patch_graphs import Editor,prop,string
ROOT=args.workdir.resolve()
ed=Editor(json.loads((ROOT/'BP_Agent.json').read_text()))
assert len(ed.a['Exports'])==465 and not any(e['ObjectName'].startswith('K2Node_Anticipation_') for e in ed.a['Exports']), 'This migration requires the audited pre-fix asset; do not apply twice.'
def named(name,graph):
 return next(i for i,e in enumerate(ed.a['Exports'],1) if e['ObjectName']==name and e['OuterIndex']>0 and ed.e(e['OuterIndex'])['ObjectName']==graph)
G='GetAnticipatedTraversalCosts';Y='GetAnticipatedResourceYield'
def old(name,g=G):return named(name,g)
math=next(-i for i,e in enumerate(ed.a['Imports'],1) if e['ObjectName']=='KismetMathLibrary')
system=next(-i for i,e in enumerate(ed.a['Imports'],1) if e['ObjectName']=='KismetSystemLibrary')
vector=next(-i for i,e in enumerate(ed.a['Imports'],1) if e['ObjectName']=='Vector')
template=old('K2Node_CallFunction_5')
def inp(n,t='real',default='',obj=0,sub='double'):return (n,0,t,obj,default,sub if t=='real' else 'None')
def out(n='ReturnValue',t='real',obj=0,sub='double'):return (n,1,t,obj,'',sub if t=='real' else 'None')
def node(label,function,specs,parent=math):
 ps=pins(ed.a,ed.e(template));tail=base64.b64decode(ed.e(template)['Extras'])[ps[-1]['end']:]
 i=ed.node(template,'K2Node_Anticipation_'+label,function,specs,2200+((len(ed.changed)//12)%5)*440,300+(len(ed.changed)%12)*160)
 ed.e(i)['Extras']=base64.b64encode(base64.b64decode(ed.e(i)['Extras'])+tail).decode()
 refs=prop(ed.e(i),'FunctionReference')['Value']
 for p in refs:
  if p['Name']=='MemberParent':p['Value']=parent
 for p in list(pins(ed.a,ed.e(i)))[::-1]:
  if p['auto_default']:ed.replace(i,p['default_end'],p['default_end']+len(string(p['auto_default'])),string(''))
 for field in ['NodeComment','bCommentBubbleVisible','bCommentBubbleVisible_InDetailsPanel']:
  ed.e(i)['Data'][:]=[p for p in ed.e(i)['Data'] if p['Name']!=field]
 return i
# Sources are (node index,pin name); literals are decimal strings.
def wire(value,dest,pin):
 if isinstance(value,tuple):ed.connect(*value,dest,pin)
 else:
  p=ed.p(dest,pin);ed.replace(dest,p['default_start'],p['default_end'],string(str(value)))
def operation(label,function,a,b=None,result='real'):
 specs=[inp('A')]+([inp('B')] if b is not None else [])+[out(t=result)]
 i=node(label,function,specs);wire(a,i,'A')
 if b is not None:wire(b,i,'B')
 return (i,'ReturnValue')
def select(label,a,b,condition):
 i=node(label,'SelectFloat',[inp('A'),inp('B'),inp('bPickA','bool'),out()]);wire(a,i,'A');wire(b,i,'B');wire(condition,i,'bPickA');return(i,'ReturnValue')
def getter(label,source,pinname,category='real',sub='double',reference=None):
 ps=pins(ed.a,ed.e(source));tail=base64.b64decode(ed.e(source)['Extras'])[ps[-1]['end']:]
 i=ed.node(source,'K2Node_Anticipation_'+label,'',[out(pinname,category,sub=sub)]+[inp('self','object',obj=next((q['Value'] for q in (reference or prop(ed.e(source),'VariableReference')['Value']) if q['Name']=='MemberParent'),0))],2600,0)
 ed.e(i)['Extras']=base64.b64encode(base64.b64decode(ed.e(i)['Extras'])+tail).decode()
 if reference is not None:prop(ed.e(i),'VariableReference')['Value']=copy.deepcopy(reference)
 target=ed.e(template)['OuterIndex']
 if ed.e(i)['OuterIndex']!=target:
  ns=prop(ed.e(ed.e(i)['OuterIndex']),'Nodes')['Value'];item=next(p for p in ns if p['Value']==i);ns.remove(item)
  ed.e(i)['OuterIndex']=target;ns=prop(ed.e(target),'Nodes')['Value'];item['Name']=str(len(ns));ns.append(item)
 return i
# Sum pellet impacts. Do not reactivate the old per-pellet passive-loss term:
# it charges the same shared journey repeatedly.
add=old('K2Node_PromotableOperator_4',Y);setter=old('K2Node_VariableSet_0',Y)
ed.disconnect(add,'B');ed.rewire(setter,'Total',add,'ReturnValue')
# Leave the now-disconnected travel term visible as historical graph evidence.
entry=old('K2Node_FunctionEntry_0');loop=old('K2Node_MacroInstance_0');cast=old('K2Node_DynamicCast_1')
reset=old('K2Node_VariableSet_0');accumulate=old('K2Node_VariableSet_4');result=old('K2Node_FunctionResult_0');zero=old('K2Node_FunctionResult_1');gate=old('K2Node_IfThenElse_1')
# Explicit per-call initialization; the loop now has one tile-cost path.
ed.rewire(entry,'then',gate,'execute')
ed.rewire(gate,'else',zero,'execute');ed.rewire(gate,'then',reset,'execute');ed.rewire(reset,'then',loop,'Exec')
ed.rewire(cast,'then',accumulate,'execute')
# Clear old branch's inputs at the new roots; unused approximation remains dead.
# Math inputs are connected directly to function parameters.
speed=(entry,'MaxSpeed');dt=(entry,'DeltaT');passive=(entry,'PassiveLossRate')
validspeed=operation('PositiveSpeed','Greater_DoubleDouble',speed,'0.000001','bool')
validdt=operation('PositiveDelta','Greater_DoubleDouble',dt,'0.000001','bool')
both=node('ValidMotion','BooleanAND',[inp('A','bool'),inp('B','bool'),out(t='bool')]);wire(validspeed,both,'A');wire(validdt,both,'B')
ed.rewire(gate,'Condition',both,'ReturnValue')
# Bounds and trace geometry. Exit time uses all three slabs; a parallel axis
# contributes 1 because the actual hit establishes that the line is inside it.
bounds=node('TileBounds','GetComponentBounds',[inp('Component','object',obj=ed.import_class('SceneComponent')),out('Origin','struct',vector),out('BoxExtent','struct',vector),out('SphereRadius','real',sub='float')],system)
ed.connect(loop,'Array Element_HitComponent',bounds,'Component')
direction=node('TraceDirection','Subtract_VectorVector',[inp('A','struct',obj=vector),inp('B','struct',obj=vector),out(t='struct',obj=vector)])
ed.connect(loop,'Array Element_TraceEnd',direction,'A');ed.connect(loop,'Array Element_TraceStart',direction,'B')
raylen=node('TraceLength','VSize',[inp('A','struct',obj=vector),out()]);ed.connect(direction,'ReturnValue',raylen,'A')
def split(label,source):
 i=node(label,'BreakVector',[inp('InVec','struct',obj=vector),out('X'),out('Y'),out('Z')]);wire(source,i,'InVec');return i
origin=split('OriginXYZ',(bounds,'Origin'));extent=split('ExtentXYZ',(bounds,'BoxExtent'));start=split('StartXYZ',(loop,'Array Element_TraceStart'));directions=split('DirectionXYZ',(direction,'ReturnValue'))
exit_fraction='1.0'
for axis in 'XYZ':
 d=(directions,axis);ex=(extent,axis)
 positive=operation(axis+'Positive','GreaterEqual_DoubleDouble',d,'0','bool')
 negative_extent=operation(axis+'NegativeExtent','Multiply_DoubleDouble',ex,'-1')
 side=select(axis+'ExitSide',ex,negative_extent,positive)
 boundary=operation(axis+'Boundary','Add_DoubleDouble',(origin,axis),side)
 distance=operation(axis+'DistanceToExit','Subtract_DoubleDouble',boundary,(start,axis))
 absd=operation(axis+'AbsDirection','Abs',d)
 nonparallel=operation(axis+'Nonparallel','Greater_DoubleDouble',absd,'0.000001','bool')
 divisor=select(axis+'SafeDivisor',d,'1',nonparallel)
 fraction=operation(axis+'ExitFraction','Divide_DoubleDouble',distance,divisor)
 fraction=select(axis+'ParallelAxis',fraction,'1',nonparallel)
 exit_fraction=operation(axis+'ClipExit','FMin',exit_fraction,fraction)
exit_distance=operation('ExitDistance','Multiply_DoubleDouble',exit_fraction,(raylen,'ReturnValue'))
segment=operation('TileSegment','Subtract_DoubleDouble',exit_distance,(loop,'Array Element_Distance'))
segment=operation('NonnegativeSegment','FMax',segment,'0')
time=operation('TileSeconds','Divide_DoubleDouble',segment,speed)
# Current tile's visit was already registered; other tiles incur one new visit.
current=operation('AlreadyOnTile','LessEqual_DoubleDouble',(loop,'Array Element_Distance'),'0.000001','bool')
future_count=select('VisitIncrement','1','2',current)
count=operation('PredictedVisitDenominator','Add_DoubleDouble',(old('K2Node_VariableGet_9'),'OverlapCount'),future_count)
count=operation('PositiveVisitDenominator','FMax',count,'1')
rate=operation('TileLossPerSecond','Divide_DoubleDouble',(old('K2Node_VariableGet_2'),'BaseHomeostaticImpact'),count)
cost=operation('TileLoss','Multiply_DoubleDouble',time,rate)
total=old('K2Node_VariableGet_5');updated=operation('AccumulateTileLoss','Add_DoubleDouble',(total,'Total'),cost)
ed.rewire(accumulate,'Total',*updated)
# Passive loss is per homeostatic period, independent of tile count or pellets.
horizon_source=named('K2Node_VariableGet_19','EventGraph')
horizon=getter('Horizon',horizon_source,'HorizonLength')
gm=json.loads((ROOT/'BP_GameMode.json').read_text())
period_ref=next(p['Value'] for e in gm['Exports'] if e['ObjectName'].startswith('K2Node_VariableGet') for p in e.get('Data',[]) if p['Name']=='VariableReference' and any(q['Name']=='MemberName' and q['Value']=='PeriodTickLength' for q in p['Value']))
# MemberParent is local in the GM getter; set the external parent from the
# existing agent-to-GameMode TileHypotenuseLength getter.
ref=copy.deepcopy(prop(ed.e(old('K2Node_VariableGet_29')),'VariableReference')['Value'])
for q in ref:
 if q['Name']=='MemberName':q['Value']='PeriodTickLength'
 elif q['Name']=='MemberGuid':q['Value']=copy.deepcopy(next(q['Value'] for q in period_ref if q['Name']=='MemberGuid'))
ed.fname('PeriodTickLength')
period=getter('PeriodTicks',old('K2Node_VariableGet_29'),'PeriodTickLength','int',reference=ref)
ed.connect(old('K2Node_VariableGet_28'),'BP Game Mode',period,'self')
# Explicit native int->double conversion, avoiding implicit serialized pin types.
period_real=node('PeriodAsDouble','Conv_IntToDouble',[inp('InInt','int'),out()]);ed.connect(period,'PeriodTickLength',period_real,'InInt')
period_safe=operation('PositivePeriod','FMax',(period_real,'ReturnValue'),'1')
period_seconds=operation('PeriodSeconds','Multiply_DoubleDouble',period_safe,dt)
ratepassive=operation('PassiveLossPerSecond','Divide_DoubleDouble',passive,period_seconds)
horizon_seconds=operation('HorizonSeconds','Divide_DoubleDouble',(horizon,'HorizonLength'),speed)
passive_cost=operation('PassiveLoss','Multiply_DoubleDouble',ratepassive,horizon_seconds)
netloss=operation('TotalTraversalLoss','Add_DoubleDouble',(total,'Total'),passive_cost)
ed.rewire(result,'TotalCostValue',*netloss)
# The old empty-hit branch is bypassed. Empty rays still accrue passive loss.
# No tile diagonal, half-endpoint branch, or 0.2 multiplier feeds the return.
for i in ed.changed:
 for p in pins(ed.a,ed.e(i)):
  for j,g in p['links']:
   peer=next(q for q in pins(ed.a,ed.e(j)) if q['guid']==g)
   assert (i,p['guid']) in peer['links'],(i,p['name'],j)
(ROOT/'BP_Agent.patched.json').write_text(json.dumps(ed.a))
(ROOT/'patch_manifest.json').write_text(json.dumps({'changed_exports':sorted(ed.changed),'added_nodes':len(ed.a['Exports'])-465,'source':'55587cbe320622be0ed231f4910e02e968af78a87fff8ff0f64e4f3238087a73'},indent=2))
print('Staged',len(ed.changed),'changed exports;',len(ed.a['Exports'])-465,'new nodes')
