"""Exercise the compiled Blueprint with real Unreal collision hits.

Runs only in an isolated project and transient map. Expected tile costs come
from an independent segment/AABB intersection, not the Blueprint graph.
"""
import unreal,json,math,pathlib,traceback
project=pathlib.Path(unreal.Paths.project_dir()).resolve()
assert (project/'ANTICIPATION_TEST_COPY').exists(), 'Run only in a disposable copy carrying ANTICIPATION_TEST_COPY.'
OUT=project/'Saved/AnticipationFix/runtime_validation.json';OUT.parent.mkdir(parents=True,exist_ok=True)
report={'tests':[]}
def check(label,actual,expected,tolerance=1e-5):
 actual=float(actual[0] if isinstance(actual,tuple) else actual);expected=float(expected);ok=math.isfinite(actual) and abs(actual-expected)<=tolerance*max(1,abs(expected))
 report['tests'].append({'name':label,'actual':actual,'expected':expected,'passed':ok});return ok
def v(t):return unreal.Vector(*t)
def length(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
try:
 # Test-only property access; recompile in memory and never save these flags.
 for path,fields in [('/Game/Agent/BP_GameMode',['PeriodTickLength','Setpoints','AnticipatedNets']),('/Game/Resource/BP_Resource',['HomeostaticImpact']),('/Game/Level/BP_Tile',['BaseHomeostaticImpact','OverlapCount']),('/Game/Agent/BP_Agent',['BP Game Mode','HorizonLength'])]:
  bp=unreal.load_asset(path)
  for field in fields:unreal.BlueprintEditorLibrary.set_blueprint_variable_instance_editable(bp,field,True)
  unreal.BlueprintEditorLibrary.compile_blueprint(bp)
 world=unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
 subsystem=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
 def spawn(path,location):
  cls=unreal.load_class(None,path+'_C');assert cls is not None,path
  actor=subsystem.spawn_actor_from_class(cls,v(location));assert actor is not None,path
  actor.set_actor_location(v(location),False,False)
  for component in actor.get_components_by_class(unreal.PrimitiveComponent):component.set_collision_object_type(next(getattr(unreal.CollisionChannel,n) for n in dir(unreal.CollisionChannel) if 'WORLD_DYNAMIC' in n))
  return actor
 agent=spawn('/Game/Agent/BP_Agent.BP_Agent',(0,-10000,100))
 gm=spawn('/Game/Agent/BP_GameMode.BP_GameMode',(0,-20000,0));agent.set_editor_property('BP Game Mode',gm);gm.set_editor_property('PeriodTickLength',5)
 def gain(hits,rate=0.2,speed=1000):return agent.call_method('GetAnticipatedResourceYield',(float(speed),hits,float(rate)))[-1]
 def loss(hits,horizon,rate=0.2,delta=0.05,speed=1000):
  agent.set_editor_property('HorizonLength',float(horizon))
  return agent.call_method('GetAnticipatedTraversalCosts',(hits,float(rate),float(delta),float(speed)))[-1]
 check('empty resource ray',gain([]),0)
 check('passive loss with no tiles',loss([],2500),2)
 resources=[]
 for x,value in [(500,3),(1000,7),(1500,11)]:
  a=spawn('/Game/Resource/BP_Resource.BP_Resource',(x,0,200));a.set_editor_property('HomeostaticImpact',value);resources.append(a)
 object_types=[getattr(unreal.ObjectTypeQuery,'OBJECT_TYPE_QUERY'+str(i)) for i in range(1,7)]
 def trace(start,end,ignore):
  hits=unreal.SystemLibrary.line_trace_multi_for_objects(world,v(start),v(end),object_types,False,ignore,unreal.DrawDebugTrace.NONE,True)
  if isinstance(hits,tuple) and len(hits)==2 and isinstance(hits[0],bool):hits=hits[1]
  return list(hits or [])
 rhits=trace((0,0,200),(2000,0,200),[agent,gm]);report['resource_hit_count']=len(rhits);report['resource_hits']=[h.export_text() for h in rhits]
 check('three real resource hits',len(rhits),3)
 check('sum all pellet values',gain(rhits),21)
 check('pellet sum independent of hit order',gain(list(reversed(rhits))),21)
 check('pellet gain has no duplicated passive charge',gain(rhits,100),21)
 check('resource total resets on next call',gain([]),0)
 tiles=[]
 for x in [0,1000,2000]:
  a=spawn('/Game/Level/BP_Tile.BP_Tile',(x,0,-50));cube=a.get_editor_property('Cube');cube.set_world_scale3d(unreal.Vector(10,10,1))
  a.set_editor_property('BaseHomeostaticImpact',10.0);a.set_editor_property('OverlapCount',0.0);tiles.append(a)
 def bounds(tile):
  origin,extent,r=unreal.SystemLibrary.get_component_bounds(tile.get_editor_property('Cube'))
  c=(origin.x,origin.y,origin.z);e=(extent.x,extent.y,extent.z)
  return tuple(x-y for x,y in zip(c,e)),tuple(x+y for x,y in zip(c,e))
 report['tile_bounds']=[bounds(t) for t in tiles]
 def expected(start,end,speed=1000,passive=.2,dt=.05,period=5):
  ray=tuple(y-x for x,y in zip(start,end));raylen=length(start,end);total=raylen/speed*passive/(period*dt)
  for t in tiles:
   lo,hi=bounds(t);enter=0.;exit=1.
   for p,d,lower,upper in zip(start,ray,lo,hi):
    if abs(d)<1e-9:
     if p<lower or p>upper:exit=-1;break
    else:
     x,y=sorted(((lower-p)/d,(upper-p)/d));enter=max(enter,x);exit=min(exit,y)
   if exit<=enter:continue
   count=float(t.get_editor_property('OverlapCount'))
   # The current tile's overlap has already incremented its visit count.
   denominator=count+(1 if enter<=1e-9 else 2)
   total+=(exit-enter)*raylen/speed*float(t.get_editor_property('BaseHomeostaticImpact'))/denominator
  return total
 def scenario(label,start,end,speed=1000,passive=.2,dt=.05,period=5):
  gm.set_editor_property('PeriodTickLength',period)
  hits=trace(start,end,[agent,gm]+resources)
  report.setdefault('scenarios',[]).append({'name':label,'hits':len(hits),'hit_text':[h.export_text() for h in hits]})
  actual=loss(hits,length(start,end),passive,dt,speed)
  check(label,actual,expected(start,end,speed,passive,dt,period));return hits,actual
 tiles[0].set_editor_property('OverlapCount',1.)
 thits,tloss=scenario('three tiles with partial endpoints',(0,0,-1),(2500,0,-1))
 check('known 2.5-second tile plus passive loss',tloss,14.5)
 scenario('short ray wholly inside one tile',(100,0,-1),(300,0,-1))
 scenario('angled ray crossing tile edges',(-300,-400,-1),(1800,400,-1))
 scenario('reverse direction',(2400,0,-1),(200,0,-1))
 scenario('parallel to other axis',(100,-300,-1),(100,300,-1))
 for i,t in enumerate(tiles):t.set_editor_property('OverlapCount',float(i*3+1))
 scenario('different canalization on each tile',(0,0,-1),(2400,0,-1))
 scenario('speed scales travel time',(0,0,-1),(2400,0,-1),speed=500)
 scenario('passive update period conversion',(0,0,-1),(2400,0,-1),period=10)
 scenario('different simulation timestep',(0,0,-1),(2400,0,-1),dt=.1)
 scenario('zero passive loss',(0,0,-1),(2400,0,-1),passive=0)
 check('tile total resets between calls',loss([],1000),.8)
 check('zero speed is finite',loss(thits,2500,speed=0),0)
 check('zero delta is finite',loss(thits,2500,delta=0),0)
 check('tile actors do not contribute pellet gain',gain(thits),0)
 gm.set_editor_property('Setpoints',[-15.]);gm.set_editor_property('AnticipatedNets',[0.])
 gm.call_method('HandlePerception',(21.,14.5))
 check('net prediction is pellet gain minus traversal loss',gm.get_editor_property('AnticipatedNets')[-1],6.5)
 assert all(t['passed'] for t in report['tests']),[t for t in report['tests'] if not t['passed']]
 report['passed']=True
 unreal.log('ANTICIPATION_VALIDATION_OK '+str(len(report['tests']))+' tests')
except Exception:
 report['passed']=False;report['error']=traceback.format_exc();raise
finally:
 OUT.write_text(json.dumps(report,indent=2))
 unreal.SystemLibrary.quit_editor()
