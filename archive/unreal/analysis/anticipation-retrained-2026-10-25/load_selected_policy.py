"""Apply a selected policy in the Unreal editor, with play stopped.

Importing this module changes nothing. Call apply('HRL') or apply('NF') to set
inference mode and the selected family. Trial order and physical settings stay
as configured in the project. The corrected anticipation asset is required.
"""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def apply(family):
 import unreal
 family=family.upper();assert family in ['HRL','NF'],'Choose HRL or NF'
 catalog=json.loads((BASE/'selected_policy_catalog.json').read_text());entry=catalog['policies'][family]
 world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
 assert world.get_path_name()=='/Game/Level/Landscape.Landscape','Open the Landscape map with play stopped before applying a policy'
 project=Path(unreal.Paths.project_dir()).resolve()
 assert digest(project/'Content/Agent/BP_Agent.uasset')==catalog['anticipation_asset_sha256'],'Install the catalogued corrected BP_Agent asset before loading these policies'
 files={k:BASE/v['file'] for k,v in entry['snapshots'].items()}
 for k,p in files.items():assert digest(p)==entry['snapshots'][k]['sha256'],k+' snapshot hash mismatch'
 bp=unreal.load_asset('/Game/Agent/BP_GameMode');c=unreal.get_default_object(bp.generated_class());c.set_editor_property('IsHRL',family=='HRL');c.set_editor_property('IsTraining',False);unreal.BlueprintEditorLibrary.compile_blueprint(bp);assert unreal.EditorAssetLibrary.save_loaded_asset(bp,False)
 bp=unreal.load_asset('/Game/Learning/BP_AgentManager')
 for prop in ['Run Inference','EncoderSnapshot','PolicySnapshot','Decoder Snapshot','Critic Snapshot']:unreal.BlueprintEditorLibrary.set_blueprint_variable_instance_editable(bp,prop,True)
 unreal.BlueprintEditorLibrary.compile_blueprint(bp);c=unreal.get_default_object(bp.generated_class());c.set_editor_property('Run Inference',True)
 props={'EncoderSnapshot':'encoder','PolicySnapshot':'policy','Decoder Snapshot':'decoder','Critic Snapshot':'critic'}
 for prop,kind in props.items():c.set_editor_property(prop,unreal.FilePath(file_path=str(files[kind])))
 unreal.BlueprintEditorLibrary.compile_blueprint(bp);assert unreal.EditorAssetLibrary.save_loaded_asset(bp,False)
 cls=unreal.load_class(None,'/Game/Learning/BP_AgentManager.BP_AgentManager_C')
 for actor in unreal.GameplayStatics.get_all_actors_of_class(world,cls):
  actor.set_editor_property('Run Inference',True)
  for prop in props:actor.set_editor_property(prop,c.get_editor_property(prop))
 manifest=project/'Content/Python/trial_training_manifests.json';records=json.loads(manifest.read_text());record=json.loads((BASE/entry['training_manifest']).read_text());record={k:v for k,v in record.items() if k!='held_out_selection'}
 identity={k:entry['snapshots'][k]['sha256'] for k in ['encoder','policy','decoder']}
 matches=[r for r in records if r['snapshot-sha256']==identity]
 assert len(matches)<=1,'Existing training manifests contain duplicate network records'
 if not matches:
  records.append(record);tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(records,indent=2)+'\n');tmp.replace(manifest)
 assert unreal.EditorLoadingAndSavingUtils.save_map(world,'/Game/Level/Landscape')
 unreal.log('SELECTED_POLICY_APPLIED '+family+' '+str(BASE))
 return entry
