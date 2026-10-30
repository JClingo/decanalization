"""Compile and save only BP_Agent; inspect the compiler log for errors."""
import unreal
path='/Game/Agent/BP_Agent'
bp=unreal.load_asset(path)
assert bp is not None
unreal.BlueprintEditorLibrary.compile_blueprint(bp)
assert unreal.EditorAssetLibrary.save_asset(path,False)
unreal.log('ANTICIPATION_COMPILED')
