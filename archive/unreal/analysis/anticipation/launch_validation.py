import unreal,time,pathlib
started=time.monotonic()
def anticipation_validation_tick(delta):
 if time.monotonic()-started<2:return
 unreal.unregister_slate_post_tick_callback(anticipation_validation_handle)
 script=pathlib.Path(__file__).resolve().with_name('validate_anticipation.py')
 exec(compile(script.read_text(),str(script),'exec'),globals())
anticipation_validation_handle=unreal.register_slate_post_tick_callback(anticipation_validation_tick)
