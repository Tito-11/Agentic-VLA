from robosuite.controllers import load_composite_controller_config
c = load_composite_controller_config(controller="BASIC", robot="panda")
print(c)
