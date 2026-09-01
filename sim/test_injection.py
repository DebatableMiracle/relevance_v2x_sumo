import sys, time
sys.path.insert(0, ".")
from harness import start, inject_object
import traci

start(gui=True)  # use GUI so you can SEE the injected object appear

# need a dummy route for the injected vehicle to be added under
traci.route.add("dummy_route", ["E0"])

traci.simulationStep()

inject_object("obj_0", "dummy_route", x=-50, y=10, angle=90)

for step in range(200):
    traci.simulationStep()
    if "obj_0" in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition("obj_0")
        print(f"step {step}: obj_0 at {pos}")
    time.sleep(0.05)  # slow down so you can watch it in GUI

traci.close()