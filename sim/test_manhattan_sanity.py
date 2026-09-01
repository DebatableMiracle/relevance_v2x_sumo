import subprocess, time
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

import sys, os
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
sys.path.insert(0, ".")

from traffic_env import TrafficEnv
import traci

env = TrafficEnv(map_name="manhattan", gui=True, spawn_rate=0.8,
                  cav_penetration=0.5,   # zero CAVs -- pure native SUMO behavior
                  max_vehicles=100)
env.start()
print(f"Route pool size: {len(env._route_pool)}")

for step in range(300):
    env.step({})
    if step % 30 == 0:
        n_active = len(traci.vehicle.getIDList())
        print(f"step {step}: active={n_active}")
    time.sleep(0.02)

traci.close()
print("✅ Manhattan CAV=0 sanity check complete")