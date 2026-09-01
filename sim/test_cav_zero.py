import subprocess, time
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

import sys
sys.path.insert(0, ".")
from traffic_env import TrafficEnv
import traci

env = TrafficEnv(map_name="grid", gui=False, spawn_rate=0.2, cav_penetration=0.0,
                  broadcast_k=3, max_vehicles=25)
env.start()
for step in range(300):
    env.step({})
    if step % 50 == 0:
        print(f"Step {step}: active vehicles = {len(traci.vehicle.getIDList())}")

traci.close()
print("✅ CAV=0 sanity check complete")
