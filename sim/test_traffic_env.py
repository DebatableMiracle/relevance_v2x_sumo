import subprocess, time
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

import sys
sys.path.insert(0, ".")
from traffic_env import TrafficEnv
import traci
import random

env = TrafficEnv(config="grid.sumocfg", gui=True, spawn_rate=0.5)
env.start()

for step in range(60):
    obs = env.get_observations()  # {tx_id: phi_t}

    # dummy random broadcast policy for now — just to prove the pipeline runs
    broadcast_actions = {}
    for tx_id, phi in obs.items():
        candidates = list(phi.keys())
        selected = random.sample(candidates, min(5, len(candidates)))
        broadcast_actions[tx_id] = selected

    _, rewards = env.step(broadcast_actions)

    n_active = len(traci.vehicle.getIDList())
    n_cav = len(env.cav_ids)
    print(f"step {step}: active={n_active} cavs={n_cav} rewards={rewards}")
    time.sleep(0.03)

traci.close()
print("\n✅ TrafficEnv ran end-to-end without crashing")