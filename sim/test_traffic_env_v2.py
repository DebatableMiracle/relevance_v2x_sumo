import subprocess, time
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

import sys, random
sys.path.insert(0, ".")
from traffic_env import TrafficEnv
import traci

env = TrafficEnv(config="grid.sumocfg", net_file="../net/grid.net.xml",
                  gui=True, spawn_rate=0.4, cav_penetration=0.5,
                  broadcast_k=5, checkpoint_every=100)
env.start()

print(f"Route pool size: {len(env._route_pool)}")

for step in range(150):
    obs = env.get_observations()

    broadcast_actions = {}
    for tx_id, phi in obs.items():
        candidates = list(phi.keys())
        selected = random.sample(candidates, min(env.broadcast_k, len(candidates)))
        broadcast_actions[tx_id] = selected

    _, rewards, info = env.step(broadcast_actions)

    if step % 10 == 0:
        n_active = len(traci.vehicle.getIDList())
        n_cav = len(env.cav_ids)
        avg_r = sum(rewards.values()) / max(len(rewards), 1)
        print(f"step {step}: active={n_active} cavs={n_cav} avg_reward={avg_r:.4f}")

traci.close()
print("\n✅ v2 TrafficEnv ran end-to-end, fringe spawning + gated delta reward active")