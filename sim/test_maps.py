import subprocess, time
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

import sys
sys.path.insert(0, ".")
from gym_env import BroadcastGymEnv
import traci

for map_name in ["corridor", "circular", "grid"]:
    print(f"\n--- Testing Map: {map_name} ---")
    env = BroadcastGymEnv(map_name=map_name, gui=False, spawn_rate=0.2, cav_penetration=0.5, broadcast_k=3, max_vehicles=25, max_steps_per_episode=64)
    obs, info = env.reset()
    print(f"Reset obs shape: {obs.shape}")
    for s in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
    print(f"Completed 30 steps on {map_name}. Final reward: {reward:.4f}, truncated: {truncated}")
    env.close()
    subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
    time.sleep(0.5)

print("\n✅ All 3 maps (corridor, circular, grid) verified end-to-end!")
