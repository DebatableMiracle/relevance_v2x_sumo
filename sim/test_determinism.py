import sys, time
sys.path.insert(0, ".")
from harness import start, rollout
import traci

TRACK_ID = "t_0"  # matches your .rou.xml trip id

start(gui=False)
traci.simulationStep()
traci.simulation.saveState("snap.xml")
traj_a = rollout(20, track_ids=[TRACK_ID])
traci.close()

time.sleep(0.5)

start(gui=False)
traci.simulation.loadState("snap.xml")
traj_b = rollout(20, track_ids=[TRACK_ID])
traci.close()

print("Branch A:", traj_a[TRACK_ID][:5], "...")
print("Branch B:", traj_b[TRACK_ID][:5], "...")
assert len(traj_a[TRACK_ID]) > 0, "Vehicle never appeared — check depart time / ID"
assert traj_a == traj_b, "NON-DETERMINISTIC — check seed and .sumocfg config"
print(f"✅ Deterministic replay confirmed over {len(traj_a[TRACK_ID])} tracked steps")