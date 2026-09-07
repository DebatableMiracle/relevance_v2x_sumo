"""
sim/test_ttc_frequency.py

Diagnostic (NOT training): runs the environment with NO broadcast at all
and counts how often TTC drops below the warning (2s) and critical (1s)
thresholds. This tells you whether your current map/density/CAV% setup
even generates enough conflict situations for RL to have something to
learn from -- run this BEFORE worrying about policy/reward sophistication.

Usage:
    python test_ttc_frequency.py --map manhattan --cav 0.5 --vehicles 40 --steps 2000
"""
import argparse
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="manhattan")
parser.add_argument("--cav", type=float, default=0.5)
parser.add_argument("--vehicles", type=int, default=40)
parser.add_argument("--spawn_rate", type=float, default=0.2)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--gui", action="store_true")
args = parser.parse_args()

if not args.gui:
    import libsumo
    sys.modules["traci"] = libsumo
    print("[ttc_freq] using libsumo backend (headless)")

subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

sys.path.insert(0, ".")
from traffic_env import TrafficEnv  # noqa: E402
import traci  # noqa: E402

env = TrafficEnv(
    map_name=args.map,
    gui=args.gui,
    spawn_rate=args.spawn_rate,
    cav_penetration=args.cav,
    max_vehicles=args.vehicles,
    broadcast_k=3,
    communication_enabled=False,   # <-- the whole point: NO broadcast, isolate raw conflict rate
)
env.start()
print(f"[ttc_freq] route pool size: {len(env._route_pool)}")

# warm up
for _ in range(50):
    env.step({})

print(f"[ttc_freq] running {args.steps} steps, logging TTC events...")
t0 = time.time()
step_counts = []
for step in range(args.steps):
    env.step({})  # empty broadcast_actions -- communication_enabled=False makes this moot anyway
    all_active = traci.vehicle.getIDList()
    colliding = set(traci.simulation.getCollidingVehiclesIDList())
    env.health.log_ttc_events(all_active)
    env.health.log_global_stats(all_active, colliding)
    counts = env.health.get_step_event_counts()
    step_counts.append(counts)

    if step % 200 == 0:
        n_active = len(all_active)
        summary = env.health.get_event_summary()
        elapsed = time.time() - t0
        print(f"  step {step}: active={n_active} "
              f"cumulative_critical={summary['critical_ttc_events']} "
              f"cumulative_warning={summary['warning_ttc_events']} "
              f"({elapsed:.1f}s elapsed)")

traci.close()

summary = env.health.get_event_summary()
total_critical = summary["critical_ttc_events"]
total_warning = summary["warning_ttc_events"]

stats = env.health.get_stats_summary()

print("\n" + "=" * 60)
print(f"TTC FREQUENCY DIAGNOSTIC -- {args.steps} steps, map={args.map}, "
      f"cav%={args.cav}, max_vehicles={args.vehicles}")
print("-" * 60)
print(f"Critical events (TTC < 1.0s): {total_critical}  "
      f"({total_critical / args.steps * 1000:.2f} per 1000 steps)")
print(f"Warning events  (TTC < 2.0s): {total_warning}  "
      f"({total_warning / args.steps * 1000:.2f} per 1000 steps)")
print("-" * 60)
print(f"Mean TTC:          {stats['mean_ttc']}")
print(f"Min TTC:           {stats['min_ttc']}")
print(f"10th pct TTC:      {stats['p10_ttc']}")
print(f"Mean speed (m/s):  {stats['mean_speed']}")
print(f"Harsh brakes/1000: {stats['harsh_brakes_per_1000_steps']:.2f}")
print(f"Collision steps:   {stats['collision_steps']}")
print("=" * 60)

if total_critical == 0 and total_warning == 0:
    print("\n⚠️  ZERO conflict events detected. RL has nothing to learn from at "
          "this density/CAV%/map. Increase density, reduce SENSE_RANGE relative "
          "to COMM_RANGE, or construct deliberate occlusion-conflict scenarios.")
elif total_warning / args.steps * 1000 < 1.0:
    print("\n⚠️  Conflict events are rare (<1 per 1000 steps). Training will be "
          "slow to find signal. Consider increasing density or CAV%.")
else:
    print("\n✅ Meaningful conflict frequency detected -- reasonable training signal available.")