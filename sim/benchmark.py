"""
sim/benchmark.py

Profiles TrafficEnv.step() throughput, headless, with realistic CAV load.
Run this any time you change perception/reward/policy code to check you
haven't regressed performance, and to compare traci vs libsumo directly.

Usage:
    python benchmark.py                 # uses libsumo (fast, headless-only)
    python benchmark.py --backend traci  # uses real traci (for comparison)
    python benchmark.py --map grid --cav 0.5 --k 3 --vehicles 40 --steps 100
"""
import argparse
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["libsumo", "traci"], default="libsumo")
parser.add_argument("--map", default="manhattan")
parser.add_argument("--cav", type=float, default=0.5, help="CAV penetration")
parser.add_argument("--k", type=int, default=3, help="broadcast_k")
parser.add_argument("--vehicles", type=int, default=40, help="max_vehicles")
parser.add_argument("--spawn_rate", type=float, default=0.2)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--steps", type=int, default=100)
args = parser.parse_args()

# --- backend selection must happen BEFORE traffic_env (and anything it
#     imports) does `import traci` ---
if args.backend == "libsumo":
    import libsumo
    sys.modules["traci"] = libsumo
    print("[benchmark] using libsumo backend")
else:
    print("[benchmark] using traci backend")

# clean slate: kill any leftover sumo processes from prior runs
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

sys.path.insert(0, ".")
from traffic_env import TrafficEnv  # noqa: E402
import traci  # noqa: E402  (resolves to libsumo or real traci per sys.modules above)

env = TrafficEnv(
    map_name=args.map,
    gui=False,  # libsumo can't do gui anyway; keep this script headless-only
    spawn_rate=args.spawn_rate,
    cav_penetration=args.cav,
    max_vehicles=args.vehicles,
    broadcast_k=args.k,
)
env.start()
print(f"[benchmark] route pool size: {len(env._route_pool)}")

print(f"[benchmark] warming up ({args.warmup} steps)...")
for _ in range(args.warmup):
    
    env.step({})

n_active_warmup = len(traci.vehicle.getIDList())
n_cav_warmup = len(env.cav_ids)
print(f"[benchmark] after warmup: active={n_active_warmup} cavs={n_cav_warmup}")

print(f"[benchmark] timing {args.steps} steps...")
t0 = time.time()
for step in range(args.steps):          # <-- change `for _ in range(...)` to `for step in range(...)`
    obs = env.get_observations()
    actions = {tx: list(phi.keys())[:args.k] for tx, phi in obs.items()}
    obs_dict, rewards, info = env.step(actions)   # <-- capture the return values (was just `env.step(actions)` before)
    print(f"step {step}: reward_sample={list(rewards.values())[:3] if rewards else 'none'}")   # <-- add this line
elapsed = time.time() - t0

n_active_end = len(traci.vehicle.getIDList())
n_cav_end = len(env.cav_ids)

traci.close()

ms_per_step = elapsed / args.steps * 1000
steps_per_sec = args.steps / elapsed

print("\n" + "=" * 50)
print(f"backend:        {args.backend}")
print(f"map:            {args.map}")
print(f"cav_penetration:{args.cav}")
print(f"broadcast_k:    {args.k}")
print(f"max_vehicles:   {args.vehicles}")
print(f"active (start/end): {n_active_warmup} / {n_active_end}")
print(f"cavs (start/end):   {n_cav_warmup} / {n_cav_end}")
print("-" * 50)
print(f"total time:     {elapsed:.2f}s for {args.steps} steps")
print(f"ms/step:        {ms_per_step:.1f}")
print(f"steps/sec:      {steps_per_sec:.2f}")
print("=" * 50)
