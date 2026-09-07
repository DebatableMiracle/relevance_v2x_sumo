"""
sim/test_comparison.py

Runs the SAME scenario (same seed, same spawn pattern) under multiple
communication conditions and reports TTC event counts + reward stats
side by side. This is what gives you an actual numbers table for Monday,
even before RL has been trained to convergence.

Conditions compared:
  A. no_comm       -- every vehicle uses IDM+MOBIL, communication OFF
  B. random_broadcast -- communication ON, but broadcast selection is
                          random-k (no learning at all)
  C. current_policy -- communication ON, broadcast selection from a
                        loaded PPO/IPPO model (if a model path is given),
                        otherwise falls back to random and is skipped

Usage:
    python test_comparison.py --map manhattan --cav 0.5 --vehicles 40 --steps 2000
    python test_comparison.py --map manhattan --model ../models/ippo_final.zip
"""
import argparse
import random
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="manhattan")
parser.add_argument("--cav", type=float, default=0.5)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--vehicles", type=int, default=40)
parser.add_argument("--spawn_rate", type=float, default=0.2)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model", default=None, help="path to a trained PPO/IPPO model .zip (optional)")
args = parser.parse_args()

import libsumo
sys.modules["traci"] = libsumo
print("[comparison] using libsumo backend (headless)")

subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

sys.path.insert(0, ".")
from traffic_env import TrafficEnv  # noqa: E402
import traci  # noqa: E402

N_MAX = 20


def run_condition(name, communication_enabled, broadcast_fn):
    """broadcast_fn(env) -> broadcast_actions dict, or None for no-comm."""
    random.seed(args.seed)  # same spawn sequence across conditions -- fair comparison

    env = TrafficEnv(
        map_name=args.map, gui=False, spawn_rate=args.spawn_rate,
        cav_penetration=args.cav, max_vehicles=args.vehicles,
        broadcast_k=args.k, communication_enabled=communication_enabled,
    )
    env.start()

    for _ in range(50):
        env.step({})

    rewards_collected = []
    for step in range(args.steps):
        broadcast_actions = broadcast_fn(env) if broadcast_fn else {}
        obs_dict, rewards, info = env.step(broadcast_actions)
        rewards_collected.extend(rewards.values())

        all_active = traci.vehicle.getIDList()
        colliding = set(traci.simulation.getCollidingVehiclesIDList())
        env.health.log_global_stats(all_active, colliding)

    event_summary = env.health.get_event_summary()
    stats_summary = env.health.get_stats_summary()
    mean_reward = sum(rewards_collected) / max(len(rewards_collected), 1)
    n_active_end = len(traci.vehicle.getIDList())

    traci.close()
    time.sleep(0.3)

    return dict(
        name=name,
        critical_events=event_summary["critical_ttc_events"],
        warning_events=event_summary["warning_ttc_events"],
        critical_per_1000=event_summary["critical_ttc_events"] / args.steps * 1000,
        warning_per_1000=event_summary["warning_ttc_events"] / args.steps * 1000,
        mean_reward=mean_reward,
        n_reward_samples=len(rewards_collected),
        n_active_end=n_active_end,
        **stats_summary,
    )


def random_broadcast_fn(env):
    obs = env.get_observations()
    actions = {}
    for tx_id, phi in obs.items():
        candidates = list(phi.keys())
        actions[tx_id] = random.sample(candidates, min(env.broadcast_k, len(candidates)))
    return actions


def make_model_broadcast_fn(model_path):
    from stable_baselines3 import PPO
    import numpy as np
    from stable_baselines3.common.utils import obs_as_tensor
    import torch

    model = PPO.load(model_path)

    def encode(tx_id, phi):
        tx_x, tx_y = traci.vehicle.getPosition(tx_id)
        candidates = sorted(
            phi.items(),
            key=lambda kv: (kv[1]["pos"][0] - tx_x) ** 2 + (kv[1]["pos"][1] - tx_y) ** 2,
        )[:N_MAX]
        candidate_ids = [cid for cid, _ in candidates]
        feat = np.zeros((N_MAX, 6), dtype=np.float32)
        tx_v = traci.vehicle.getSpeed(tx_id)
        for i, (cid, state) in enumerate(candidates):
            ox, oy = state["pos"]
            feat[i, 0] = ox - tx_x
            feat[i, 1] = oy - tx_y
            feat[i, 2] = state["speed"] - tx_v
            feat[i, 3] = state["angle"]
            feat[i, 5] = 1.0
        own_speed = traci.vehicle.getSpeed(tx_id)
        own_angle = np.radians(traci.vehicle.getAngle(tx_id))
        own = np.array([own_speed, np.sin(own_angle), np.cos(own_angle)], dtype=np.float32)
        return np.concatenate([feat.flatten(), own]), candidate_ids

    def fn(env):
        obs = env.get_observations()
        actions = {}
        for tx_id, phi in obs.items():
            if not phi:
                continue
            obs_vec, cand_ids = encode(tx_id, phi)
            action, _ = model.predict(obs_vec, deterministic=True)
            n = len(cand_ids)
            scores = action[:n]
            k = min(env.broadcast_k, n)
            top_idx = np.argsort(scores)[-k:]
            actions[tx_id] = [cand_ids[j] for j in top_idx]
        return actions

    return fn


results = []

print("\n[comparison] Running Condition A: no_comm ...")
results.append(run_condition("A: no_comm", communication_enabled=False, broadcast_fn=None))

print("[comparison] Running Condition B: random_broadcast ...")
results.append(run_condition("B: random_broadcast", communication_enabled=True, broadcast_fn=random_broadcast_fn))

if args.model:
    print(f"[comparison] Running Condition C: policy ({args.model}) ...")
    results.append(run_condition("C: learned_policy", communication_enabled=True,
                                  broadcast_fn=make_model_broadcast_fn(args.model)))
else:
    print("[comparison] No --model given, skipping Condition C.")

def fmt(v, spec=".3f"):
    return f"{v:{spec}}" if v is not None else "N/A"

print("\n" + "=" * 130)
header = (f"{'Condition':<22} {'Crit/1000':>10} {'Warn/1000':>10} {'MeanTTC':>9} "
          f"{'MinTTC':>8} {'P10 TTC':>8} {'MeanSpd':>9} {'HarshBrk/1k':>12} "
          f"{'CollSteps':>10} {'MeanRew':>9}")
print(header)
print("-" * 130)
for r in results:
    print(f"{r['name']:<22} {r['critical_per_1000']:>10.3f} {r['warning_per_1000']:>10.3f} "
          f"{fmt(r['mean_ttc']):>9} {fmt(r['min_ttc']):>8} {fmt(r['p10_ttc']):>8} "
          f"{fmt(r['mean_speed'], '.2f'):>9} {r['harsh_brakes_per_1000_steps']:>12.2f} "
          f"{r['collision_steps']:>10d} {r['mean_reward']:>9.4f}")
print("=" * 130)

if len(results) >= 2:
    a, b = results[0], results[1]
    if a["warning_per_1000"] > 0:
        pct_change = (b["warning_per_1000"] - a["warning_per_1000"]) / a["warning_per_1000"] * 100
        print(f"\nWarning-event change, no_comm -> random_broadcast: {pct_change:+.1f}%")
    if a["critical_per_1000"] > 0:
        pct_change = (b["critical_per_1000"] - a["critical_per_1000"]) / a["critical_per_1000"] * 100
        print(f"Critical-event change, no_comm -> random_broadcast: {pct_change:+.1f}%")

print("\n✅ Comparison complete")