"""
sim/train_ippo.py

IPPO: Independent PPO with full parameter sharing.
Every active CAV is a Tx, every simulation step. All CAVs share ONE policy
network. Each step, we batch every active CAV's observation through the
policy in a single forward pass (this IS the "all agents learn in parallel"
behavior you asked for), apply each agent's chosen broadcast, step the
(non-resetting) TrafficEnv once, and store every agent's transition into a
shared PPO rollout buffer.

This reuses Stable-Baselines3's PPO network (ActorCriticPolicy) and its
tested train() update step, but replaces SB3's single-agent
collect_rollouts() with a custom multi-agent loop, since SB3's public API
doesn't support "many agents, one shared policy, variable agent count"
out of the box.

KNOWN SIMPLIFICATIONS (documented on purpose, not hidden):
  - Per-agent episode boundaries (a CAV despawning mid-window) are NOT
    specially bootstrapped -- we simply stop adding that agent's
    transitions. This slightly biases the value function near despawn
    events. Acceptable for a first working version; can be refined later
    by tracking per-agent "done" flags properly.
  - The end-of-buffer bootstrap value uses the value estimate of the last
    batch collected, not a per-agent-correct bootstrap. Standard PPO
    truncation-bootstrap approximation, fine for a continuing task.
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="manhattan")
parser.add_argument("--cav", type=float, default=0.5)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--vehicles", type=int, default=200)
parser.add_argument("--spawn_rate", type=float, default=0.25)
parser.add_argument("--n_steps", type=int, default=512, help="agent-transitions per PPO update")
parser.add_argument("--n_updates", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--wandb", action="store_true", default=True)
parser.add_argument("--no-wandb", dest="wandb", action="store_false")
args = parser.parse_args()

# --- backend: libsumo for headless (fast), real traci if --gui was requested ---
if not args.gui:
    import libsumo
    sys.modules["traci"] = libsumo
    print("[train_ippo] using libsumo backend (headless)")
else:
    print("[train_ippo] using traci backend (gui)")

subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

sys.path.insert(0, ".")
from traffic_env import TrafficEnv  # noqa: E402
import traci  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.buffers import RolloutBuffer  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from stable_baselines3.common.utils import obs_as_tensor  # noqa: E402

from observation import encode_obs, N_MAX, OBS_DIM


class _DummySpaceEnv(gym.Env):
    """Only used so SB3's PPO constructor can read observation_space /
    action_space. Never actually stepped."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(N_MAX,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


# ---------------------------------------------------------------------------
# Build PPO (network + optimizer), reusing SB3's implementation, but we will
# drive rollout collection ourselves.
# ---------------------------------------------------------------------------
dummy_vec_env = DummyVecEnv([lambda: _DummySpaceEnv()])

model = PPO(
    "MlpPolicy",
    dummy_vec_env,
    device="cpu",
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,     # buffer_size (n_envs=1, so this IS buffer_size)
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    verbose=1,
)
device = model.device

# Replace SB3's auto-created rollout_buffer with one we control directly
# (same class, same shapes -- just being explicit about ownership).
buffer = model.rollout_buffer

# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
if args.wandb:
    import wandb
    run = wandb.init(
        project="v2x-broadcast-selection",
        config=dict(
            algo="IPPO-param-shared",
            map=args.map,
            cav_penetration=args.cav,
            broadcast_k=args.k,
            max_vehicles=args.vehicles,
            n_steps=args.n_steps,
            n_updates=args.n_updates,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
    )

# ---------------------------------------------------------------------------
# Environment (continuous, non-resetting)
# ---------------------------------------------------------------------------
env = TrafficEnv(
    map_name=args.map,
    gui=args.gui,
    spawn_rate=args.spawn_rate,
    cav_penetration=args.cav,
    max_vehicles=args.vehicles,
    broadcast_k=args.k,
)
env.start()
print(f"[train_ippo] route pool size: {len(env._route_pool)}")


# warm up so CAVs exist before the first update
for _ in range(30):
    env.step({})

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


from stable_baselines3.common.utils import configure_logger

model.set_logger(configure_logger(verbose=model.verbose))

model.policy.set_training_mode(False)

for update in range(args.n_updates):
    buffer.reset()
    ep_rewards = []
    last_values = None
    n_added = 0

    while n_added < args.n_steps:
        active_cavs = [v for v in env.cav_ids if v in traci.vehicle.getIDList()]
        if not active_cavs:
            env.step({})
            continue

        los_by_cav = env.get_observations()
        obs_batch, cand_ids_batch, tx_batch = [], [], []
        for tx_id in active_cavs:
            phi = los_by_cav.get(tx_id, {})
            obs_vec, cand_ids = encode_obs(
                tx_id, phi,
                los_by_cav=los_by_cav,
                cav_ids=env.cav_ids,
                comm_range=env.comm_range,
            )
            obs_batch.append(obs_vec)
            cand_ids_batch.append(cand_ids)
            tx_batch.append(tx_id)

        obs_tensor = obs_as_tensor(np.array(obs_batch, dtype=np.float32), device)
        with torch.no_grad():
            actions_t, values_t, log_probs_t = model.policy(obs_tensor)
        actions_np = actions_t.cpu().numpy()
        values_np = values_t.cpu().numpy()
        log_probs_np = log_probs_t.cpu().numpy()

        broadcast_actions = {}
        for i, tx_id in enumerate(tx_batch):
            cand_ids = cand_ids_batch[i]
            if not cand_ids:
                continue
            n = len(cand_ids)
            scores = actions_np[i][:n]
            k = min(env.broadcast_k, n)
            top_idx = np.argsort(scores)[-k:]
            broadcast_actions[tx_id] = [cand_ids[j] for j in top_idx]

        _, rewards, info = env.step(broadcast_actions)

        for i, tx_id in enumerate(tx_batch):
            if n_added >= args.n_steps:
                break
            r = rewards.get(tx_id, 0.0)
            ep_rewards.append(r)
            buffer.add(
                obs_batch[i],
                actions_np[i],
                np.array([r], dtype=np.float32),
                np.array([False]),          # episode_start (continuing task)
                values_t[i],                # <-- was values_np[i]
                log_probs_t[i],             # <-- was log_probs_np[i]
            )
            n_added += 1

        last_values = values_t  # bootstrap approx: last batch's values

    with torch.no_grad():
        dones = np.array([False])
        # bootstrap using the mean value of the last collected batch as an
        # approximation (see module docstring for the caveat)
        bootstrap_value = last_values.mean(dim=0, keepdim=True) if last_values is not None else torch.zeros(1, 1, device=device)
        buffer.compute_returns_and_advantage(last_values=bootstrap_value, dones=dones)

    model.policy.set_training_mode(True)
    model.train()
    model.policy.set_training_mode(False)

    sb3_metrics = {k: v for k, v in model.logger.name_to_value.items()}  

    mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    n_active = len(traci.vehicle.getIDList())
    n_cav = len(env.cav_ids)
    print(f"[update {update}] mean_reward={mean_r:.4f} agent_transitions={n_added} "
          f"active={n_active} cavs={n_cav}")

    if args.wandb:
        wandb.log({
            "train/mean_reward": mean_r,
            "train/agent_transitions": n_added,
            "env/active_vehicles": n_active,
            "env/cav_count": n_cav,
            "update": update,
        })

    if update % 20 == 0 and update > 0:
        model.save(f"../models/ippo_update_{update}")

traci.close()
model.save("../models/ippo_final")
if args.wandb:
    run.finish()
print("✅ IPPO training complete")