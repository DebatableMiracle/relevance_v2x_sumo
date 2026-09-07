"""
Gymnasium wrapper implementing the Box-action/top-k broadcast selection design.
Single-agent framing for now: at each step, we pick ONE active CAV as "the" Tx
(round-robin or random), so this trains a per-Tx policy in a single-agent shell.
This is the PPO-first stepping stone before MAPPO.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from traffic_env import TrafficEnv
from observation import encode_obs, N_MAX, FEAT_DIM, OWN_DIM
import traci


class BroadcastGymEnv(gym.Env):
    def __init__(self, map_name="corridor", config=None, net_file=None,
                 gui=False, spawn_rate=0.2, cav_penetration=0.5, broadcast_k=3,
                 max_steps_per_episode=64, max_vehicles=25):
        super().__init__()
        self.env = TrafficEnv(map_name=map_name, gui=gui,
                               spawn_rate=spawn_rate, cav_penetration=cav_penetration,
                               broadcast_k=broadcast_k, max_vehicles=max_vehicles)
        self.max_steps_per_episode = max_steps_per_episode
        self._episode_step = 0
        self._started = False

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(N_MAX * FEAT_DIM + OWN_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(N_MAX,), dtype=np.float32
        )

        self._current_tx = None
        self._current_candidates = []

    def _pick_tx(self):
        cav_list = list(self.env.cav_ids)
        if not cav_list:
            return None
        return np.random.choice(cav_list)

    def _encode_obs(self, tx_id, phi, los_by_cav=None):
        los_by_cav = los_by_cav if los_by_cav is not None else {tx_id: phi}
        obs, cands = encode_obs(
            tx_id, phi,
            los_by_cav=los_by_cav,
            cav_ids=self.env.cav_ids,
            comm_range=self.env.comm_range,
        )
        self._current_candidates = cands
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._started:
            self.env.start()
            self._started = True
        self._episode_step = 0

        # warm up a few steps with no broadcast so CAVs exist before we pick a Tx
        for _ in range(10):
            self.env.step({})
            if self.env.cav_ids:
                break

        self._current_tx = self._pick_tx()
        if self._current_tx is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._current_candidates = []
        else:
            phi = self.env.get_observations()
            los = phi
            obs = self._encode_obs(self._current_tx, los.get(self._current_tx, {}), los)

        return obs, {}

    def step(self, action):
        self._episode_step += 1

        broadcast_actions = {}
        if self._current_tx is not None and self._current_candidates:
            n = len(self._current_candidates)
            scores = action[:n]
            k = self.env.broadcast_k
            top_idx = np.argsort(scores)[-k:]
            selected = [self._current_candidates[i] for i in top_idx]
            broadcast_actions[self._current_tx] = selected

        obs_dict, rewards, info = self.env.step(broadcast_actions)

        reward = rewards.get(self._current_tx, 0.0) if self._current_tx else 0.0

        self._current_tx = self._pick_tx()
        if self._current_tx is None:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._current_candidates = []
        else:
            phi = obs_dict.get(self._current_tx, {})
            next_obs = self._encode_obs(self._current_tx, phi, obs_dict)

        terminated = False
        truncated = self._episode_step >= self.max_steps_per_episode

        step_info = info.get(self._current_tx, {}) if self._current_tx else {}
        return next_obs, float(reward), terminated, truncated, step_info

    def close(self):
        try:
            traci.close()
        except Exception:
            pass