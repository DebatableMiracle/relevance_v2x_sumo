"""
Road-health reward computation. Standalone, testable independent of RL.
Improvements over v1: frozen normalization (no drift during training),
delta-reward (change, not level), sample-size gating, real collision
detection, no-leader vehicles excluded from TTC percentile.
"""
import numpy as np
import traci

HARSH_BRAKE_THRESHOLD = -3.0   # m/s^2
COLLISION_PENALTY = 50.0
WARMUP_STEPS = 200               # steps to collect stats before freezing normalization
SAMPLE_GATE_C = 3.0                # smoothing constant for local-sample-size gating


class RunningNorm:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4
        self.frozen = False

    def update(self, x):
        if self.frozen:
            return
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def freeze(self):
        self.frozen = True

    def normalize(self, x):
        std = max((self.var / max(self.count, 1)) ** 0.5, 1e-3)
        return (x - self.mean) / std


class RoadHealthTracker:
    def __init__(self, w_ttc=1.0, w_ttc_var=0.5, w_collision=2.0,
                 w_harsh_brake=0.5, w_vel=0.5, w_vel_var=0.3,
                 comm_range=175.0):
        self.w = dict(ttc=w_ttc, ttc_var=w_ttc_var, collision=w_collision,
                      harsh_brake=w_harsh_brake, vel=w_vel, vel_var=w_vel_var)
        self.comm_range = comm_range
        self.norms = {k: RunningNorm() for k in ["ttc_p10", "ttc_var", "vel_mean", "vel_var"]}
        self._prev_speed = {}
        self._prev_H = {}   # per-Tx last H value, for delta reward
        self._step_count = 0

    def _prune_stale(self):
        active = set(traci.vehicle.getIDList())
        self._prev_speed = {k: v for k, v in self._prev_speed.items() if k in active}
        self._prev_H = {k: v for k, v in self._prev_H.items() if k in active}

    def _maybe_freeze(self):
        self._step_count += 1
        if self._step_count == WARMUP_STEPS:
            for n in self.norms.values():
                n.freeze()

    def _raw_health(self, vehicle_ids, colliding_ids):
        if not vehicle_ids:
            return None

        ttcs, speeds, harsh_brakes = [], [], 0
        for vid in vehicle_ids:
            if vid not in traci.vehicle.getIDList():
                continue
            speed = traci.vehicle.getSpeed(vid)
            speeds.append(speed)

            leader = traci.vehicle.getLeader(vid, self.comm_range)
            if leader is not None:
                leader_id, gap = leader
                leader_speed = traci.vehicle.getSpeed(leader_id)
                closing = speed - leader_speed
                if closing > 0.1:
                    ttcs.append(gap / closing)
            # no-leader vehicles: excluded from TTC entirely, not padded with a sentinel

            prev = self._prev_speed.get(vid, speed)
            accel = (speed - prev) / 0.1
            if accel < HARSH_BRAKE_THRESHOLD:
                harsh_brakes += 1
            self._prev_speed[vid] = speed

        ttc_p10 = np.percentile(ttcs, 10) if len(ttcs) >= 2 else (ttcs[0] if ttcs else None)
        ttc_var = np.var(ttcs) if len(ttcs) > 1 else 0.0
        vel_mean = np.mean(speeds) if speeds else 0.0
        vel_var = np.var(speeds) if len(speeds) > 1 else 0.0
        collision = bool(colliding_ids & set(vehicle_ids))

        for k, v in [("ttc_var", ttc_var), ("vel_mean", vel_mean), ("vel_var", vel_var)]:
            self.norms[k].update(v)
        if ttc_p10 is not None:
            self.norms["ttc_p10"].update(ttc_p10)

        n_ttc = self.norms["ttc_p10"].normalize(ttc_p10) if ttc_p10 is not None else 0.0
        n_ttc_var = self.norms["ttc_var"].normalize(ttc_var)
        n_vel = self.norms["vel_mean"].normalize(vel_mean)
        n_vel_var = self.norms["vel_var"].normalize(vel_var)

        H = (self.w["ttc"] * n_ttc
             - self.w["ttc_var"] * n_ttc_var
             - self.w["collision"] * COLLISION_PENALTY * float(collision)
             - self.w["harsh_brake"] * harsh_brakes
             + self.w["vel"] * n_vel
             - self.w["vel_var"] * n_vel_var)

        return dict(H=H, ttc_p10=ttc_p10, ttc_var=ttc_var, vel_mean=vel_mean,
                     vel_var=vel_var, harsh_brakes=harsh_brakes, collision=collision,
                     n_conflict_pairs=len(ttcs), n_vehicles=len(vehicle_ids))

    def compute_reward(self, tx_id, vehicle_ids, colliding_ids):
        """
        Returns (reward, breakdown). reward is a GATED DELTA:
            reward = [n/(n+c)] * (H_t - H_{t-1})
        Call once per Tx per step. Must call step_maintenance() once per
        global sim step (not per-Tx) to advance warmup/pruning correctly.
        """
        raw = self._raw_health(vehicle_ids, colliding_ids)
        if raw is None:
            return 0.0, None

        H = raw["H"]
        prev_H = self._prev_H.get(tx_id, H)  # first call: no delta, reward 0
        delta = H - prev_H
        self._prev_H[tx_id] = H

        n = raw["n_vehicles"]
        gate = n / (n + SAMPLE_GATE_C)
        reward = gate * delta

        raw["reward"] = reward
        raw["gate"] = gate
        return reward, raw

    def step_maintenance(self):
        """Call exactly once per simulation step (not per Tx)."""
        self._maybe_freeze()
        self._prune_stale()

        