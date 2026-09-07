"""
sim/road_health.py

Road-health reward computation + safety event logging. Standalone,
testable independent of RL.

FIXES applied (vs earlier version):
  - WARMUP_STEPS raised 200 -> 2000 (normalization freezes on a more
    representative sample of traffic conditions, not just early/light
    density)
  - Final reward is bounded via tanh() regardless of upstream normalization
    quality -- prevents a single extreme event from producing a
    training-destabilizing outlier
  - No-leader vehicles are EXCLUDED from the TTC percentile (not padded
    with a sentinel value)
  - Real collision detection wired in (colliding_ids passed by caller)
  - NEW: discrete TTC threshold event logging (warning <2s, critical <1s)
    for reporting hard, interpretable safety numbers alongside the
    continuous RL reward
"""
import numpy as np
import traci

HARSH_BRAKE_THRESHOLD = -3.0   # m/s^2
COLLISION_PENALTY = 50.0
WARMUP_STEPS = 2000               # global sim steps before normalization freezes
SAMPLE_GATE_C = 3.0                 # smoothing constant for local-sample-size gating
REWARD_TANH_SCALE = 5.0               # final reward bound: tanh(raw / this)

TTC_CRITICAL = 1.0   # seconds -- hard safety threshold
TTC_WARNING = 2.0      # seconds -- soft safety threshold


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
        self._prev_H = {}
        self._step_count = 0

        # event logging
        self.event_log = []
        self._events_this_step = {"critical": 0, "warning": 0}

        # global descriptive stats (separate from reward -- for reporting)
        self._stats_ttc = []
        self._stats_speed = []
        self._stats_harsh_brakes = 0
        self._stats_collisions = 0
        self._stats_steps = 0
        self._stats_prev_speed = {}

    # ---------------------------------------------------------------
    # per-global-step maintenance (call ONCE per env.step(), not per Tx)
    # ---------------------------------------------------------------
    def step_maintenance(self):
        self._step_count += 1
        if self._step_count == WARMUP_STEPS:
            for n in self.norms.values():
                n.freeze()
        active = set(traci.vehicle.getIDList())
        self._prev_speed = {k: v for k, v in self._prev_speed.items() if k in active}
        self._prev_H = {k: v for k, v in self._prev_H.items() if k in active}

    # ---------------------------------------------------------------
    # TTC threshold event detection/logging
    # ---------------------------------------------------------------
    def log_ttc_events(self, vehicle_ids):
        """Call once per step with the full active vehicle list (not just
        one Tx's local set) to get a global, de-duplicated event count."""
        sim_time = traci.simulation.getTime()
        self._events_this_step = {"critical": 0, "warning": 0}
        for vid in vehicle_ids:
            if vid not in traci.vehicle.getIDList():
                continue
            leader = traci.vehicle.getLeader(vid, self.comm_range)
            if leader is None:
                continue
            leader_id, gap = leader
            speed = traci.vehicle.getSpeed(vid)
            leader_speed = traci.vehicle.getSpeed(leader_id)
            closing = speed - leader_speed
            if closing <= 0.1 or gap < 0:
                continue
            ttc = gap / closing

            if ttc < TTC_CRITICAL:
                self.event_log.append({"time": sim_time, "vehicle": vid,
                                        "leader": leader_id, "ttc": ttc,
                                        "severity": "critical"})
                self._events_this_step["critical"] += 1
            elif ttc < TTC_WARNING:
                self.event_log.append({"time": sim_time, "vehicle": vid,
                                        "leader": leader_id, "ttc": ttc,
                                        "severity": "warning"})
                self._events_this_step["warning"] += 1

    def get_step_event_counts(self):
        return dict(self._events_this_step)

    def get_event_summary(self):
        critical = sum(1 for e in self.event_log if e["severity"] == "critical")
        warning = sum(1 for e in self.event_log if e["severity"] == "warning")
        return {"critical_ttc_events": critical, "warning_ttc_events": warning,
                "total_events": len(self.event_log)}

    def reset_event_log(self):
        self.event_log = []

    # ---------------------------------------------------------------
    # Global descriptive statistics (mean/min TTC, harsh brakes, speed,
    # collisions) -- separate from the reward computation, purely for
    # reporting real numbers (e.g. in a results table). Call once per
    # step with the FULL active vehicle list.
    # ---------------------------------------------------------------
    def log_global_stats(self, vehicle_ids, colliding_ids=None):
        colliding_ids = colliding_ids or set()
        self._stats_steps += 1
        for vid in vehicle_ids:
            if vid not in traci.vehicle.getIDList():
                continue
            speed = traci.vehicle.getSpeed(vid)
            self._stats_speed.append(speed)

            leader = traci.vehicle.getLeader(vid, self.comm_range)
            if leader is not None:
                leader_id, gap = leader
                leader_speed = traci.vehicle.getSpeed(leader_id)
                closing = speed - leader_speed
                if closing > 0.1 and gap >= 0:
                    self._stats_ttc.append(gap / closing)

            prev = self._stats_prev_speed.get(vid, speed)
            accel = (speed - prev) / 0.1
            if accel < HARSH_BRAKE_THRESHOLD:
                self._stats_harsh_brakes += 1
            self._stats_prev_speed[vid] = speed

        if colliding_ids:
            self._stats_collisions += 1  # count STEPS with >=1 collision, not vehicle-count

    def get_stats_summary(self):
        ttc_arr = np.array(self._stats_ttc) if self._stats_ttc else np.array([])
        speed_arr = np.array(self._stats_speed) if self._stats_speed else np.array([])
        return {
            "mean_ttc": float(np.mean(ttc_arr)) if len(ttc_arr) else None,
            "min_ttc": float(np.min(ttc_arr)) if len(ttc_arr) else None,
            "median_ttc": float(np.median(ttc_arr)) if len(ttc_arr) else None,
            "p10_ttc": float(np.percentile(ttc_arr, 10)) if len(ttc_arr) else None,
            "n_ttc_samples": int(len(ttc_arr)),
            "mean_speed": float(np.mean(speed_arr)) if len(speed_arr) else None,
            "min_speed": float(np.min(speed_arr)) if len(speed_arr) else None,
            "harsh_brake_events": self._stats_harsh_brakes,
            "collision_steps": self._stats_collisions,
            "steps_logged": self._stats_steps,
            "harsh_brakes_per_1000_steps": (self._stats_harsh_brakes / max(self._stats_steps, 1)) * 1000,
        }

    def reset_stats(self):
        self._stats_ttc = []
        self._stats_speed = []
        self._stats_harsh_brakes = 0
        self._stats_collisions = 0
        self._stats_steps = 0
        self._stats_prev_speed = {}

    # ---------------------------------------------------------------
    # reward computation
    # ---------------------------------------------------------------
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
                if closing > 0.1 and gap >= 0:
                    ttcs.append(gap / closing)
            # no-leader vehicles: excluded from TTC entirely (no sentinel padding)

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
        Returns (reward, breakdown). reward is a GATED, BOUNDED DELTA:
            raw    = [n/(n+c)] * (H_t - H_{t-1})
            reward = tanh(raw / REWARD_TANH_SCALE)   -- always in (-1, 1)
        """
        raw = self._raw_health(vehicle_ids, colliding_ids)
        if raw is None:
            return 0.0, None

        H = raw["H"]
        prev_H = self._prev_H.get(tx_id, H)
        delta = H - prev_H
        self._prev_H[tx_id] = H

        n = raw["n_vehicles"]
        gate = n / (n + SAMPLE_GATE_C)
        raw_reward = gate * delta

        reward = float(np.tanh(raw_reward / REWARD_TANH_SCALE))

        raw["reward"] = reward
        raw["raw_reward_prebound"] = raw_reward
        raw["gate"] = gate
        return reward, raw