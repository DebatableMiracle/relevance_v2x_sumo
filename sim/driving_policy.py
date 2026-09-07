"""
Layer 3 (fixed, NOT learned): pi_react.
IDM (longitudinal) + MOBIL (lateral) consuming an arbitrary known-object set (theta),
NOT SUMO's internal leader detection.
"""
import math
import traci

# ---- tunable defaults (SUMO vType-consistent) ----
V0 = 13.89        # desired speed, m/s
A_MAX = 2.6        # max accel
B_COMFORT = 4.5     # comfortable decel
T_HEADWAY = 1.5      # desired time gap, s
S0 = 2.5             # min gap, m
DELTA = 4             # IDM exponent
POLITENESS = 0.3       # MOBIL politeness factor
B_SAFE = 4.0            # MOBIL safety braking limit
DELTA_A_TH = 0.2         # MOBIL incentive threshold


def disable_native_safety(veh_id):
    """Must call once per vehicle: strips SUMO's built-in collision-avoidance
    override so our own IDM/MOBIL commands actually govern behavior."""
    traci.vehicle.setSpeedMode(veh_id, 0)
    traci.vehicle.setLaneChangeMode(veh_id, 0)


def idm_accel(v, v0, s, delta_v, a_max=A_MAX, b=B_COMFORT, T=T_HEADWAY, s0=S0, delta=DELTA):
    """s=None -> free flow (no known relevant leader)."""
    if s is None:
        return a_max * (1 - (v / v0) ** delta)
    s_star = s0 + max(0.0, v * T + (v * delta_v) / (2 * math.sqrt(a_max * b)))
    return a_max * (1 - (v / v0) ** delta - (s_star / max(s, 0.1)) ** 2)


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _state_edge(state):
    """Prefer explicit edge; fall back to lane-id prefix (e.g. A0B0_0 -> A0B0)."""
    edge = state.get("edge")
    if edge:
        return edge
    lane = state.get("lane") or ""
    if "_" in lane:
        return lane.rsplit("_", 1)[0]
    return lane or None


def _wrap_deg(delta):
    return (delta + 180.0) % 360.0 - 180.0


def _velocity(speed, angle_deg):
    """SUMO heading: 0 = north (+y), 90 = east (+x)."""
    rad = math.radians(angle_deg)
    return speed * math.sin(rad), speed * math.cos(rad)


def _reachable_lanes(lane_id, hops=3):
    """Outgoing lanes (including internals / next edges) within a few hops."""
    reached = set()
    frontier = [lane_id]
    for _ in range(hops):
        nxt = []
        for ln in frontier:
            try:
                links = traci.lane.getLinks(ln)
            except traci.exceptions.TraCIException:
                continue
            for link in links:
                for item in link:
                    if isinstance(item, str) and item and item not in reached:
                        reached.add(item)
                        nxt.append(item)
        frontier = nxt
    return reached


def _on_path_ahead(ego_edge, ego_lane, ego_lane_index, state, lane_filter, reachable):
    """Same-edge same-lane, or a successor lane of ego's current lane."""
    other_lane = state.get("lane") or ""
    other_edge = _state_edge(state)
    if other_edge == ego_edge:
        return state["lane_index"] == lane_filter
    return bool(other_lane) and other_lane in reachable


def find_leader(ego_id, known_objects, target_lane_index=None, max_ahead_angle=40, max_gap=100.0):
    """
    Closest known object ahead on ego's current edge+lane, or on a successor
    lane through the next junction (so a leader does not vanish at the stop line).
    Returns (obj_state, gap) or (None, None).
    """
    ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    ego_angle = traci.vehicle.getAngle(ego_id)
    ego_lane_index = traci.vehicle.getLaneIndex(ego_id)
    ego_edge = traci.vehicle.getRoadID(ego_id)
    ego_lane = traci.vehicle.getLaneID(ego_id)
    lane_filter = ego_lane_index if target_lane_index is None else target_lane_index
    reachable = _reachable_lanes(ego_lane)

    best, best_gap = None, None
    for vid, state in known_objects.items():
        if vid == ego_id:
            continue
        if not _on_path_ahead(ego_edge, ego_lane, ego_lane_index, state, lane_filter, reachable):
            continue
        x, y = state["pos"]
        bearing = math.degrees(math.atan2(x - ego_x, y - ego_y))
        rel_angle = _wrap_deg(bearing - ego_angle)
        if abs(rel_angle) > max_ahead_angle:
            continue
        gap = _dist((ego_x, ego_y), (x, y))
        if gap > max_gap:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = state, gap
    return best, best_gap


def find_follower(ego_id, known_objects, target_lane_index, max_gap=100.0):
    """Closest known object roughly behind ego in target lane on the same edge."""
    ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    ego_angle = traci.vehicle.getAngle(ego_id)
    ego_edge = traci.vehicle.getRoadID(ego_id)

    best, best_gap = None, None
    for vid, state in known_objects.items():
        if vid == ego_id:
            continue
        if state["lane_index"] != target_lane_index:
            continue
        if _state_edge(state) != ego_edge:
            continue
        x, y = state["pos"]
        bearing = math.degrees(math.atan2(x - ego_x, y - ego_y))
        rel_angle = (bearing - ego_angle + 180) % 360 - 180
        if abs(rel_angle) < 140:  # not roughly behind
            continue
        gap = _dist((ego_x, ego_y), (x, y))
        if gap > max_gap:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = state, gap
    return best, best_gap


def find_crossing_conflict(ego_id, known_objects, max_gap=60.0, max_ttc=4.0, miss_dist=8.0):
    """
    Closest known object on a roughly perpendicular heading whose linearly
    predicted paths come within miss_dist in the next max_ttc seconds.
    This is how a broadcast of an occluded cross-street vehicle can brake ego.
    Returns (obj_state, gap) or (None, None).
    """
    ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    ego_angle = traci.vehicle.getAngle(ego_id)
    ego_v = traci.vehicle.getSpeed(ego_id)
    evx, evy = _velocity(ego_v, ego_angle)

    best, best_gap = None, None
    for vid, state in known_objects.items():
        if vid == ego_id:
            continue
        heading_delta = abs(_wrap_deg(state["angle"] - ego_angle))
        if heading_delta < 50.0 or heading_delta > 130.0:
            continue
        x, y = state["pos"]
        gap = _dist((ego_x, ego_y), (x, y))
        if gap > max_gap:
            continue
        ovx, ovy = _velocity(state["speed"], state["angle"])
        rx, ry = x - ego_x, y - ego_y
        rvx, rvy = ovx - evx, ovy - evy
        rv = rx * rvx + ry * rvy
        vv = rvx * rvx + rvy * rvy
        if rv >= 0.0 or vv < 1e-6:
            continue
        t_star = -rv / vv
        if t_star > max_ttc:
            continue
        miss = math.hypot(rx + t_star * rvx, ry + t_star * rvy)
        if miss > miss_dist:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = state, gap
    return best, best_gap


def classify_relevance(ego_id, obj_state):
    """How pi_react would use this single object: 'leader', 'conflict', or None."""
    theta = {"_obj": obj_state}
    leader, gap = find_leader(ego_id, theta)
    if leader is not None:
        return "leader", gap
    conflict, gap = find_crossing_conflict(ego_id, theta)
    if conflict is not None:
        return "conflict", gap
    return None, None


def _accel_for(v, leader_state, gap):
    if leader_state is None:
        return idm_accel(v, V0, None, 0)
    delta_v = v - leader_state["speed"]
    return idm_accel(v, V0, gap, delta_v)


def evaluate_lane_change(ego_id, known_objects, target_lane_index):
    """
    MOBIL-style multi-object-aware lane change decision.
    known_objects: theta_t for ego — everything ego currently knows (LOS + broadcast).
    Returns True if ego should change to target_lane_index.
    """
    ego_v = traci.vehicle.getSpeed(ego_id)
    current_lane_index = traci.vehicle.getLaneIndex(ego_id)

    old_leader, old_gap = find_leader(ego_id, known_objects, current_lane_index)
    new_leader, new_gap = find_leader(ego_id, known_objects, target_lane_index)
    new_follower, nf_gap = find_follower(ego_id, known_objects, target_lane_index)

    a_current = _accel_for(ego_v, old_leader, old_gap)
    a_hypothetical = _accel_for(ego_v, new_leader, new_gap)

    # safety check: would the new follower be forced to brake dangerously?
    if new_follower is not None:
        follower_delta_v = new_follower["speed"] - ego_v  # follower closing on ego
        a_follower_after = idm_accel(new_follower["speed"], V0, nf_gap, follower_delta_v)
        if a_follower_after < -B_SAFE:
            return False

    gain = a_hypothetical - a_current
    return gain > DELTA_A_TH

def step_policy(ego_id, known_objects, allow_lane_change=True):
    ego_v = traci.vehicle.getSpeed(ego_id)
    current_lane_index = traci.vehicle.getLaneIndex(ego_id)

    leader, gap = find_leader(ego_id, known_objects, current_lane_index)
    conflict, cgap = find_crossing_conflict(ego_id, known_objects)
    accel = min(
        _accel_for(ego_v, leader, gap),
        _accel_for(ego_v, conflict, cgap),
    )
    new_speed = max(0.0, ego_v + accel * 0.1)
    traci.vehicle.setSpeed(ego_id, new_speed)

    if allow_lane_change:
        road_id = traci.vehicle.getRoadID(ego_id)
        if road_id and not road_id.startswith(":"):   # skip if not inserted yet / on internal junction lane
            num_lanes = traci.edge.getLaneNumber(road_id)
            for target in (current_lane_index - 1, current_lane_index + 1):
                if 0 <= target < num_lanes:
                    if evaluate_lane_change(ego_id, known_objects, target):
                        traci.vehicle.changeLaneRelative(ego_id, target - current_lane_index, duration=1.0)
                        break