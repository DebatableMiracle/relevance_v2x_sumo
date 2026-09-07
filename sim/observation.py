"""
Broadcast-selection observation.

Each candidate is encoded as: kinematics relative to Tx, plus *receiver-side
relevance* — how many nearby CAVs cannot see this object, and whether
pi_react would treat it as a same-path leader or a crossing conflict if
they were told.

That is the quantity the policy is supposed to optimize: not "is this
object near me", but "would broadcasting it change someone else's driving".
"""
import numpy as np
import traci

from driving_policy import classify_relevance, _wrap_deg

N_MAX = 20
# rel_x, rel_y, rel_speed, rel_heading, valid,
# n_blind, n_leader_if_told, n_conflict_if_told, min_ttc_if_told, obj_near_junction
FEAT_DIM = 10
OWN_DIM = 4  # speed, sin heading, cos heading, approaching_junction
OBS_DIM = N_MAX * FEAT_DIM + OWN_DIM
TTC_CAP = 10.0


def _approaching_junction(vid):
    try:
        road = traci.vehicle.getRoadID(vid)
        if road.startswith(":"):
            return 1.0
        lane = traci.vehicle.getLaneID(vid)
        rem = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(vid)
        links = traci.lane.getLinks(lane)
    except traci.exceptions.TraCIException:
        return 0.0
    goes_internal = any(
        isinstance(item, str) and item.startswith(":")
        for link in links
        for item in link
    )
    if not goes_internal:
        return 0.0
    return float(max(0.0, 1.0 - rem / 80.0))


def _obj_near_junction(state):
    edge = state.get("edge") or ""
    if edge.startswith(":"):
        return 1.0
    lane = state.get("lane") or ""
    if not lane:
        return 0.0
    try:
        links = traci.lane.getLinks(lane)
    except traci.exceptions.TraCIException:
        return 0.0
    return 1.0 if any(
        isinstance(item, str) and item.startswith(":")
        for link in links
        for item in link
    ) else 0.0


def encode_obs(tx_id, phi, los_by_cav=None, cav_ids=None, comm_range=175.0):
    """
    phi: Tx LOS dict {vid: state} — candidates the Tx can actually send.
    los_by_cav: {cav_id: {vid: state}} from TrafficEnv.get_observations().
    """
    tx_x, tx_y = traci.vehicle.getPosition(tx_id)
    tx_v = traci.vehicle.getSpeed(tx_id)
    tx_angle = traci.vehicle.getAngle(tx_id)

    candidates = sorted(
        phi.items(),
        key=lambda kv: (kv[1]["pos"][0] - tx_x) ** 2 + (kv[1]["pos"][1] - tx_y) ** 2,
    )[:N_MAX]
    candidate_ids = [cid for cid, _ in candidates]

    los_by_cav = los_by_cav or {}
    cav_ids = list(cav_ids or los_by_cav.keys())

    receivers = []
    for rid in cav_ids:
        if rid == tx_id or rid not in traci.vehicle.getIDList():
            continue
        rx, ry = traci.vehicle.getPosition(rid)
        if (rx - tx_x) ** 2 + (ry - tx_y) ** 2 > comm_range ** 2:
            continue
        receivers.append(rid)

    feat = np.zeros((N_MAX, FEAT_DIM), dtype=np.float32)
    for i, (oid, state) in enumerate(candidates):
        ox, oy = state["pos"]
        feat[i, 0] = ox - tx_x
        feat[i, 1] = oy - tx_y
        feat[i, 2] = state["speed"] - tx_v
        feat[i, 3] = _wrap_deg(state["angle"] - tx_angle)
        feat[i, 4] = 1.0

        n_blind = n_lead = n_conf = 0
        min_ttc = TTC_CAP
        for rid in receivers:
            if oid in los_by_cav.get(rid, {}):
                continue
            n_blind += 1
            role, gap = classify_relevance(rid, state)
            if role is None or gap is None:
                continue
            closing = traci.vehicle.getSpeed(rid) - state["speed"]
            ttc = gap / closing if closing > 0.1 else TTC_CAP
            ttc = min(max(ttc, 0.0), TTC_CAP)
            min_ttc = min(min_ttc, ttc)
            if role == "leader":
                n_lead += 1
            elif role == "conflict":
                n_conf += 1

        feat[i, 5] = n_blind / 8.0
        feat[i, 6] = n_lead / 4.0
        feat[i, 7] = n_conf / 4.0
        feat[i, 8] = min_ttc / TTC_CAP
        feat[i, 9] = _obj_near_junction(state)

    own_angle = np.radians(tx_angle)
    own = np.array(
        [tx_v, np.sin(own_angle), np.cos(own_angle), _approaching_junction(tx_id)],
        dtype=np.float32,
    )
    return np.concatenate([feat.flatten(), own]), candidate_ids
