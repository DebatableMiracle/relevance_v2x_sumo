"""
Continuous, non-resetting SUMO environment.
- Fringe-only spawn/despawn (entrances/exits at the network boundary)
- Configurable CAV_PENETRATION and BROADCAST_K (for experiment sweeps)
- Periodic checkpointing for long (multi-hour) runs
"""
import os, sys
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.path.append("/usr/share/sumo/tools")

import random
import traci
import sumolib
from perception import get_los_visible, get_known_objects, get_vehicle_state
from driving_policy import disable_native_safety, step_policy
from road_health import RoadHealthTracker


SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SIM_DIR, ".."))

MAP_REGISTRY = {
    "corridor": dict(config="corridor.sumocfg", net_file=os.path.join(PROJECT_DIR, "net/corridor/corridor.net.xml")),
    "circular": dict(config="circular.sumocfg", net_file=os.path.join(PROJECT_DIR, "net/circular/circular.net.xml")),
    "grid":     dict(config="grid.sumocfg",     net_file=os.path.join(PROJECT_DIR, "net/grid/grid.net.xml")),
    "manhattan": dict(config="manhattan.sumocfg",net_file=os.path.join(PROJECT_DIR,"net/manhattan/manhattan9x7_speed14.net.xml")),
}


class TrafficEnv:
    def __init__(self, map_name="grid", gui=False, spawn_rate=0.3, sense_range=50.0,
                 comm_range=175.0, broadcast_k=3, cav_penetration=0.5,
                 max_vehicles=25, checkpoint_every=2000, checkpoint_path="../logs/checkpoint.xml",
                 communication_enabled=True):
        map_cfg = MAP_REGISTRY[map_name]
        self.config = map_cfg["config"]
        self.net_file = map_cfg["net_file"]
        self.gui = gui
        self.spawn_rate = spawn_rate
        self.sense_range = sense_range
        self.comm_range = comm_range
        self.broadcast_k = broadcast_k
        self.cav_penetration = cav_penetration
        self.max_vehicles = max_vehicles
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path
        self.communication_enabled = communication_enabled

        self.health = RoadHealthTracker(comm_range=comm_range)
        self.cav_ids = set()
        self.all_managed_ids = set()
        self._route_pool = []
        self._global_step = 0
    def start(self):
        from harness import start as sumo_start
        sumo_start(gui=self.gui, config=self.config)
        self._route_pool = self._discover_fringe_routes()
        if not self._route_pool:
            raise RuntimeError("No fringe routes found — check net_file / grid geometry")
    def _discover_fringe_routes(self):
        net = sumolib.net.readNet(self.net_file)
        all_degrees = [len(n.getIncoming()) + len(n.getOutgoing()) for n in net.getNodes()]
        max_degree = max(all_degrees) if all_degrees else 0
        fringe_edges = [e for e in net.getEdges()
                         if (len(e.getFromNode().getIncoming()) + len(e.getFromNode().getOutgoing())) < max_degree]

        pool = []
        for from_e in fringe_edges:
            for to_e in fringe_edges:
                if from_e.getID() == to_e.getID():
                    continue
                try:
                    path, cost = net.getShortestPath(from_e, to_e)
                    if path:
                        edge_ids = [e.getID() for e in path]
                        rid = f"auto_{from_e.getID()}_{to_e.getID()}"
                        traci.route.add(rid, edge_ids)
                        pool.append(rid)
                except Exception:
                    continue
        print(f"[debug] fringe_edges found: {len(fringe_edges)}, routes successfully added: {len(pool)}")
        return pool
    def _maybe_spawn(self):
        if len(traci.vehicle.getIDList()) >= self.max_vehicles:
            return
        if random.random() > self.spawn_rate or not self._route_pool:
            return
        vid = f"veh_{self._global_step}_{random.randint(0, 9999)}"
        route = random.choice(self._route_pool)
        try:
            traci.vehicle.add(vid, route, typeID="car",
                               departPos="random_free", departSpeed="random")
        except traci.exceptions.TraCIException:
            return

        disable_native_safety(vid)      # EVERY vehicle uses our policy now
        self.all_managed_ids.add(vid)   # new: tracks every policy-driven vehicle

        if random.random() < self.cav_penetration:
            self.cav_ids.add(vid)       # only CAVs can RECEIVE broadcasts

    def register_cav(self, vid):
        disable_native_safety(vid)
        self.all_managed_ids.add(vid)
        self.cav_ids.add(vid)

    def _cleanup_despawned(self):
        active = set(traci.vehicle.getIDList())
        self.cav_ids &= active
        self.all_managed_ids &= active

    def get_observations(self):
        obs = {}
        for vid in self.cav_ids:
            if vid in traci.vehicle.getIDList():
                obs[vid] = get_los_visible(vid, self.sense_range, fov_deg=180, use_occlusion=True)
        return obs

    def _maybe_checkpoint(self):
        if self.checkpoint_every and self._global_step % self.checkpoint_every == 0:
            try:
                traci.simulation.saveState(self.checkpoint_path)
            except Exception as e:
                print(f"WARNING: checkpoint failed: {e}")

    def step(self, broadcast_actions):
        """
        broadcast_actions: {tx_id: [object_ids]}, each list truncated to broadcast_k.
        Returns (observations, rewards, info)
        """
        self._maybe_spawn()

        received = {vid: {} for vid in traci.vehicle.getIDList()}
        for tx_id, selected_ids in broadcast_actions.items():
            if tx_id not in traci.vehicle.getIDList():
                continue
            selected_ids = selected_ids[:self.broadcast_k]
            tx_x, tx_y = traci.vehicle.getPosition(tx_id)
            packet = {oid: get_vehicle_state(oid) for oid in selected_ids
                      if oid in traci.vehicle.getIDList()}
            for rx_id in traci.vehicle.getIDList():
                if rx_id == tx_id:
                    continue
                rx_x, rx_y = traci.vehicle.getPosition(rx_id)
                if ((tx_x - rx_x) ** 2 + (tx_y - rx_y) ** 2) ** 0.5 <= self.comm_range:
                    received[rx_id].update(packet)

        for vid in list(self.all_managed_ids):
            if vid not in traci.vehicle.getIDList():
                continue
            if vid in self.cav_ids and self.communication_enabled:
                theta = get_known_objects(vid, broadcast_objects=received.get(vid, {}),
                                           sensing_range=self.sense_range, use_occlusion=True)
            else:
                theta = get_known_objects(vid, broadcast_objects=None,
                                           sensing_range=self.sense_range, use_occlusion=True)
            step_policy(vid, theta)

        traci.simulationStep()
        self._cleanup_despawned()
        self._global_step += 1
        self.health.step_maintenance()
        self._maybe_checkpoint()

        colliding_ids = set(traci.simulation.getCollidingVehiclesIDList())
        rewards, info = {}, {}
        for tx_id in broadcast_actions:
            if tx_id not in traci.vehicle.getIDList():
                rewards[tx_id] = 0.0
                continue
            tx_x, tx_y = traci.vehicle.getPosition(tx_id)
            local_ids = [v for v in traci.vehicle.getIDList()
                         if ((tx_x - traci.vehicle.getPosition(v)[0]) ** 2 +
                             (tx_y - traci.vehicle.getPosition(v)[1]) ** 2) ** 0.5 <= self.comm_range]
            r, breakdown = self.health.compute_reward(tx_id, local_ids, colliding_ids)
            rewards[tx_id] = r
            info[tx_id] = breakdown

        return self.get_observations(), rewards, info