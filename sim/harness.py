import traci
import numpy as np

SUMO_BINARY = "sumo"  # or "sumo-gui" for visual debugging
import os
#CONFIG = os.path.join(os.path.dirname(__file__), "..", "grid.sumocfg")
CONFIG = "../grid.sumocfg"

def start(gui=False, port=8813, config="grid.sumocfg"):
    binary = "sumo-gui" if gui else "sumo"
    cfg_path = os.path.join(os.path.dirname(__file__), "..", config)
    traci.start([binary, "-c", cfg_path, "--step-length", "0.1",
                 "--no-warnings", "true", "--collision.action", "none"],
                port=port)


def inject_object(veh_id, route_id, x, y, angle=0, edge_id="", lane_index=-1, type_id="DEFAULT_VEHTYPE"):
    """
    Adds a new vehicle and immediately places it at an exact (x,y) via moveToXY,
    bypassing normal route-based departure logic.
    """
    traci.vehicle.add(veh_id, routeID=route_id, typeID=type_id, departPos="0", departSpeed="0")
    traci.vehicle.moveToXY(veh_id, edgeID=edge_id, laneIndex=lane_index,
                            x=x, y=y, angle=angle, keepRoute=2)
    

def rollout(T_steps, track_ids=None):
    traj = {vid: [] for vid in (track_ids or [])}
    for _ in range(T_steps):
        traci.simulationStep()
        for vid in list(traj.keys()):
            if vid in traci.vehicle.getIDList():
                traj[vid].append(traci.vehicle.getPosition(vid))
    return traj