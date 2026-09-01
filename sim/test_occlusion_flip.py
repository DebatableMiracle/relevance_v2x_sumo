import subprocess
subprocess.run(["pkill", "-9", "sumo"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)

import sys, time, math
sys.path.insert(0, ".")
from harness import start
from perception import get_known_objects, get_vehicle_state
from driving_policy import disable_native_safety, step_policy
import traci

time.sleep(0.5)


def point_along_lane(lane_id, dist):
    """Interpolate a point + heading `dist` meters along a lane (straight-lane assumption)."""
    shape = traci.lane.getShape(lane_id)
    (x0, y0), (x1, y1) = shape[0], shape[-1]
    length = traci.lane.getLength(lane_id)
    frac = dist / length
    x = x0 + frac * (x1 - x0)
    y = y0 + frac * (y1 - y0)
    angle = math.degrees(math.atan2(x1 - x0, y1 - y0))
    return x, y, angle


def place(vid, lane_id, dist, speed):
    x, y, angle = point_along_lane(lane_id, dist)
    traci.vehicle.moveToXY(vid, edgeID="", laneIndex=-1, x=x, y=y, angle=angle, keepRoute=2)
    traci.vehicle.setSpeed(vid, speed)


def setup_scenario():
    start(gui=True, config="test_occlusion.sumocfg")
    traci.route.add("straight", ["B0B1", "B1B2"])

    traci.vehicle.add("ego", "straight", typeID="car", departPos="0",  departSpeed="0")
    traci.vehicle.add("X",   "straight", typeID="car", departPos="20", departSpeed="0")
    traci.vehicle.add("Y",   "straight", typeID="car", departPos="40", departSpeed="0")

    for _ in range(20):
        traci.simulationStep()
        if all(v in traci.vehicle.getIDList() for v in ["ego", "X", "Y"]):
            break
    else:
        print("WARNING: not all vehicles inserted:", traci.vehicle.getIDList())

    place("ego", "B0B1_0", 0, 10)
    place("X",   "B0B1_0", 15, 8)
    place("Y",   "B0B1_1", 25, 8)
    traci.simulationStep() 

    for vid in ["ego", "X", "Y"]:
        disable_native_safety(vid)

    print("Startup check:",
          "ego lane =", traci.vehicle.getLaneIndex("ego"),
          "| X lane =", traci.vehicle.getLaneIndex("X"),
          "| Y lane =", traci.vehicle.getLaneIndex("Y"))


def run(broadcast_y=False, steps=40):
    setup_scenario()
    lane_changed = False
    for step in range(steps):
        active = traci.vehicle.getIDList()
        if "ego" not in active:
            print(f"step {step}: ego left the simulation, stopping early")
            break

        for vid, spd in [("X", 8), ("Y", 8)]:
            if vid in active:
                traci.vehicle.setSpeed(vid, spd)

        y_state = get_vehicle_state("Y") if "Y" in active else None
        broadcast = {"Y": y_state} if (broadcast_y and y_state) else {}
        theta_ego = get_known_objects("ego", broadcast_objects=broadcast,
                                       sensing_range=50, fov_deg=160, use_occlusion=True)

        was_lane = traci.vehicle.getLaneIndex("ego")
        step_policy("ego", theta_ego)
        traci.simulationStep()
        now_lane = traci.vehicle.getLaneIndex("ego")

        print(f"step {step}: ego knows {list(theta_ego.keys())}, lane={now_lane}")
        if now_lane != was_lane:
            lane_changed = True
        time.sleep(0.03)

    traci.close()
    return lane_changed


print("\n--- Run A: LOS only (Y occluded, not broadcast) ---")
changed_a = run(broadcast_y=False)
time.sleep(1)
print("\n--- Run B: LOS + broadcast(Y) ---")
changed_b = run(broadcast_y=True)

print(f"\nRun A (no broadcast) lane change happened: {changed_a}")
print(f"Run B (with broadcast) lane change happened: {changed_b}")