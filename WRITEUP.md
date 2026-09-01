# Relevance-Based V2X Broadcast Selection — Setup Notes & Command Reference

Project: `~/development/sumoslammers/relevance_v2x`
Conda env: `sumo`

---

## 1. Environment Setup

```bash
mkdir -p ~/development/sumoslammers/relevance_v2x/{net,scenarios,sim,models,data,logs}
cd ~/development/sumoslammers/relevance_v2x
conda activate sumo
pip install traci sumolib numpy pandas lightgbm optuna shapely --break-system-packages
```

**Project structure:**
```
relevance_v2x/
├── net/            # .net.xml, .rou.xml network + route files
├── scenarios/       # generated scenario configs (future)
├── sim/             # Python TraCI harness + test scripts
├── models/           # trained LightGBM / RL models (future)
├── data/             # logged (state, ADE, H) tuples (future)
├── logs/             # run logs (future)
└── *.sumocfg          # config files, project root
```

---

## 2. Basic SUMO sanity checks

Run headless (no window, just simulates and exits):
```bash
sumo -c test.sumocfg --step-length 0.1
```

Run with GUI (visual, press ▶ to step):
```bash
sumo-gui -c test.sumocfg --step-length 0.1
```

`--no-step-log` suppresses per-step console spam (only use once you trust it's working — first runs should NOT use this flag so you see the step counter).

---

## 3. Network generation

### Simple single-edge net (hand-drawn in netedit) — `test.net.xml`
No junctions of interest — two `dead_end` nodes. Good only for single-vehicle Krauss car-following tests.

### 2x2 grid (corners only — NO real intersections, U-turns disabled)
```bash
netgenerate --grid --grid.number=2 --grid.length=100 \
  --default.lanenumber=2 --output-file=grid.net.xml
```
⚠️ Every junction here is a corner (2 neighbors only) — cannot exercise crossing/yielding logic.

### 3x3 grid (center node = real 4-way junction) — the one currently in use
```bash
cd ~/development/sumoslammers/relevance_v2x/net
netgenerate --grid --grid.number=3 --grid.length=100 \
  --default.lanenumber=2 --output-file=grid.net.xml
```
Center node `B1` connects to `A1` (north), `C1` (south), `B0` (west), `B2` (east) — genuine 4-way `priority` junction.

### Inspect a net's edges/nodes
```bash
python -c "import sumolib; net = sumolib.net.readNet('grid.net.xml'); print([e.getID() for e in net.getEdges()])"
python -c "import sumolib; net = sumolib.net.readNet('grid.net.xml'); print([j.getID() for j in net.getNodes()])"
```

---

## 4. Route file — `net/grid.rou.xml`

Two routes crossing through the center junction `B1`, staggered departures to force right-of-way negotiation:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>

    <!-- horizontal through B1: west -> east -->
    <route id="route_horizontal" edges="B0B1 B1B2"/>
    <!-- vertical through B1: north -> south -->
    <route id="route_vertical"   edges="A1B1 B1C1"/>

    <vehicle id="veh0" type="car" route="route_horizontal" depart="0"/>
    <vehicle id="veh1" type="car" route="route_vertical"   depart="1"/>
    <vehicle id="veh2" type="car" route="route_horizontal" depart="5"/>
    <vehicle id="veh3" type="car" route="route_vertical"   depart="5.5"/>
    <vehicle id="veh4" type="car" route="route_horizontal" depart="10"/>
    <vehicle id="veh5" type="car" route="route_vertical"   depart="10.2"/>

</routes>
```

⚠️ Known pitfall: routes like `"A0B0 B0A0"` are **U-turns** and will fail (`No connection between edge...`) — SUMO disables U-turns at simple junctions by default. Always chain edges through distinct legs, not back on themselves.

---

## 5. Config file — `grid.sumocfg`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="net/grid.net.xml"/>
        <route-files value="net/grid.rou.xml"/>
    </input>
    <random_number>
        <seed value="42"/>
    </random_number>
    <processing>
        <collision.action value="none"/>
    </processing>
</configuration>
```

Run it:
```bash
cd ~/development/sumoslammers/relevance_v2x
sumo-gui -c grid.sumocfg --step-length 0.1
```

---

## 6. `sim/harness.py` — TraCI control library

```python
import traci
import os

CONFIG = os.path.join(os.path.dirname(__file__), "..", "grid.sumocfg")

def start(gui=False, port=8813):
    binary = "sumo-gui" if gui else "sumo"
    traci.start([binary, "-c", CONFIG, "--step-length", "0.1",
                 "--no-warnings", "true", "--collision.action", "none"],
                port=port)

def rollout(T_steps, track_ids=None):
    traj = {vid: [] for vid in (track_ids or [])}
    for _ in range(T_steps):
        traci.simulationStep()
        for vid in list(traj.keys()):
            if vid in traci.vehicle.getIDList():
                traj[vid].append(traci.vehicle.getPosition(vid))
    return traj

def inject_object(veh_id, route_id, x, y, angle=0, edge_id="", lane_index=-1, type_id="DEFAULT_VEHTYPE"):
    """One-shot placement — SUMO's own dynamics take over immediately after."""
    traci.vehicle.add(veh_id, routeID=route_id, typeID=type_id, departPos="0", departSpeed="0")
    traci.vehicle.moveToXY(veh_id, edgeID=edge_id, laneIndex=lane_index,
                            x=x, y=y, angle=angle, keepRoute=2)

def inject_object_trajectory(veh_id, route_id, trajectory, type_id="DEFAULT_VEHTYPE"):
    """trajectory: list of (x, y, angle) tuples, one per sim step."""
    traci.vehicle.add(veh_id, routeID=route_id, typeID=type_id, departPos="0", departSpeed="0")
    traci.vehicle.moveToXY(veh_id, edgeID="", laneIndex=-1,
                            x=trajectory[0][0], y=trajectory[0][1],
                            angle=trajectory[0][2], keepRoute=2)

def step_object(veh_id, trajectory, step_idx):
    """Call every simulationStep() to keep object on its precomputed path."""
    if step_idx < len(trajectory):
        x, y, angle = trajectory[step_idx]
        traci.vehicle.moveToXY(veh_id, edgeID="", laneIndex=-1,
                                x=x, y=y, angle=angle, keepRoute=2)
```

---

## 7. Test scripts

### `sim/test_determinism.py` — validates save/load-state branching

```python
import sys, time
sys.path.insert(0, ".")
from harness import start, rollout
import traci

TRACK_ID = "t_0"  # or "veh0" depending on active .rou.xml

start(gui=False)
traci.simulationStep()
traci.simulation.saveState("snap.xml")
traj_a = rollout(20, track_ids=[TRACK_ID])
traci.close()

time.sleep(0.5)

start(gui=False)
traci.simulation.loadState("snap.xml")
traj_b = rollout(20, track_ids=[TRACK_ID])
traci.close()

assert len(traj_a[TRACK_ID]) > 0, "Vehicle never appeared — check depart time / ID"
assert traj_a == traj_b, "NON-DETERMINISTIC — check seed and .sumocfg config"
print(f"✅ Deterministic replay confirmed over {len(traj_a[TRACK_ID])} tracked steps")
```
Run: `cd sim && python test_determinism.py`

### `sim/test_injection.py` — validates one-shot object placement

```python
import sys, time
sys.path.insert(0, ".")
from harness import start, inject_object
import traci

start(gui=True)
traci.route.add("dummy_route", ["E0"])   # adjust edge id to active net
traci.simulationStep()
inject_object("obj_0", "dummy_route", x=-50, y=10, angle=90)

for step in range(200):
    traci.simulationStep()
    if "obj_0" in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition("obj_0")
        print(f"step {step}: obj_0 at {pos}")
    time.sleep(0.05)

traci.close()
```
Run: `python test_injection.py`

### `sim/test_trajectory_playback.py` — validates puppeteered (fully controlled) object motion

```python
import sys, time
sys.path.insert(0, ".")
from harness import start, inject_object_trajectory, step_object
import traci

start(gui=True)
traci.route.add("dummy_route", ["E0"])
traci.simulationStep()

trajectory = [(-50 + 0.5*i, 10.0, 90) for i in range(20)]
inject_object_trajectory("obj_0", "dummy_route", trajectory)

for step in range(20):
    step_object("obj_0", trajectory, step)
    traci.simulationStep()
    if "obj_0" in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition("obj_0")
        print(f"step {step}: obj_0 at {pos}  (target: {trajectory[step][:2]})")
    time.sleep(0.05)

traci.close()
```
Run: `python test_trajectory_playback.py`
Success criteria: actual position ≈ target position every step (no drift, unlike `test_injection.py`).

---

## 8. Useful TraCI calls reference (used or will be used)

| Call | Purpose |
|---|---|
| `traci.vehicle.getIDList()` | all currently active vehicle IDs |
| `traci.vehicle.getPosition(id)` | (x, y) ground truth |
| `traci.vehicle.getSpeed(id)` | current speed, m/s |
| `traci.vehicle.getAngle(id)` | heading, degrees |
| `traci.vehicle.add(...)` | insert a new vehicle mid-sim |
| `traci.vehicle.moveToXY(...)` | force-place a vehicle at exact XY (bypasses safety checks with `keepRoute=2`) |
| `traci.vehicle.slowDown(id, speed, duration)` | commanded deceleration override |
| `traci.vehicle.setSpeedMode(id, bitmask)` | **disable SUMO's built-in safety checks** — needed to make collisions/near-misses actually possible for "blind" vehicles |
| `traci.vehicle.setLaneChangeMode(id, bitmask)` | same, for lane-change safety |
| `traci.simulation.saveState(path)` | snapshot sim state to disk |
| `traci.simulation.loadState(path)` | restore sim state — enables branching for counterfactual comparisons |
| `traci.route.add(id, edges)` | define a new route dynamically |
| `traci.polygon.getIDList()` / `getShape()` | building/obstacle polygons (only present if net has them) |

---

## 9. Known pitfalls encountered so far

1. **Wrong vehicle ID in track_ids** — assert passes vacuously on empty-list comparison. Always verify the ID exists in the `.rou.xml` first.
2. **"Retrying in 1 seconds" on TraCI start** — port not released from previous run. Add `time.sleep(0.5–1)` between `traci.close()` and next `traci.start()`, and/or pass explicit `port=` per session once running parallel instances.
3. **U-turn routes fail** — `"A0B0 B0A0"` style routes are invalid; SUMO disables U-turns by default. Chain through separate legs instead.
4. **2x2 grids have no real junctions** — every node is a corner (2 neighbors). Need 3x3+ for a genuine 4-way crossing.
5. **`moveToXY` bypasses safety checks entirely** (`keepRoute=2`) — no automatic collision prevention on injected vehicles; this is by design but must be handled explicitly.
6. **"Simulation ended" dialog in GUI is NOT a crash** — check the "Reason" field. `"TraCI requested termination"` = your script's loop just finished normally and called `traci.close()`.
7. **Krauss/junction logic is collision-avoidant by construction and omniscient** — it always sees true ground truth regardless of any perception mask you build on top. To make collisions/near-misses actually possible for "blind" (non-broadcast-informed) vehicles, you MUST explicitly disable relevant `setSpeedMode` bits — otherwise your broadcast mechanism will show no measurable safety benefit, since nothing unsafe can happen in the first place.

---

## 10. Where the project stands / what's next (not yet built)

- ✅ Harness: start/stop, deterministic save/load-state branching, object injection (one-shot and puppeteered)
- ✅ Multi-vehicle scenario with a real 4-way junction, crossing routes
- ⬜ **Layer 2 — perception mask**: range + FOV cone (`get_visible_vehicles`), plus occlusion via raycasting (`shapely.geometry.LineString.intersects`) against building/vehicle-bounding-box obstacles
- ⬜ **Layer 3 — broadcast-triggered reaction override**: hardcoded rule first (e.g., `slowDown` when informed of an unseen hazard), applied only to vehicles with default safety selectively disabled via `setSpeedMode`
- ⬜ Road-health metric `H(S)`: built from continuous surrogate safety measures (TTC, PET, harsh-braking events, min-gap) rather than raw collision count (too sparse a signal)
- ⬜ Greedy marginal-gain object selection (`Δ(o|S) = H(S∪{o}) − H(S)`), with lazy-greedy optimization for efficiency
- ⬜ Offline dataset generation (parallelized, `libsumo` instead of `traci` for speed) → LightGBM surrogate for `H(S)`/`Δ(o|S)`
- ⬜ RL policy trained against the LightGBM surrogate (not live SUMO) via Stable-Baselines3 + Gymnasium custom `Env` wrapper — small MLP or attention-based selector over variable-count candidate objects

## Not needed for current scope (per advisor guidance / analysis)
- ❌ CARLA (SUMO's structured ground-truth state is sufficient; CARLA's GPU cost buys camera/LiDAR rendering fidelity you don't need)
- ❌ ns-3 / OMNeT++ / Veins radio-layer co-simulation (explicitly deferred by advisor — future work item)
- ❌ Autoware / LiDAR-based planning policy (paper itself bypasses Autoware's perception stack and feeds structured object state directly to planning — same abstraction level as your TraCI calls)