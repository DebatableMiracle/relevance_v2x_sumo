"""
Layer 2: Perception + broadcast fusion.
Builds theta = LOS-visible objects UNION broadcast-received objects,
for a given ego vehicle at the current sim step.

LOS occluders: other vehicles' bounding boxes AND (when loaded) building
polygons from the map's poly.xml — without buildings, Manhattan corners
are optically transparent and V2X has almost nothing to reveal.
"""
import math
import os
import xml.etree.ElementTree as ET
import traci
from shapely.geometry import LineString, Polygon, box


_BUILDING_POLYS = []


def load_building_polygons(path):
    """Parse SUMO poly.xml / obstacles.xml; shrink slightly so street
    corridors are not clipped by building corners."""
    global _BUILDING_POLYS
    _BUILDING_POLYS = []
    if not path or not os.path.isfile(path):
        return 0
    tree = ET.parse(path)
    for poly in tree.getroot().iter("poly"):
        shape = poly.get("shape") or ""
        pts = []
        for tok in shape.strip().split():
            if "," not in tok:
                continue
            x, y = tok.split(",")
            pts.append((float(x), float(y)))
        if len(pts) < 3:
            continue
        geom = Polygon(pts)
        shrunk = geom.buffer(-1.0)
        _BUILDING_POLYS.append(shrunk if not shrunk.is_empty else geom)
    return len(_BUILDING_POLYS)


def get_vehicle_state(vid):
    x, y = traci.vehicle.getPosition(vid)
    return {
        "id": vid,
        "pos": (x, y),
        "speed": traci.vehicle.getSpeed(vid),
        "angle": traci.vehicle.getAngle(vid),
        "lane": traci.vehicle.getLaneID(vid),
        "lane_index": traci.vehicle.getLaneIndex(vid),
        "edge": traci.vehicle.getRoadID(vid),
        "length": traci.vehicle.getLength(vid),
        "width": traci.vehicle.getWidth(vid),
    }


def get_vehicle_obstacle_polygons(exclude_ids=()):
    """Treat every other vehicle's bounding box as a potential occluder."""
    polys = {}
    for vid in traci.vehicle.getIDList():
        if vid in exclude_ids:
            continue
        x, y = traci.vehicle.getPosition(vid)
        l = traci.vehicle.getLength(vid)
        w = traci.vehicle.getWidth(vid)
        # axis-aligned approx box around vehicle (good enough at this fidelity;
        # rotate properly later if needed)
        polys[vid] = box(x - l / 2, y - w / 2, x + l / 2, y + w / 2)
    return polys


def has_line_of_sight(p1, p2, obstacle_polys, ignore_ids=(), extra_polys=None):
    sightline = LineString([p1, p2])
    for poly in (extra_polys or _BUILDING_POLYS):
        if sightline.intersects(poly):
            return False
    for oid, poly in obstacle_polys.items():
        if oid in ignore_ids:
            continue
        if sightline.intersects(poly):
            return False
    return True


def get_los_visible(ego_id, sensing_range=50.0, fov_deg=180, use_occlusion=True):
    """Range + FOV + (optional) occlusion-based visibility mask."""
    ego = get_vehicle_state(ego_id)
    ego_x, ego_y = ego["pos"]
    ego_angle = ego["angle"]

    obstacle_polys = get_vehicle_obstacle_polygons(exclude_ids={ego_id}) if use_occlusion else {}

    visible = {}
    for vid in traci.vehicle.getIDList():
        if vid == ego_id:
            continue
        state = get_vehicle_state(vid)
        x, y = state["pos"]
        dist = math.hypot(x - ego_x, y - ego_y)
        if dist > sensing_range:
            continue

        bearing = math.degrees(math.atan2(x - ego_x, y - ego_y))
        rel_angle = (bearing - ego_angle + 180) % 360 - 180
        if abs(rel_angle) > fov_deg / 2:
            continue

        if use_occlusion:
            # a target vehicle can't occlude itself
            if not has_line_of_sight((ego_x, ego_y), (x, y), obstacle_polys, ignore_ids={vid}):
                continue

        visible[vid] = state

    return visible


def get_known_objects(ego_id, broadcast_objects=None, sensing_range=50.0,
                       fov_deg=180, use_occlusion=True):
    """
    theta_t for ego_id = LOS-visible objects UNION broadcast_objects.
    broadcast_objects: dict {vid: state_dict}, e.g. output of get_vehicle_state()
                       for objects some Tx decided to include in its packet.
                       Pass {} or None for "no broadcast" baseline.
    """
    theta = get_los_visible(ego_id, sensing_range, fov_deg, use_occlusion)
    if broadcast_objects:
        for vid, state in broadcast_objects.items():
            if vid == ego_id:
                continue
            theta.setdefault(vid, state)  # LOS truth wins if already sensed directly
    return theta