import math

def get_bounding_box(latitude: float, longitude: float, distance_km: float = 1.0) -> dict[str,float]:
    box_width = distance_km/(40075*math.cos(math.radians(latitude)))*360
    box_height = distance_km/(40075/360)
    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    return {"north":north,
            "east":east,
            "south":south,
            "west":west}
