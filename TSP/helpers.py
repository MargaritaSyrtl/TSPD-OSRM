import requests
from loguru import logger
import math


def get_coordinates(city_name):
    """Get lat and lon coordinates of the city"""
    url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json"
    response = requests.get(url, headers={"User-Agent": "geoapi"})
    if response.status_code == 200:
        data = response.json()
        # print(f"Coordinates of {city_name}: {data[0]['lat']}, {data[0]['lon']}")
        coord = (data[0]['lat'], data[0]['lon'])
        return coord
    else:
        raise Exception(f"No coordinates for {city_name} found.")


def get_city_names(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    response = requests.get(url, headers={"User-Agent": "geoapi"})
    if response.status_code == 200:
        data = response.json()
        # logger.debug(data)
        city = data.get("address", {}).get("city")
        if city:
            # logger.info(f"Found city: {city}")
            return city
        else:
            logger.error(f"City not found in response for ({lat},{lon})")
            return (lat, lon)
    else:
        logger.error(f"Error: {response.status_code}, {response.text}")
        return (lat, lon)


def euclidean_distance(coord1, coord2):
    """
    Calculates the approximate Euclidean distance between two coordinates (lat, lon) in meters.
    Uses simple approximation assuming flat Earth for small distances.
    """
    lat1, lon1 = map(float, coord1)
    lat2, lon2 = map(float, coord2)

    # Approximate conversions
    R = 6371000  # Earth radius in meters
    deg_to_rad = math.pi / 180

    dlat = (lat2 - lat1) * deg_to_rad
    dlon = (lon2 - lon1) * deg_to_rad
    lat1_rad = lat1 * deg_to_rad
    lat2_rad = lat2 * deg_to_rad

    # Approximate distance on Earth's surface (great-circle)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance