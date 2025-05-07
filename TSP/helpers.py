import requests
from loguru import logger


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
