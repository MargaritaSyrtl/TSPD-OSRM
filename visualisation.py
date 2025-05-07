import folium
from selenium import webdriver
import time
import os
from PIL import Image


def create_optimal_route_html(optimal_route, return_route, filename, cities, drone_nodes, drone_route):
    if not optimal_route and not return_route:
        print("No truck route provided!")
        return

    all_coords = optimal_route + return_route
    min_lat = min(lat for lat, lon in all_coords)
    max_lat = max(lat for lat, lon in all_coords)
    min_lon = min(lon for lat, lon in all_coords)
    max_lon = max(lon for lat, lon in all_coords)

    map_route = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=12)

    # Маршрут грузовика
    if optimal_route:
        folium.PolyLine(optimal_route, color="blue", weight=2.5, opacity=1).add_to(map_route)
    if return_route:
        folium.PolyLine(return_route, color="blue", weight=2.5, opacity=1).add_to(map_route)

    # Нумерация всех городов
    for i, (lat, lon) in enumerate(cities):
        label = f"{i}"
        folium.Marker(
            location=(lat, lon),
            popup=label,
            tooltip=label,
            icon=folium.DivIcon(
                html=f"""
                    <div style="font-size: 12px; color: white; background-color: blue;
                                border-radius: 50%; width: 24px; height: 24px;
                                display: flex; align-items: center; justify-content: center;">
                        {label}
                    </div>
                    """
            )
        ).add_to(map_route)

    # Рисуем полёты дрона: от launch → к drone → к land
    for drone_node, (launch_node, land_node) in drone_route.items():
        path = [launch_node, drone_node, land_node]
        folium.PolyLine(path, color="red", weight=2.5, opacity=0.8).add_to(map_route)

        # Пометка позиции дрона
        folium.Marker(
            location=(drone_node[0], drone_node[1]),
            popup="Drone",
            tooltip="Drone",
            icon=folium.Icon(color="red", icon="drone", prefix="fa")
        ).add_to(map_route)

    map_route.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    map_route.save(filename)
    # save_map_as_png(filename, filename.replace(".html", ".png"))
    # save_map_as_pdf(filename.replace(".html", ".png"), filename.replace(".html", ".pdf"))


def save_map_as_png(html_file, png_filename):
    """
    Convert the map from html to png.
    """

    abs_html_path = f"file://{os.path.abspath(html_file)}"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920x1080")

    with webdriver.Chrome(options=options) as browser:
        browser.get(abs_html_path)
        time.sleep(3)
        browser.save_screenshot(png_filename)


def save_map_as_pdf(png_filename, pdf_filename):
    """
    Convert the map from png to pdf.
    """
    image = Image.open(png_filename)
    image.convert("RGB").save(pdf_filename)


def get_route_from_ranking(tour, places):
    """
    Matches the rankings from the tour generated via MST/DFS
    with the destinations.
    """
    optimal_route = [places[stop] for stop in tour]
    return optimal_route
