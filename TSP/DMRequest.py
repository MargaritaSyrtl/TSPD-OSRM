import requests
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from itertools import combinations


class DMRequest(object):
	def __init__(self, places):
		self.base_url = 'http://localhost:5000/table/v1/driving'
		self.places = places

	def __get_distances(self):
		"""
		Sends GET request to the OSRM
		"""
		# collect coordinates of points in OSRM format: (longitude, latitude)
		coords = ';'.join([f"{lon},{lat}" for lat, lon in self.places])
		try:
			response = requests.get(f"{self.base_url}/{coords}?annotations=distance,duration").json()
			# print(response)
			return response
		except requests.exceptions.RequestException as e:
			print(f"Request failed: {e}")
			return None

	def get_geometry_for_route(self, waypoint1, waypoint2):
		""" Get the geometry of a route between two points via the OSRM API
			If the route is not available, None is returned."""
		coords = f"{waypoint1[1]},{waypoint1[0]};{waypoint2[1]},{waypoint2[0]}"
		url = f"http://localhost:5000/route/v1/driving/{coords}?overview=full&geometries=geojson"

		try:
			response = requests.get(url).json()
			# print(response)
			if response.get("code") == "Ok" and "routes" in response:
				geometry = response['routes'][0]['geometry']['coordinates']
				return [(lat, lon) for lon, lat in geometry]  # flip the coordinates
			else:
				print(f"OSRM could not find route geometry between {waypoint1} and {waypoint2}.")
		except Exception as exc:
			print(f"Could not find geometry between {waypoint1} and {waypoint2}: {exc}")
		return None

	def get_response_data_mst(self):
		response = self.__get_distances()
		data = {
			'distance': response['distances'],
			'duration': response['durations']
		}
		return data

	def get_response_data_ga(self):
		"""
			Send GET request to OSRM
			Using a format for the genetic algorithm
			Parse API response for distance and duration values
		"""
		data = {'waypoints_distances': {}, 'waypoints_durations': {}, 'waypoints_geometries': {}}  # = frozenset({(lat1, lon1), (lat2, lon2)}) -> distance

		for (waypoint1, waypoint2) in combinations(self.places, 2):
			try:
				# coordinate string in the format "lon,lat;lon,lat"
				coords = f"{waypoint1[1]},{waypoint1[0]};{waypoint2[1]},{waypoint2[0]}"
				# resp = requests.get(f"{self.base_url}/{coords}?annotations=distance,duration").json()
				url = f"http://localhost:5000/route/v1/driving/{coords}?overview=full&geometries=geojson"
				resp = requests.get(url).json()
				if resp.get("code") == "Ok" and "routes" in resp:
					dist = resp['routes'][0]['distance']
					dur = resp['routes'][0]['duration']
					geometry = resp['routes'][0]['geometry']['coordinates']
					data['waypoints_distances'][frozenset([waypoint1, waypoint2])] = dist
					data['waypoints_durations'][frozenset([waypoint1, waypoint2])] = dur
					data['waypoints_geometries'][frozenset([waypoint1, waypoint2])] = [
						(lat, lon) for lon, lat in geometry
					]
				# if resp.get("code") == "Ok" and "distances" in resp:
					# distances[0][1] — distance from the first point to the second
				# 	dist = resp['distances'][0][1]
				# 	dur = resp['durations'][0][1]
				# 	data['waypoints_distances'][frozenset([waypoint1, waypoint2])] = dist
				# 	data['waypoints_durations'][frozenset([waypoint1, waypoint2])] = dur
				else:
					print(f"OSRM could not find route from {waypoint1} to {waypoint2}.")
			except Exception as exc:
				print(f"Could not find route from {waypoint1} to {waypoint2}: {exc}")
		# print(data)
		return data

	def set_params(self, custom_params=None):
		if custom_params:
			my_params = custom_params
		else:
			my_params = {
					'origins': '|'.join(self.places),
					'destinations': '|'.join(self.places),
				}
		return my_params

	@staticmethod
	def compute_route_metrics_mst_dfs_tsp(route, matrix):
		""" Calculates the total route's length or duration. :param route: list of route's points [(lat1, lon1), (lat2, lon2), ...]. :param matrix: Matrix of distances or durations (list of lists). """

		total_metric = 0.0
		# dict to convert coordinates to indices
		places_index = {place: idx for idx, place in enumerate(route)}

		# iteration over pairs of waypoints
		for i in range(len(route) - 1):
			point1 = route[i]
			point2 = route[i + 1]
			idx1, idx2 = places_index[point1], places_index[point2]
			total_metric += matrix[idx1][idx2]

		# route closure: from the last point to the first
		idx1, idx2 = places_index[route[-1]], places_index[route[0]]
		total_metric += matrix[idx1][idx2]
		return total_metric

	@staticmethod
	def compute_route_metrics_ga(route, waypoints_data):
		""" Calculates the total route's length or duration. :param route: list of route's points [(lat1, lon1), (lat2, lon2), ...]. :param waypoints_data: dict with distances or durations between points. """

		total_metric = 0.0
		for i in range(len(route) - 1):
			point1 = route[i]
			point2 = route[i + 1]
			metric = waypoints_data.get(frozenset([point1, point2]))
			if metric is not None:
				total_metric += metric
			else:
				print(f"No data found between {point1} and {point2}")

		#  route closure: from the last point to the first
		metric = waypoints_data.get(frozenset([route[-1], route[0]]))
		if metric is not None:
			total_metric += metric
		else:
			print(f"No data found between {route[-1]} and {route[0]}")
		return total_metric

	@staticmethod
	def print_distance_matrix_table(data, places):
		"""
		Creates a distance matrix for TSP computation by the Concorde Solver.
		This will be used when we submit a job to the NEOS server.
		"""
		distances = data['distance']

		df = pd.DataFrame(distances, columns=places, index=places)
		dm = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
		return dm

	@staticmethod
	def build_full_distance_matrix(upper_triangle):
		"""
		Creates a symmetric matrix from an upper triangle
		"""
		out = upper_triangle.T + upper_triangle
		# the diagonal remains unchanged
		np.fill_diagonal(out, np.diag(upper_triangle))
		return out

	@staticmethod
	def build_upper_triangle_matrix(data):
		"""
		Upper triangle matrix format in case we need to use for solving TSP
		"""
		# data['distance'] is a 2D array (N×N) from OSRM
		lt = np.triu(data['distance'])  # take the upper triangular part
		return lt

	@staticmethod
	def build_lower_triangle_matrix(data):
		"""
		Lower triangle matrix format in case we need to use for solving TSP
		"""
		lt = np.tril(data['distance'])
		return lt


