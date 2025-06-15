tspd_google.py/tspd_osrm.py
The genetic algorithm implemented combines type-aware chromosome encoding, dynamic programming for efficient evaluation of truck and drone routes, and a set of specialized local search operators. 
The population consists of feasible and infeasible subpopulations, with dynamic penalties applied to guide the search. 
The algorithm integrates real-time traffic data (from Google Maps or OpenStreetMap) and generates visualizations for route validation.