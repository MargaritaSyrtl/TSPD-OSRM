import math
from loguru import logger
from itertools import combinations
import folium
import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
from scipy.spatial.distance import cdist
import random


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
    return round(distance, 1)


def join_algorithm(chromosome, truck_time, drone_time, drone_range):
    action_trace = {}  # i -> best action ('MT' or 'LL')
    n = len(chromosome)
    logger.debug(chromosome)
    full_seq = [0] + [abs(x) for x in chromosome] + [n + 1]  # virtual end node
    logger.debug(f"full_seq {full_seq}")

    node_types = {abs(x): 'truck' if x >= 0 else 'drone' for x in chromosome}
    logger.debug(f"nodes_types {node_types}")
    all_nodes_to_serve = set(i for i in range(1, n + 1))  # всё, кроме депо
    logger.debug(all_nodes_to_serve)
    truck_nodes = [0]
    drone_nodes = []
    for i in range(1, n + 1):  # n+1 virtual node -> 3=0
        if node_types.get(i) == "truck":
            truck_nodes.append(i)
        if node_types.get(i) == "drone":
            drone_nodes.append(i)
    truck_nodes.append(n + 1)  # add virtual end node e.g. (0')=(3)
    logger.debug(f"truck_nodes {truck_nodes}")
    logger.debug(f"drone_nodes {drone_nodes}")

    #### DP
    # C[i] — minimum time from truck node i to the end
    C = {}
    C[n + 1] = 0  # virtual end of the route
    C[truck_nodes[-1]] = 0  # C(end) = 0
    logger.debug(f"C init {C}")

    logger.debug(len(truck_nodes) - 1)
    logger.debug(range(len(truck_nodes) - 1))
    best_mt = 0
    best_ll = 0
    all_options = []
    # move truck from i to j
    for idx in reversed(
            range(len(truck_nodes) - 1)):  # from the end of the route to the beginning including the depot at the end
        logger.debug(f"idx {idx}")
        i = truck_nodes[idx]  # the truck can start moving from truck_nodes[idx]
        logger.debug(f"truck_nodes[{idx}]={i}")
        CMT = float('inf')  # Move Truck
        CLL = float('inf')  # Launch and Land

        # MT
        logger.info(f"Start MT")
        CMT_best = float('inf')

        # all truck nodes after i
        for j in truck_nodes[idx + 1:]:
            served_mt = {}
            logger.debug(f"truck_nodes[{idx + 1:}]={j}")
            if full_seq[j] == full_seq[-1] and any(
                    full_seq[k] in node_types and node_types[full_seq[k]] == 'truck'
                    for k in range(i + 1, j)
            ):
                continue  # skip jump to the end ???

            t_time = truck_time[full_seq[i]][full_seq[j]]
            logger.debug(f"time between {i} and {j} = {t_time}")
            logger.debug(f"C[{j}]={C.get(j, float('inf'))}")  # minimum time from j to the end of the route
            CMT = min(CMT, t_time + C.get(j, float('inf')))
            logger.debug(f"C_MT= {CMT}")
            if CMT < CMT_best:
                CMT_best = CMT
                best_mt = ('MT', full_seq[i], full_seq[j], CMT)
                logger.debug(f"best mt={best_mt}")  # ('MT', 0, 2, 14.25)
                all_options.append((CMT, ('MT', full_seq[i], full_seq[j], CMT)))
            served_mt[j] = CMT
            logger.debug(f"served_mt={served_mt}")

        # LL
        logger.info(f"Start LL")

        # the first drone node after i
        for d in range(i + 1, len(full_seq) - 1):
            served_ll = {}
            logger.debug(f"d={d}")
            if node_types.get(d, '') != 'drone':
                continue

            deliver = full_seq[d]
            logger.debug(f"deliver {deliver}")
            CLL_best = float('inf')
            # land nodes - all truck stations after delivery
            for k in truck_nodes:
                if k <= d:
                    continue
                land = full_seq[k]
                logger.debug(f"land {land}")

                logger.debug(drone_time[full_seq[i]][deliver])  # launch->d
                logger.debug(drone_time[deliver][land])  # d->land
                d_flight = drone_time[full_seq[i]][deliver] + drone_time[deliver][land]  # full time
                logger.debug(f"d flight {d_flight}")
                if full_seq[j] == full_seq[-1]:  # 0=(0')
                    CLL = d_flight
                    # if there are unvisited truck nodes between i and j
                else:
                    t_drive = truck_time[full_seq[i]][land]  # launch->land on truck!
                    logger.debug(f"t drive {t_drive}")

                    if d_flight <= drone_range:
                        logger.debug(f"C[{k}]={C.get(k)}")
                        total = max(d_flight, t_drive) + C.get(k, float('inf'))
                        logger.debug(f"total={total}")
                        CLL = min(CLL, total)
                        logger.debug(f"CLL={CLL}")

                if CLL < CLL_best:
                    CLL_best = CLL
                    best_ll = ('LL', full_seq[i], deliver, land, CLL)
                    logger.debug(f"best ll={best_ll}")
                    all_options.append((CLL, ('LL', full_seq[i], deliver, land, CLL)))

            served_ll[d] = CLL_best
            logger.debug(f"served_ll={served_ll}")
            # break  # только первый drone после i, как в статье??
            continue

        logger.info(f"best values: {best_mt}, {best_ll}")  # ('MT', 0, 2, 28.5), ('LL', 0, 1, 3, 356.98)

        # C[i] = min(CMT, CLL)
        logger.debug(f"compare: {CMT} vs {CLL}")
        if CMT <= CLL:
            C[i] = CMT
            res = best_mt
        else:
            C[i] = CLL
            res = best_ll
        action_trace[i] = res
        # logger.debug(f"res: {res}")
        # logger.debug(f"C[{i}] at the end: {C[i]}")
        logger.debug(f"all_options: {all_options}")

    # calculate the optimal route
    required_nodes = set(abs(x) for x in chromosome)
    optimal_route = None
    best_makespan = float('inf')

    for r in range(1, len(all_options) + 1):
        for combo in combinations(all_options, r):
            actions = [a for _, a in combo]  # actions without time
            served = set()  # served nodes
            ends_at_depot = False

            for a in actions:
                if a[0] == 'MT':
                    served.add(a[2])
                    if a[2] == n + 1:  # end at depot
                        ends_at_depot = True
                elif a[0] == 'LL':
                    served.add(a[2])

            if served >= required_nodes and ends_at_depot:
                # ((14.25,   ('MT', 2,    3, 14.25)),
                #  (28.5,    ('MT', 0,    2, 28.5)),
                #  (364.095, ('LL', 0, 1, 2, 364.095)),
                #  (356.98,  ('LL', 0, 1, 3, 356.98)))
                makespan = max(t for t, _ in combo)
                if makespan < best_makespan:
                    optimal_route = actions
                    best_makespan = makespan
    logger.debug(f"optimal route: {optimal_route}")
    logger.debug(f"best_makespan: {best_makespan}")
    return optimal_route, best_makespan


def visualize_route(places, route):
    # depo
    center = places[0]
    m = folium.Map(location=center, zoom_start=14)

    # markers
    for idx, (lat, lon) in enumerate(places):
        folium.Marker([lat, lon], tooltip=f"Point {idx}").add_to(m)

    # visualise
    for action in route:
        if action[0] == 'MT':
            _, start, end, _ = action
            points = [places[start], places[end]]
            folium.PolyLine(points, color="blue", weight=2.5, tooltip=f"Truck {start}->{end}").add_to(m)
        elif action[0] == 'LL':
            _, launch, deliver, land, _ = action
            drone_points = [places[launch], places[deliver], places[land]]
            folium.PolyLine(drone_points, color="green", weight=2.5, dash_array='5,10',
                            tooltip=f"Drone {launch}->{deliver}->{land}").add_to(m)
    m.save("route_map.html")
    return True


def generate_initial_population(n, size):
    base = list(range(1, n + 1))
    logger.debug(base)
    population = []
    for _ in range(size):
        chrom = random.sample(base, n)
        chrom = [g if random.random() > 0.5 else -g for g in chrom]
        population.append(chrom)
    logger.debug(f"init population: {population}")
    return population


def evaluate(chromosome, truck_time, drone_time, drone_range):

    route, cost = join_algorithm(chromosome, truck_time, drone_time, drone_range)
    logger.debug(f"Route: {route}")
    logger.debug(f"cost: {cost}")
    return cost


def genetic_algorithm(places, generations=1, population_size=2, truck_speed=10, drone_range=float('inf')):
    drone_speed = 2 * truck_speed
    feasible = {}
    infeasible_1 = {}
    infeasible_2 = {}

    # TSP
    #route, cost = solve_tsp_local_search(truck_time)
    #logger.debug(f"Route for TSP: {route}")
    #logger.debug(f"Time for TSP: {cost}")
    # TSPD
    #tspd = [1, -3, 2]

    # init
    n = len(places) - 1
    population = generate_initial_population(n, population_size)

    # for join algo
    places.append(places[0])
    n = len(places) - 1  # n=3 (без учёта 0′)
    logger.info(f"For {n} points.")
    logger.info(f"{places}")
    # init time matrix
    truck_time_matrix = [[0] * (n + 2) for _ in range(n + 2)]
    drone_time_matrix = [[0] * (n + 2) for _ in range(n + 2)]
    for i in range(n + 1):
        for j in range(n + 1):
            dist = euclidean_distance(places[i], places[j])
            truck_time_matrix[i][j] = dist / truck_speed
            drone_time_matrix[i][j] = dist / drone_speed

    best_chrom = []
    best_cost = []
    for g in range(generations):
        # evaluate fitness of all chromosomes
        fitnesses = [evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range) for chrom in population]
        logger.info(fitnesses)






if __name__ == "__main__":
    drone_speed = 20  # m/s
    truck_speed = 10
    drone_range = float('inf')

    places = [(50.149, 8.666),  # idx=0
              (50.145, 8.616),  # idx=1
              (50.147, 8.668),  # idx=2
              (50.146, 8.777),
              # (50.155, 7.777)
              ]
    n = len(places)  # n=3 (без учёта 0′)
    logger.info(f"For {n} points.")
    genetic_algorithm(places)

    # places.append((places[0][0] + 0.00001, places[0][1]))  # move depo by 1 meter

    #generations = 1  # number of iterations
    #population_size = 2  # number of agents
    #n = len(places) - 1  # n=3 (без учёта 0′)
    #logger.info(f"For {n} points.")
    #logger.info(f"{places}")
    # init time matrix
    #truck_time_matrix = [[0] * (n + 2) for _ in range(n + 2)]
    #drone_time_matrix = [[0] * (n + 2) for _ in range(n + 2)]

    #for i in range(n + 1):
    #    for j in range(n + 1):
    #        dist = euclidean_distance(places[i], places[j])
    #        truck_time_matrix[i][j] = dist / truck_speed  # in sec
    #        drone_time_matrix[i][j] = dist / drone_speed  # in sec

    #logger.debug(f"truck_time_matrix {truck_time_matrix}")
    #logger.debug(f"drone_time_matrix {drone_time_matrix}")

    chromosome1 = [1, 2, 3]
    chromosome2 = [-1, 2, 3]
    chromosome3 = [1, -2, 3]
    chromosome4 = [1, 2, -3]
    chromosome5 = [-1, -2, 3]
    chromosome6 = [-1, 2, -3]
    chromosome7 = [1, -2, -3]
    # chromosome8 = [-1, -2, -3]  # need to be tested


    # route_with_return = route + [route[0]]

    # TSPD
    #updated_route, drone_nodes, drone_ops = assign_best_drone_operations_single_drone(
    #    route,
    #    truck_time_matrix,
    #    drone_time_matrix,
    #    drone_range
    #)
    #logger.debug(f"Truck route for TSPD: {updated_route}")
    #logger.debug(f"Drone nodes: {drone_nodes}")
    #used_drone_nodes = set()
    #used_drone_ops = []
    #for op in drone_ops:
    #    d = op["drone"]
    #    if d in drone_nodes and d not in used_drone_nodes:
    #        used_drone_ops.append(op)
    #        used_drone_nodes.add(d)
    #for op in used_drone_ops:
    #    logger.debug(
    #        f"USED: Start: {op['start']}, drone node: {op['drone']}, end: {op['end']}, time: {round(op['duration'], 2)} sec")

    # todo
    route_tspd = [('MT', 3, 4, 0), ('MT', 2, 3, 776.78), ('MT', 0, 2, 803.1899999999999), ('LL', 0, 1, 2, 365.13)]


    #places.append(places[0])  # depo needed for join ! todo
    #optimal_route, makespan = join_algorithm(chromosome2, truck_time_matrix, drone_time_matrix, drone_range=float('inf'))
    #visualize_route(places, optimal_route)

