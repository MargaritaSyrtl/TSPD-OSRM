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

                    # does not discard LL actions even if they are out of bounds
                    if d_flight > drone_range:
                        penalty = (d_flight - drone_range) * 2  # (for example) todo
                        d_flight += penalty

                    # if d_flight <= drone_range:
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
    if optimal_route is None:
        logger.warning("No feasible route found in join_algorithm — returning empty route with high cost.")
        return [], float('inf')

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
    """Compute fitness of the chromosome.
    Returns fitness, feasibility and route"""
    route, cost = join_algorithm(chromosome, truck_time, drone_time, drone_range)
    logger.debug(f"Route: {route}")
    logger.debug(f"cost: {cost}")

    feasibility = 0
    # 0 for feas, 1 for infeas type 1, 2 for infeas type 2
    prev_action = None
    for action in route:
        if action[0] == 'LL':
            _, launch, deliver, land, duration = action
            # drone range check
            if duration > drone_range:
                feasibility = 2
                break
            # two consecutive drone nodes check
            if prev_action and prev_action[0] == 'LL':
                feasibility = 1
                break
        prev_action = action
    logger.debug(f"route: {route}, feasibility: {feasibility}, fitness: {cost}")
    return cost, feasibility, route


def tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range, k=2):
    """ kTOURNAMENT individuals are randomly selected from the entire population
    and the best one is selected as the parent based on fitness"""
    # sort by fitness
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0])]
    logger.debug(sorted_pop)
    # take k random individuals from the sorted population
    candidates = random.sample(sorted_pop, k)
    logger.debug(candidates)
    # return the best one
    return min(candidates, key=lambda chrom: evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range))


def sign_mutation(chromosome, prob=0.1):
    """independently changes the sign of each element in the chromosome with a fixed probability of 0.1"""
    logger.info(f"Starting sign mutation.")
    return [g if random.random() > prob else -g for g in chromosome]


def tour_mutation(chromosome, fraction=0.2):
    """20% of the indices in the chromosome are randomly shuffled"""
    logger.info(f"Starting tour mutation.")
    chrom = chromosome[:]
    n = len(chrom)
    count = max(1, int(fraction * n))  # number of genes to be shuffled
    # logger.debug(f"count: {count}")
    indices = random.sample(range(n), count)  # indices to be shuffled
    # logger.debug(f"indices: {indices}")
    values = [chrom[i] for i in indices]  # values by indexes
    # logger.debug(f"values: {values}")
    random.shuffle(values)
    # reassemble the chromosome
    for idx, val in zip(indices, values):
        chrom[idx] = val
    # logger.debug(f"mutated chromosome: {chrom}")
    return chrom


def tox1(parent1, parent2):
    """TOX1 crossover: copies a fragment of one type from p1, adds the rest from p2."""
    logger.info(f"Starting with tox1.")
    n = len(parent1)
    i1, i2 = sorted(random.sample(range(n), 2))
    # logger.debug(f"fragment between i1={i1} and i2={i2}")
    segment_type = 'truck' if random.random() < 0.5 else 'drone'
    # logger.debug(f"segment_type: {segment_type}")

    def type_of_node(g):
        return 'truck' if g > 0 else 'drone'

    # copy the substring of the required type from p1
    segment = [g for g in parent1[i1:i2+1] if type_of_node(g) == segment_type]
    # logger.debug(f"segment: {segment}")
    segment_ids = set(abs(g) for g in segment)
    # logger.debug(f"segment ids: {segment_ids}")

    # add missing genes from p2 in their order and type
    rest = [g for g in parent2 if abs(g) not in segment_ids]
    # logger.debug(f"rest: {rest}")

    # resulting child
    offspring = segment + rest
    # logger.debug(f"offspring: {offspring}")
    return offspring


def tox2(parent1, parent2):
    """TOX2 crossover"""
    logger.info(f"Starting with tox2.")
    n = len(parent1)
    i1, i2 = sorted(random.sample(range(n), 2))
    # logger.debug(f"between: {i1} and {i2}")
    # copy the segment from p1
    segment = parent1[i1:i2+1]
    # logger.debug(f"segment: {segment}")
    segment_ids = set(abs(g) for g in segment)  # stores nodes without signs that are already in this segment
    # logger.debug(f"segment ids: {segment_ids}")

    # add missing genes from p2 in order
    rest = [g for g in parent2 if abs(g) not in segment_ids]
    # logger.debug(f"rest: {rest}")

    # offspring = start + segment + end
    offspring_base = rest[:i1] + segment + rest[i1:]
    # logger.debug(f"base: {offspring_base}")

    # assign signs: for i1..i2: use signs from p2, rest: use signs from p1
    sign_p1 = {abs(g): g for g in parent1}
    # logger.debug(f"sign_p1: {sign_p1}")
    sign_p2 = {abs(g): g for g in parent2}
    # logger.debug(f"sign_p2: {sign_p2}")
    offspring = []
    for idx, g in enumerate(offspring_base):
        node_id = abs(g)
        if i1 <= idx <= i2:
            signed = sign_p2.get(node_id, g)  # sign from P2
        else:
            signed = sign_p1.get(node_id, g)  # sign from P1
        offspring.append(signed)
    # logger.debug(f"offspring: {offspring}")
    return offspring


def local_search(chromosome, truck_time, drone_time, drone_range):
    logger.info(f"Starting with local search.")
    logger.debug(f"chrom: {chromosome}")
    best_chrom = chromosome
    best_cost, _, _ = evaluate(best_chrom, truck_time, drone_time, drone_range)
    max_attempts = 5  # how much?
    attempts = 0
    improved = True
    local_moves = [local_search_l1  # todo more
                   ]
    while improved and attempts < max_attempts:
        improved = False
        move = random.choice(local_moves)
        new_chrom = move(best_chrom)
        new_cost, _, _ = evaluate(new_chrom, truck_time, drone_time, drone_range)
        if new_cost < best_cost:
            best_cost = new_cost
            best_chrom = new_chrom
            improved = True
        attempts += 1
    logger.debug(f"chrom was: {chromosome}")
    logger.debug(f"chrom ist: {best_chrom}")
    return best_chrom, best_cost


def local_search_l1(chromosome):
    """
    Choose three consecutive truck nodes and convert the middle one to a drone node
    """
    chrom = chromosome[:]
    n = len(chrom)
    # find three consecutive truck nodes
    for i in range(n - 2):
        if chrom[i] > 0 and chrom[i+1] > 0 and chrom[i+2] > 0:
            chrom[i+1] = -chrom[i+1]  # convert the middle one to a drone node
            return chrom
    return chrom


def repair(chromosome, truck_time, drone_time, drone_range, p_repair=0.5):
    """
    Repair infeasible chromosome by converting drone nodes to truck nodes
    based on violations detected in join_algorithm.
    """
    # copy the chromosome
    chrom = chromosome[:]
    logger.debug(f"chromosome: {chrom}")
    # join algorithm for flights
    route, _ = join_algorithm(chrom, truck_time, drone_time, drone_range)
    logger.debug(f"route: {route}")
    # violating drone nodes
    violating_nodes = set()
    prev_action = None

    for action in route:
        if action[0] == 'LL':
            _, launch, deliver, land, duration = action
            # drone range
            if duration > drone_range:
                violating_nodes.add(deliver)

            if prev_action and prev_action[0] == 'LL':
                # two consecutive drone nodes
                violating_nodes.add(deliver)
        prev_action = action

    # repair with probability p_repair
    for i, g in enumerate(chrom):
        if abs(g) in violating_nodes and g < 0:
            if random.random() < p_repair:
                chrom[i] = abs(g)
    logger.debug(f"repaired chromosome: {chromosome}")
    return chrom


def genetic_algorithm(places, generations=1, population_size=3, truck_speed=10, drone_range=float('inf')):
    drone_speed = 2 * truck_speed
    feasible_pop = []
    infeasible_1_pop = []
    infeasible_2_pop = []
    max_no_improve = 10  # ItNI

    # TSP
    #route, cost = solve_tsp_local_search(truck_time)
    #logger.debug(f"Route for TSP: {route}")
    #logger.debug(f"Time for TSP: {cost}")

    # init TSPD
    n = len(places) - 1
    population = generate_initial_population(n, population_size)
    # fallback values
    fallback_solution = None
    fallback_route = None
    fallback_fitness = float('inf')

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
    fitnesses = []
    best_fitness = float('inf')  # min makespan
    best_solution = None  # chromosome that gave the best result
    best_route = None  # list of actions (MT/LL) for the best chromosome
    improved = 0
    no_improve_count = 0

    # Evaluate initial population and set fallback
    for chrom in population:
        fit, feas, route = evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range)
        fitnesses.append(fit)
        if fit < fallback_fitness:
            fallback_fitness = fit
            fallback_solution = chrom
            fallback_route = route

    with open("mutations.txt", "w", encoding="utf-8") as file:

        for g in range(generations):
            # evaluate fitness of all chromosomes
            # fitnesses = [evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range) for chrom in population]
            for chrom in population:
                fit, feas, route = evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range)
                fitnesses.append(fit)
            logger.info(fitnesses)
            file.write(str(fitnesses))

            # tournament selection
            p1 = tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range)
            logger.info(f"first parent: {p1}")  # [3, -1, 2]
            p2 = tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range)
            while p1 == p2:
                p2 = tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range)
            logger.info(f"second parent: {p2}")  # [2, -1, 3]
            file.write(str(p1))
            file.write(str(p2))
            # if still p1==p2 ?

            # crossover -> new child
            child = tox1(p1, p2) if random.random() < 0.5 else tox2(p1, p2)
            logger.debug(f"child: {child}")  # child: [-1, 2, -3]
            file.write(str(child))
            # mutation
            if random.random() < 0.5:
                child = sign_mutation(child)
            else:
                child = tour_mutation(child)
            logger.debug(f"child after mutation: {child}")  # child after mutation: [-1, 2, 3]
            file.write(str(child))

            # evaluate child after all mutations
            fitness, feasible, route = evaluate(child, truck_time_matrix, drone_time_matrix, drone_range)
            # if infeasible -> repair
            if feasible in [1, 2]:
                child = repair(child, truck_time_matrix, drone_time_matrix, drone_range, p_repair=0.5)
                fitness, feasible, route = evaluate(child, truck_time_matrix, drone_time_matrix, drone_range)
                # if infeasible -> penalty
                if feasible in [1, 2]:
                    fitness += 99999  # todo adjust penalties
            # if feasible
            if feasible == 0:
                # local search only for feasible chrom
                child, fitness = local_search(child, truck_time_matrix, drone_time_matrix, drone_range)
                logger.debug(f"child after local search: {child}")
                file.write(str(child))
                feasible_pop.append((child, fitness))
                logger.debug(f"fitness: {fitness}")
                # save the best solutions
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = child
                    best_route = route  # можно также переоценить: route, _ = join_algorithm(child, ...)
                    improved = True

            # if infeasible
            if feasible == 1:
                infeasible_1_pop.append((child, fitness))
            if feasible == 2:
                infeasible_2_pop.append((child, fitness))

            # update population
            population.append(child)
            population = sorted(population, key=lambda chrom: evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range)[0])
            population = population[:population_size]  # keep best

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= max_no_improve:
                break
    # If no feasible solution was found, return fallback
    #if best_solution is None:
    #    logger.warning("No feasible solution found. Returning best from initial population.")
    #    return fallback_solution, fallback_route, fallback_fitness

    return best_solution, best_route, best_fitness


if __name__ == "__main__":
    drone_speed = 20  # m/s
    truck_speed = 10
    drone_range = float('inf')

    #places = [(50.149, 8.666),  # idx=0
    #          (50.145, 8.616),  # idx=1
    #          (50.147, 8.668),  # idx=2
    #          (50.146, 8.777),
    #          # (50.155, 7.777)
    #          ]

    places = [(50.116237219723246, 8.675090069320808),
              (50.117308811831265, 8.655780559661203),
              (50.098311570186645, 8.674780332636217),
              (50.10687159272978, 8.686167957918643),
              (50.10399599096225, 8.66606177300608),
              (50.11114186944436, 8.689019112449396)
              ]
    n = len(places)  # n=3 (без учёта 0′)
    logger.info(f"For {n} points.")
    chrom, route, fitness = genetic_algorithm(places)
    # ([3, 2, -1],
    # (784.21, 0,
    # [('MT', 1, 4, 359.05), ('MT', 2, 1, 730.26), ('MT', 0, 2, 52.82000000000001), ('LL', 0, 3, 2, 784.21)]))
    places.append(places[0])
    logger.info(f"Finally: chrom={chrom}, route={route}, fitness={fitness}")
    visualize_route(places, route)



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


    #places.append(places[0])  # depo needed for join
    #optimal_route, makespan = join_algorithm(chromosome2, truck_time_matrix, drone_time_matrix, drone_range=float('inf'))
    #visualize_route(places, optimal_route)

