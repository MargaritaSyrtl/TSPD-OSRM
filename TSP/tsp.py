from DMRequest import DMRequest
from visualisation import create_optimal_route_html
from helpers import *
from mutations_crossovers import *
from local_search import *
from statistics import mean
from python_tsp.heuristics import solve_tsp_local_search
import random
import math
from math import isinf
import numpy as np
import copy
from copy import deepcopy
from loguru import logger
logger.remove()
logger.add(
    "tspd.log",
    level="DEBUG",            # логгировать всё
    format="{time} | {level} | {name}:{line} - {message}",
    rotation="5 MB",
    encoding="utf-8",
    mode="w"
)


def advanced_join_algorithm(
        chromosome,
        drone_speed_ratio=2.0,
        drone_range=float('inf'),
        s_R=30,  # drone takeoff time
        s_L=30,  # drone landing time
        epsilon=900,  # time limit for the entire LL operation for a drone in seconds (eg 15min)
):
    logger.debug(f"chromosome: {chromosome}")
    INF = float('inf')
    truck_indices = [idx for idx, (node, t) in enumerate(chromosome) if t == 'truck']
    logger.debug(f"truck_indices: {truck_indices}")
    m = len(truck_indices)  # number of truck nodes
    logger.debug(f"m={m}")
    if m == 0:
        return math.inf, {}

    truck_coords = [chromosome[i][0] for i in truck_indices]
    logger.debug(f"truck_indices {truck_indices}")

    C = [INF] * (m + 1)
    C_MT = [INF] * (m + 1)
    C_LL = [INF] * (m + 1)
    choice = [None] * (m + 1)
    logger.info(f"C {C}")

    C[0] = 0.0
    C_MT[0] = 0.0
    C_LL[0] = 0.0
    choice[m] = ("END", None)
    logger.info(f"C[m] {C}")

    truck_speed = 10.0  # m/s

    truck_dist = {}
    for i_ in range(m - 1):  # running through all the track nodes
        for k_ in range(i_ + 1, m):
            logger.debug(f"i_ {i_}")
            logger.debug(f"k_ {k_}")
            if k_ == m:  # depot definition
                depot_coord = chromosome[0][0]
                # definition of distances between nodes
                dist = euclidean_distance(truck_coords[i_], depot_coord)
                # real traffic
                # dist = waypoints_distances.get(frozenset([truck_coords[i_], depot_coord]), INF)  # in meters ?
                truck_dist[(i_, k_)] = dist  # in meters
            else:
                coord_i = truck_coords[i_]
                coord_k = truck_coords[k_]
                # real traffic
                # dist = 0.0 if coord_i == coord_k else waypoints_distances.get(frozenset([coord_i, coord_k]), INF)
                dist = 0.0 if coord_i == coord_k else euclidean_distance(coord_i, coord_k)
                truck_dist[(i_, k_)] = dist
                # logger.debug(dist)

    def find_all_drones_between(i_chromosome, k_chromosome):
        """ Searches all nodes between i and k """
        drones = []
        for j in range(i_chromosome + 1, k_chromosome):
            logger.debug(f"j: {j}")
            if chromosome[j][1] == 'drone':
                drones.append(j)
        return drones

    logger.debug(f"m = {m}")
    for i_ in range(m - 1):
        best_val_mt = INF
        best_k_mt = None
        logger.info(f"Start MT cycle")
    # MT
        for k_ in range(i_ + 1, m):
            cost = (truck_dist.get((i_, k_), INF) / truck_speed) + C[i_]  # in sec
            # logger.debug(f"truck_ speed {truck_speed}")
            logger.debug(f"Truck dist for MT: {truck_dist.get((i_, k_), INF)} between {i_} and {k_}")
            logger.debug(f"Truck time for MT: {cost} between {i_} and {k_}")
            if cost < best_val_mt:
                best_val_mt = cost
                best_k_mt = k_
        C_MT[i_] = best_val_mt

    # LL
        logger.info(f"Start LL cycle")
        i_chromosome = truck_indices[i_]
        logger.debug(f"i_chromosome {i_chromosome}")
        best_val_ll = INF  # global minimum for given i_
        best_k_ll = None
        best_d_ll = None

        for k_ in range(i_ + 1, m + 1):
            logger.debug(f"i_={i_}")
            logger.debug(f"k_={k_}")
            k_chromosome = truck_indices[k_] if k_ < m else len(chromosome)
            logger.debug(f"k_chromosome {k_chromosome}")
            available_drones = find_all_drones_between(i_chromosome, k_chromosome)
            logger.debug(f"available drones {available_drones}")
            for d_i in available_drones:
                # compute launch/land nodes
                node_launch = chromosome[i_chromosome][0]
                node_drone = chromosome[d_i][0]
                node_land = chromosome[truck_indices[k_]][0] if k_ < m else chromosome[0][0]
                # chromosome[0][0] - depo
                if any([
                    node_land is None,
                    node_launch is None,
                    node_launch == node_drone,
                    node_land == node_drone
                    # drone can't launch and land from/on drone node
                ]):
                    logger.debug(f"node_launch={node_launch}, node_land={node_land}, node_drone={node_drone}")
                    continue

                start_drone = euclidean_distance(node_launch, node_drone)
                stop_drone = euclidean_distance(node_drone, node_land)
                drone_flight_dist = start_drone + stop_drone

                if drone_flight_dist > drone_range:
                    continue

                t_truck = truck_dist.get((i_, k_), INF) / truck_speed  # in seconds
                t_drone = drone_flight_dist / (drone_speed_ratio * truck_speed)  # in seconds
                logger.debug(f"Truck dist for LL: {truck_dist.get((i_, k_), INF)} between {i_} and {k_}")
                logger.debug(f"Truck time for LL: {t_truck} between {i_} and {k_}")

                logger.debug(f"Drone dist for LL: {drone_flight_dist} between {i_} and {k_}")
                logger.debug(f"launch -> drone: {node_launch} -> {node_drone} = {start_drone}")
                logger.debug(f"drone -> land: {node_drone} -> {node_land} = {stop_drone}")
                logger.debug(f"Drone time for LL: {t_drone} between {i_} and {k_}")
                # FSTSP
                #sigma_k = 1 if choice[k_] and choice[k_][0] == "LL" else 0
                #feasible = (
                #        (t_truck + s_R + sigma_k * s_L <= epsilon) and
                #        (t_drone + s_R <= epsilon)
                #)
                #if not feasible:
                #    continue
                #segtime = max(t_truck + sigma_k * s_L + s_R, t_drone + s_R)
                segtime = max(t_drone, t_truck)
                logger.debug(f"segtime {segtime}")
                logger.debug(f"C[{i_}]={C[i_]}")
                val = segtime + C[i_]
                logger.debug(f"val {val}")
                logger.debug(f"best_val_ll {best_val_ll}")
                logger.debug(val < best_val_ll)
                if val < best_val_ll and all([node_land != node_drone, node_launch != node_drone]):  # drone can't launch and land from/on drone node
                    logger.debug(f"val<best_val_ll")
                    best_val_ll, best_k_ll, best_d_ll = val, k_, d_i

        logger.debug(f"C_MT {C_MT} for i={i_}")
        logger.debug(f"C_LL {C_LL} for i={i_}")
        if best_k_ll is not None and best_d_ll is not None:
            C_LL[i_] = best_val_ll
        else:
            C_LL[i_] = INF
        logger.info(f"C_LL[i_] <= C_MT[i_] {C_LL[i_] <= C_MT[i_]}")
        logger.debug(f"C_LL[i_] > C_MT[i_] {C_LL[i_] > C_MT[i_]}")

        logger.debug(f"for I={i_}: \nC_MT={C_MT}\nC_LL={C_LL}")
        if C_LL[i_] <= C_MT[i_] and C_LL[i_] < INF:
            C[i_+1] = C_LL[i_]
            logger.info(f"C_LL chosen for next i={i_+1}")
            choice[i_] = ("LL", best_k_ll, best_d_ll)
        elif C_MT[i_] < INF:
            C[i_+1] = C_MT[i_]
            logger.info(f"C_MT chosen for next i={i_ + 1}")
            choice[i_] = ("MT", best_k_mt)
        else:
            C[i_] = INF
            logger.info(f"No choice for next i={i_ + 1}")
            choice[i_] = None




    logger.debug(f"Choice: {choice}")  # todo что хотим, что видим
    makespan = C[0]
    flight_map = {}

    def backtrack(i_idx):  # todo далее
        if i_idx >= m:
            return
        mode = choice[i_idx]
        if mode is None or mode[0] == "END":
            return
        if mode[0] == "MT":
            nxt = mode[1]
            if nxt is not None:
                backtrack(nxt)
        elif mode[0] == "LL":
            _, k_, d_idx = mode
            if d_idx is not None and d_idx != -1:
                launch_node = tuple(map(float, chromosome[truck_indices[i_idx]][0]))
                land_node = tuple(map(float, chromosome[truck_indices[k_]][0])) if k_ < len(truck_indices) else tuple(
                    map(float, chromosome[0][0]))
                drone_node = tuple(map(float, chromosome[d_idx][0]))

                #if euclidean_distance(land_node, drone_node) >= 1e-6 and \
                #        euclidean_distance(launch_node, land_node) >= 1e-6 and \
                #        euclidean_distance(launch_node, drone_node) >= 1e-6 and \
                if drone_node != land_node and drone_node != launch_node:   # drone can't launch and land from/on drone node
                    flight_map[d_idx] = (truck_indices[i_idx],  # launch – index in chromosome
                                         truck_indices[k_] if k_ < m else len(chromosome))  # land

            if k_ is not None:
                backtrack(k_)

    backtrack(0)
    return makespan, flight_map


def compute_fitness(chromosome, drone_speed_ratio=2.0, drone_range=float('inf'), w1=2.0, w2=2.0):
    try:
        makespan, flight_map = advanced_join_algorithm(chromosome, drone_speed_ratio)
        logger.debug(f"makespan {makespan}: flight_map: {flight_map}")
        # check type 1
        type1_viol = 0
        type1_pos = None
        for i in range(1, len(chromosome)):
            if chromosome[i][1] == 'drone' and chromosome[i - 1][1] == 'drone':
                type1_viol = 1
                type1_pos = i
                break

        # check type 2
        type2_viol = 0
        type2_pos = None
        if flight_map:
            for d_idx, (launch_truck_idx, land_truck_idx) in flight_map.items():
                if (launch_truck_idx is None) or (land_truck_idx is None):
                    continue

                #node_launch = chromosome[truck_indices[launch_truck_idx]][0]
                #node_land = chromosome[truck_indices[land_truck_idx]][0]
                node_launch = chromosome[launch_truck_idx][0]
                node_land = chromosome[land_truck_idx][0]
                node_drone = chromosome[d_idx][0]
                logger.info(node_drone)
                logger.info(node_launch)
                logger.info(node_land)

                dist_launch_drone = euclidean_distance(node_launch, node_drone)
                dist_drone_land = euclidean_distance(node_drone, node_land)
                drone_flight_dist = dist_launch_drone + dist_drone_land

                if drone_flight_dist > drone_range:
                    type2_viol = 1
                    type2_pos = d_idx
                    break

        # choose type
        if isinf(makespan):
            type2_viol = 1
            type2_pos = 0
        if type1_viol and type2_viol:
            if type1_pos is not None and (type2_pos is None or type1_pos <= type2_pos):
                feasibility_status = "type1"
            else:
                feasibility_status = "type2"
        elif type1_viol:
            feasibility_status = "type1"
        elif type2_viol:
            feasibility_status = "type2"
        else:
            feasibility_status = "feasible"

        fitness_value = makespan + w1 * type1_viol + w2 * type2_viol
        logger.info(f"Makespan: {makespan}, Feasibility: {feasibility_status}, "
                    f"fitness_value: {fitness_value}, flight_map: {flight_map}")
        for d_idx, (launch, land) in flight_map.items():
            logger.info(f"DRONE USED: drone #{d_idx} from truck #{launch} to #{land}")
        return fitness_value, makespan, flight_map, feasibility_status
    except Exception as e:
        print(f"Error in compute_fitness: {e}")
        return float('inf'), float('inf'), None, "type2"



def get_node(individual):
    return individual[0]


def remove_duplicates(population):
    seen = set()
    unique = []
    for agent in population:
        key = tuple(get_node(gene) for gene in agent)
        if key not in seen:
            seen.add(key)
            unique.append(agent)
    return unique



def generate_initial_tsp_solution_lkh(waypoints):
    """
    Generates a TSP route via python-tsp.
    Returns a list of (node, 'truck') - all customers are served by the truck.
    """
    n = len(waypoints)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = euclidean_distance(waypoints[i], waypoints[j])
            else:
                dist_matrix[i][j] = 1e9

    # TSP solver
    permutation, distance = solve_tsp_local_search(dist_matrix)

    # generate a route
    tsp_route = [(waypoints[i], 'truck') for i in permutation]
    logger.info(f"Initial tsp solution lkh: {tsp_route}")
    return tsp_route


def generate_agent_from_tsp_base(tsp_base):
    """
    Creates an agent based on a TSP solution (all trucks), modifying:
        - order (sequence-wise)
        - types (element-wise)
    """
    chrom = list(copy.deepcopy(tsp_base))  # convert to list
    # We convert 50% of clients into drones
    num_drones = max(1, int(0.5 * (len(chrom) - 1)))
    drone_positions = random.sample(range(1, len(chrom)), num_drones)
    # sequence-wise modification - mix everything except depot
    seq = chrom[1:]
    random.shuffle(seq)
    chrom = [chrom[0]] + seq
    for i in range(1, len(chrom)):
        node, _ = chrom[i]
        new_type = 'drone' if i in drone_positions else 'truck'
        chrom[i] = (node, new_type)

    # element-wise modification - change types randomly
    # for i in range(1, len(chrom)):
    #    node, _ = chrom[i]
    #    new_type = 'drone' if random.random() < 0.5 else 'truck'
    #    chrom[i] = (node, new_type)
    logger.info(chrom)

    return tuple(chrom)


def run_genetic(places, generations, population_size):
    """
    Requests data for GA from DMRequest and pass it to HGA-TAC implementation.
    """
    #waypoints_data_ga = dm.get_response_data_ga()
    #if waypoints_data_ga:
    #    waypoints_distances = waypoints_data_ga['waypoints_distances']
    #    if waypoints_distances:
    return run_genetic_algorithm(
                places=places,
                generations=generations,
                population_size=population_size)


def tournament_selection(population, tournament_size=3,
                         drone_speed_ratio=2.0, drone_range=float('inf'),
                         w1=2.0, w2=2.0, reference_length=None):
    """
    Tournament of random individuals.
    Returns the winner (the best by fitness), whose length matches reference_length.
    """
    filtered_population = [agent for agent in population if reference_length is None or len(agent) == reference_length]
    if not filtered_population:
        logger.warning("No agents match the required length — falling back to random from full population.")
        filtered_population = population

    competitors = random.sample(filtered_population, min(tournament_size, len(filtered_population)))
    best = None
    best_fitness = float('inf')

    for agent in competitors:
        fitness, _, _, _ = compute_fitness(agent, drone_speed_ratio, drone_range, w1, w2)
        if fitness < best_fitness:
            best = agent
            best_fitness = fitness

    if best is None:
        logger.warning("Tournament selection failed. Returning random agent.")
        return random.choice(filtered_population)
    return best


def sort_population_by_fitness(population, drone_speed_ratio=2.0, drone_range=float('inf'), w1=2.0, w2=2.0):
    """
    Calculates fitness for each agent.
    Returns a list of agents, sorted by fitness (from smallest to largest).
    Select kTOURNAMENT random individuals from the already sorted population
    and choose the 'best' one (with the smallest fitness) among them.
    """
    pop_with_fit = []
    for agent in population:
        fit_val, _, _, _ = compute_fitness(agent, drone_speed_ratio, drone_range, w1, w2)
        pop_with_fit.append((fit_val, agent))
    # sort via fit_val
    pop_with_fit.sort(key=lambda x: x[0])
    # return chromosomes in ascending order of fit value
    sorted_agents = [x[1] for x in pop_with_fit]
    return sorted_agents


def run_genetic_algorithm(places, generations, population_size, drone_speed_ratio=2.0, drone_range=float('inf'), escape_trigger=10):
    # init penalties
    w1 = 2.0
    w2 = 2.0

    pop_feas = []
    pop_type1_inf = []
    pop_type2_inf = []

    initial_tsp_agent = tuple(generate_initial_tsp_solution_lkh(places))

    fval, mspan, fl_map, st = compute_fitness(initial_tsp_agent, drone_speed_ratio, drone_range, w1, w2)
    if st == 'feasible':
        pop_feas.append(initial_tsp_agent)
    elif st == 'type1':
        pop_type1_inf.append(initial_tsp_agent)
    else:
        pop_type2_inf.append(initial_tsp_agent)

    # add (population_size - 1) agents
    for _ in range(population_size - 1):
            agent = generate_agent_from_tsp_base(initial_tsp_agent)
            logger.debug(agent)
            with open("log.txt", "a", encoding="utf-8") as file:
                file.write(str(agent))
            #return True
            #agent = (((50.147668332518805, 8.666132309606185), 'truck'), ((50.14721305727534, 8.66827645827173), 'drone'), ((50.14541029284123, 8.616449018150927), 'truck'))
            #logger.info(agent)
            fval, mspan, fl_map, st = compute_fitness(
                agent, drone_speed_ratio, drone_range, w1, w2
            )
            if st == 'feasible':
                pop_feas.append(agent)
            elif st == 'type1':
                pop_type1_inf.append(agent)
            else:
                pop_type2_inf.append(agent)
    pop_feas = remove_duplicates(pop_feas)
    pop_type1_inf = remove_duplicates(pop_type1_inf)
    pop_type2_inf = remove_duplicates(pop_type2_inf)

    best_sol = None
    best_fit_val = float('inf')
    best_sol_flights = None
    no_improve_count = 0

    for generation in range(generations):
        logger.info(f"Generation {generation}: "
                    f"Best fitness = {best_fit_val}, Feasible = {len(pop_feas)}, Type1 = {len(pop_type1_inf)}, Type2 = {len(pop_type2_inf)}")

        # unite all subpopulations
        all_pop = pop_feas + pop_type1_inf + pop_type2_inf
        # logger.debug(f"All population: {all_pop}")
        logger.debug(len(all_pop))
        if len(all_pop) < 2:
            break

        # sort the entire population
        sorted_pop = sort_population_by_fitness(all_pop, drone_speed_ratio, drone_range, w1, w2)
        # logger.debug(f"sorted population: {sorted_pop}")

        new_population = []
        num_offsprings = max(2, population_size // 2)
        # logger.debug(num_offsprings)
        recent_statuses = []  # statuses: feasible, type1 or type2
        # generate offspings
        for _ in range(num_offsprings):
            ref_length = len(initial_tsp_agent)

            p1 = tournament_selection(sorted_pop, 3, drone_speed_ratio, drone_range, w1, w2,
                                      reference_length=ref_length)
            p2 = tournament_selection(sorted_pop, 3, drone_speed_ratio, drone_range, w1, w2,
                                      reference_length=ref_length)

            #p1 = tournament_selection(sorted_pop, waypoints_distances, 3, drone_speed_ratio, drone_range, w1, w2)
            #p2 = tournament_selection(sorted_pop, waypoints_distances, 3, drone_speed_ratio, drone_range, w1, w2)
            logger.debug(f"{p1} \n{p2}")
            # crossover
            logger.info("start offspring1")
            offspring1 = crossover(p1, p2)
            logger.debug(f"{offspring1}")
            offspring1 = repair_chromosome(offspring1, [get_node(gene) for gene in initial_tsp_agent])
            logger.debug(f"{offspring1}")
            logger.info("start offspring2")
            offspring2 = crossover(p2, p1)
            logger.debug(offspring2)
            offspring2 = repair_chromosome(offspring2, [get_node(gene) for gene in initial_tsp_agent])
            logger.debug(f"{offspring2}")

            for child in [offspring1, offspring2]:
                logger.debug(child)
                # skip the chromosome where after crossover there are None
                if None in child:
                    logger.warning("Skipping child due to incomplete genome")
                    continue
                # mutation
                mut_child = mutate_agent(child, 3)
                logger.debug(mut_child)
                mut_child = repair_chromosome(mut_child, [get_node(gene) for gene in initial_tsp_agent])
                logger.debug(mut_child)
                # repair
                repaired = repair_consecutive_drones(mut_child)
                logger.debug(repaired)
                # local search
                improved_child, improved_fit, improved_flights = local_search(repaired,drone_speed_ratio)
                new_population.append(improved_child)
            # logger.debug(new_population)

        # evaluate the new population
        for agent in new_population:
            fval, mspan, fl_map, st = compute_fitness(agent, drone_speed_ratio, drone_range, w1, w2)
            if st == 'feasible':
                pop_feas.append(agent)
            elif st == 'type1':
                pop_type1_inf.append(agent)
            else:
                pop_type2_inf.append(agent)
            recent_statuses.append(st)  # save recent status
            if len(recent_statuses) > 100:
                recent_statuses.pop(0)  # remove the oldest element
        # calculate the best among feasible
        improved_flag = False
        for c in pop_feas:
            fval, mspan, fl_map, st = compute_fitness(c, drone_speed_ratio, drone_range, w1, w2)
            if mspan < best_fit_val:
                best_fit_val = mspan
                best_sol = c
                best_sol_flights = fl_map
                improved_flag = True

        if improved_flag:
            no_improve_count = 0
        else:
            no_improve_count += 1

        # calculate the proportions of feasible, type1 or type2 solutions obtained in the last 100 individuals
        recent_feasible = sum(1 for st in recent_statuses if st == 'feasible')
        recent_type1 = sum(1 for st in recent_statuses if st == 'type1')
        recent_type2 = sum(1 for st in recent_statuses if st == 'type2')
        total_count = len(recent_statuses)

        if total_count > 0:
                recent_feasible_ratio = recent_feasible / total_count
                recent_type1_ratio = recent_type1 / total_count
                recent_type2_ratio = recent_type2 / total_count
        else:
                recent_feasible_ratio = 0.0
                recent_type1_ratio = 0.0
                recent_type2_ratio = 0.0
        w1, w2 = adjust_penalties(recent_feasible_ratio, recent_type1_ratio, recent_type2_ratio, w1, w2)

        # escape strategy
        if no_improve_count > 10:
            logger.info("No improvement for 10 generations. Stopping.")
            break

    return best_sol, best_sol_flights


def adjust_penalties(recent_feasible_ratio,
                     recent_type1_ratio,
                     recent_type2_ratio,
                     w1, w2,
                     w1_min=1.0, w1_max=10.0,
                     w2_min=1.0, w2_max=10.0,
                     target_feas=0.3,  # wish: ~30% feas
                     eta_increase=1.1,
                     eta_decrease=0.9):
    """
      if recent_feasible_ratio < target_feas -> increase one of the penalties, depending on what happens more often: type1 or type2
      if recent_feasible_ratio > target_feas + 0.1 -> decrease the penalty.
    """
    # threshold
    low_thresh = target_feas - 0.05
    high_thresh = target_feas + 0.05

    new_w1 = w1
    new_w2 = w2

    if recent_feasible_ratio < low_thresh:
        # increase
        if recent_type1_ratio > recent_type2_ratio:
            new_w1 = min(w1 * eta_increase, w1_max)
        else:
            new_w2 = min(w2 * eta_increase, w2_max)

    elif recent_feasible_ratio > high_thresh:
        # decrease
        if recent_type1_ratio > recent_type2_ratio:
            new_w1 = max(w1 * eta_decrease, w1_min)
        else:
            new_w2 = max(w2 * eta_decrease, w2_min)

    return new_w1, new_w2


def repair_chromosome(chromosome, all_nodes):
    """
    Chromosome Repair:
        removes duplicates.
        adds missing cities to random positions.
    """
    seen = set()
    repaired = []
    for coords, node_type in chromosome:
        if coords not in seen:
            repaired.append((coords, node_type))
            seen.add(coords)

    current_nodes = set(get_node(gene) for gene in repaired)
    missing_nodes = set(all_nodes) - current_nodes

    for missing in missing_nodes:
        insert_idx = random.randint(1, len(repaired))
        repaired.insert(insert_idx, (missing, 'truck'))

    return tuple(repaired)


def repair_consecutive_drones(chromosome):
    """
        If a chromosome contains two drone-nodes in a row, convert the second one to 'truck'.
        """
    chrom_list = list(chromosome)
    for i in range(1, len(chrom_list)):
        if chrom_list[i-1][1] == 'drone' and chrom_list[i][1] == 'drone':
            # change the type
            chrom_list[i] = (chrom_list[i][0], 'truck')
    return tuple(chrom_list)


if __name__ == "__main__":
    # (1) Find the optimal route for cities
    # cities = ("Frankfurt", "Wiesbaden", "Offenbach", "Hanau", "Mainz", "Gießen")#, "Kassel", "Darmstadt")  # todo doesn't work for 3 cities
    # print(f"Find the best route from {cities[0]} to all cities from the list: {cities[1:]}")
    # places = []
    # for city in cities:
    #     places.append(get_coordinates(city))

    # (2) Find the optimal route for places in Frankfurt
    places = [(50.147668332518805, 8.666132309606185),
              (50.14541029284123, 8.616449018150927),
              (50.14721305727534, 8.66827645827173),
              ]

    generations = 1  # number of iterations
    population_size = 2  # number of agents
    # dm = DMRequest(places)

    optimal_route_ga = run_genetic(places, generations, population_size)
    logger.info(f"Optimal route: {optimal_route_ga}")

    best_chromosome = optimal_route_ga[0]
    logger.debug(f"chromoseom {best_chromosome}")
    # todo delete
    makespan, flights = advanced_join_algorithm(best_chromosome,
                                                drone_speed_ratio=2.0)
    logger.info(f"flights: {flights}")
    tmp = {}
    for key, value in flights.items():
        if value is not None:
            tmp[key] = value

    chrom_list = list(best_chromosome)
    for idx, (coords, typ) in enumerate(chrom_list):
        if typ == 'drone' and (idx not in flights or flights[idx] is None):
            chrom_list[idx] = (coords, 'truck')
    cleaned_chromosome = tuple(chrom_list)
    logger.info(f"Was: {best_chromosome}")
    logger.info(f"Ist: {cleaned_chromosome}")

    drone_dict = {}
    for key, value in tmp.items():
        logger.info(key)
        logger.info(value)
        drone_node = tuple(map(float, cleaned_chromosome[key][0]))
        launch_node = tuple(map(float, cleaned_chromosome[value[0]][0]))
        land_node = tuple(map(float, cleaned_chromosome[value[1]][0]))
        drone_dict[drone_node] = (launch_node, land_node)
    logger.info(f"drone_dict: {drone_dict}")

    cities_dict_ga = {}
    coord_dict_ga = {}
    for i, ((lat_str, lon_str), vehicle_type) in enumerate(cleaned_chromosome, start=0):
            lat = float(lat_str)
            lon = float(lon_str)
            city_name = get_city_names(lat, lon)
            logger.debug(f"{city_name} - ({lat},{lon})")
            logger.info(f"{i}. {city_name} ({vehicle_type})")
            cities_dict_ga[city_name] = vehicle_type
            coord_dict_ga[(lat, lon)] = vehicle_type

    drone_nodes = set()
    for drone_idx, (launch_idx, land_idx) in flights.items():
        drone_node = tuple(map(float, cleaned_chromosome[drone_idx][0]))
        drone_nodes.add(drone_node)

    truck_nodes = [
        tuple(map(float, node))
        for idx, (node, typ) in enumerate(cleaned_chromosome)
        if typ == 'truck' and tuple(map(float, node)) not in drone_nodes
    ]

    logger.debug(f"Truck nodes at first: {truck_nodes}")
    logger.debug(f"Drone nodes at first: {drone_nodes}")
    #geometry_route_ga = []
    #for i in range(len(truck_nodes) - 1):
    #    segment = dm.get_geometry_for_route(truck_nodes[i], truck_nodes[i + 1])
    #    if segment:
    #        geometry_route_ga.extend(segment)
    #return_route = dm.get_geometry_for_route(truck_nodes[-1], truck_nodes[0]) or []
    logger.info(f"Drone nodes: {drone_nodes}")
    logger.info(f"Drone dict: {drone_dict}")  # drone_dict[drone_node] = (launch_node, land_node)
    logger.info(f"Truck nodes: {truck_nodes}")
    # if the truck's path is empty, duplicate one point so that it doesn't fall
    if len(truck_nodes) < 2:
        logger.debug(f"Only 1 truck node")
        geometry_route_ga = [truck_nodes[0]]
        return_route = [truck_nodes[0]]

    # visualisation
    create_optimal_route_html(optimal_route=cleaned_chromosome, filename="route.html",
                              cities=truck_nodes, drone_nodes=drone_nodes, drone_route=drone_dict)
