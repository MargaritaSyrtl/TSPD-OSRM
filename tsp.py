from TSP.DMRequest import DMRequest
from visualisation import create_optimal_route_html
from TSP.helpers import *
from statistics import mean
from python_tsp.heuristics import solve_tsp_local_search
import random
import math
from math import isinf
import numpy as np
import copy
from copy import deepcopy


def advanced_join_algorithm(
        chromosome,
        drone_speed_ratio=2.0,
        drone_range=float('inf'),
        s_R=30,  # drone takeoff time
        s_L=30,  # drone landing time
        epsilon=900,  # time limit for the entire LL operation for a drone in seconds (eg 15min)
):

    INF = float('inf')
    truck_indices = [idx for idx, (node, t) in enumerate(chromosome) if t == 'truck']
    m = len(truck_indices)  # number of truck nodes
    if m == 0:
        return math.inf, {}

    truck_coords = [chromosome[i][0] for i in truck_indices]

    C = [INF] * (m + 1)
    C_MT = [INF] * (m + 1)
    C_LL = [INF] * (m + 1)
    choice = [None] * (m + 1)

    C[m] = 0.0
    C_MT[m] = 0.0
    C_LL[m] = 0.0
    choice[m] = ("END", None)

    truck_speed = 10.0  # m/s todo adjust better

    truck_dist = {}
    for i_ in range(m):  # running through all the track nodes
        for k_ in range(i_ + 1, m + 1):
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

    def find_all_drones_between(i_chromosome, k_chromosome):
        """ Searches all nodes between i and k """
        return [
            j for j in range(i_chromosome + 1, k_chromosome)
            if chromosome[j][1] == 'drone'
        ]

    # MT
    for i_ in range(m - 1, -1, -1):
        best_val_mt = INF
        best_k_mt = None
        for k_ in range(i_ + 1, m + 1):
            cost = truck_dist.get((i_, k_), INF) / truck_speed + C[k_]  # in sec
            logger.debug(f"Truck time: {cost}")
            if cost < best_val_mt:
                best_val_mt = cost
                best_k_mt = k_
        C_MT[i_] = best_val_mt

    # LL
        i_chromosome = truck_indices[i_]
        best_val_ll = INF  # global minimum for given i_
        best_k_ll = None
        best_d_ll = None

        for k_ in range(i_ + 1, m + 1):
            k_chromosome = truck_indices[k_] if k_ < m else len(chromosome)
            available_drones = find_all_drones_between(i_chromosome, k_chromosome)

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
                    continue

                dist_ld = euclidean_distance(node_launch, node_drone)
                dist_dl = euclidean_distance(node_drone, node_land)
                drone_flight_dist = dist_ld + dist_dl

                if drone_flight_dist > drone_range:
                    continue

                t_truck = truck_dist.get((i_, k_), INF) / truck_speed  # in seconds
                t_drone = drone_flight_dist / (drone_speed_ratio * truck_speed)  # in seconds
                logger.debug(f"Drone time: {t_drone}")
                logger.debug(f"Truck time: {t_truck}")
                # FSTSP
                sigma_k = 1 if choice[k_] and choice[k_][0] == "LL" else 0
                feasible = (
                        (t_truck + s_R + sigma_k * s_L <= epsilon) and
                        (t_drone + s_R <= epsilon)
                )
                if not feasible:
                    continue
                segtime = max(t_truck + sigma_k * s_L + s_R, t_drone + s_R)
                # segtime = max(t_drone, t_truck)
                val = segtime + C[k_]

                if val < best_val_ll and all([node_land != node_drone, node_launch != node_drone]):  # drone can't launch and land from/on drone node
                    best_val_ll, best_k_ll, best_d_ll = val, k_, d_i

        if best_k_ll is not None and best_d_ll is not None:
            C_LL[i_] = best_val_ll
        else:
            C_LL[i_] = INF

        if C_LL[i_] < C_MT[i_] and C_LL[i_] < INF:
            C[i_] = C_LL[i_]
            choice[i_] = ("LL", best_k_ll, best_d_ll)
        elif C_MT[i_] < INF:
            C[i_] = C_MT[i_]
            choice[i_] = ("MT", best_k_mt)
        else:
            C[i_] = INF
            choice[i_] = None

    makespan = C[0]
    flight_map = {}

    def backtrack(i_idx):
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


def local_search(chromosome, drone_speed_ratio=2.0, max_attempts=20):
    best_chrom = chromosome
    fitness_value, best_fit, best_flights, feasibility_status = compute_fitness(best_chrom, drone_speed_ratio)

    attempts = 0
    improved = True
    local_moves = [
        convert_to_drone,
        convert_to_truck,
        swap_neighbors,
        random_reorder,
        local_search_L1,
        local_search_L2,
        local_search_L3,
        local_search_L4,
        local_search_L5,
        local_search_L6,
        local_search_L7,
        repair_three_trucks_make_middle_drone,
        N1_relocate_truck_1_1,
        N2_relocate_truck_2_1_forward,
        N3_relocate_truck_2_1_reverse,
        N4_swap_truck_1_1,
        N7_2opt_truck,
        N8_2opt_truck_reverse,
        N9_swap_drone_truck,
        N10_swap_i_j,
        N11_swap_j_k,
        N12_swap_i_k,
        N15_drone_swap_1_1
    ]

    while improved and attempts < max_attempts:
        improved = False
        move = random.choice(local_moves)
        new_chrom = move(best_chrom)
        new_fitness, new_fit, new_flights, new_status = compute_fitness(new_chrom,
                                                                        drone_speed_ratio)
        if new_fit < best_fit:
            best_fit = new_fit
            best_chrom = new_chrom
            improved = True
            best_flights = new_flights

        attempts += 1
    return best_chrom, best_fit, best_flights


def convert_to_drone(chromosome):
    chrom_list = list(chromosome)
    truck_positions = [i for i, (n, t) in enumerate(chrom_list) if t == 'truck' and i != 0]
    if truck_positions:
        pos = random.choice(truck_positions)
        node, _ = chrom_list[pos]
        chrom_list[pos] = (node, 'drone')
    return tuple(chrom_list)


def convert_to_truck(chromosome):
    chrom_list = list(chromosome)
    drone_positions = [i for i, (n, t) in enumerate(chrom_list) if t == 'drone']
    if drone_positions:
        pos = random.choice(drone_positions)
        node, _ = chrom_list[pos]
        chrom_list[pos] = (node, 'truck')
    return tuple(chrom_list)


def swap_neighbors(chromosome):
    if len(chromosome) < 3:
        return chromosome
    chrom_list = list(chromosome)

    i = random.randint(1, len(chrom_list) - 2)
    chrom_list[i], chrom_list[i + 1] = chrom_list[i + 1], chrom_list[i]
    return tuple(chrom_list)


def random_reorder(chromosome):
    """ Changes the order of cities for local search so that the drone can be not only in 2nd place
    """
    chrom_list = list(chromosome)
    seq = chrom_list[1:]
    random.shuffle(seq)
    return tuple([chrom_list[0]] + seq)


def local_search_L1(chromosome):
    """
    Change middle of three consecutive truck nodes into a drone node.
    """
    chrom = list(chromosome)
    truck_positions = [i for i, (n, t) in enumerate(chrom) if t == 'truck']
    if len(truck_positions) < 3:
        return chromosome

    # select three consecutive truck nodes
    idx = random.randint(0, len(truck_positions) - 3)
    middle_idx = truck_positions[idx + 1]
    node, _ = chrom[middle_idx]
    chrom[middle_idx] = (node, 'drone')
    return tuple(chrom)


def local_search_L2(chromosome):
    """
    Moving a drone node between two truck nodes
    """
    chrom = list(chromosome)
    drone_positions = [i for i, (n, t) in enumerate(chrom) if t == 'drone']
    truck_positions = [i for i, (n, t) in enumerate(chrom) if t == 'truck']

    if not drone_positions or len(truck_positions) < 2:
        return chromosome

    drone_idx = random.choice(drone_positions)
    node, _ = chrom.pop(drone_idx)
    insert_idx = random.randint(1, len(chrom)-1)
    chrom.insert(insert_idx, (node, 'drone'))
    return tuple(chrom)


def local_search_L3(chromosome):
    """
    Exchange of truck and drone nodes without changing the position type.
    """
    chrom = list(chromosome)
    truck_positions = [i for i, (n, t) in enumerate(chrom) if t == 'truck']
    drone_positions = [i for i, (n, t) in enumerate(chrom) if t == 'drone']

    if not truck_positions or not drone_positions:
        return chromosome

    t_idx = random.choice(truck_positions)
    d_idx = random.choice(drone_positions)
    chrom[t_idx], chrom[d_idx] = chrom[d_idx], chrom[t_idx]
    return tuple(chrom)


def local_search_L4(chromosome):
    """
    Rearrangement of two truck arcs with reversal of the middle.
    """
    chrom = list(chromosome)
    truck_positions = [i for i, (n, t) in enumerate(chrom) if t == 'truck']

    if len(truck_positions) < 4:
        return chromosome

    i, j = sorted(random.sample(truck_positions, 2))
    chrom[i:j+1] = reversed(chrom[i:j+1])
    return tuple(chrom)


def local_search_L5(chromosome):
    """
    Exchange two drone nodes and convert them into a truck.
    """
    chrom = list(chromosome)
    drone_positions = [i for i, (n, t) in enumerate(chrom) if t == 'drone']
    if len(drone_positions) < 2:
        return chromosome

    i, j = random.sample(drone_positions, 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    node_i, _ = chrom[i]
    node_j, _ = chrom[j]
    chrom[i] = (node_i, 'truck')
    chrom[j] = (node_j, 'truck')
    return tuple(chrom)


def local_search_L6(chromosome):
    """
    Exchange two drone nodes and convert one of them into a truck.
    """
    chrom = list(chromosome)
    drone_positions = [i for i, (n, t) in enumerate(chrom) if t == 'drone']
    if len(drone_positions) < 2:
        return chromosome

    i, j = random.sample(drone_positions, 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    if random.random() < 0.5:
        node_i, _ = chrom[i]
        chrom[i] = (node_i, 'truck')
    else:
        node_j, _ = chrom[j]
        chrom[j] = (node_j, 'truck')
    return tuple(chrom)


def local_search_L7(chromosome):
    chrom = list(chromosome)
    drone_pos = [i for i,(n,t) in enumerate(chrom) if t=='drone']
    if not drone_pos:
        return chromosome
    i = random.choice(drone_pos)
    node, _ = chrom.pop(i)
    j = random.randint(1, len(chrom))
    chrom.insert(j, (node,'drone'))
    return tuple(chrom)


def repair_three_trucks_make_middle_drone(chrom):
    """
    If three consecutive truck nodes (T T T) are encountered,
    transforms the middle one into a drone node: (T D T).
    """
    chrom_list = list(chrom)
    for i in range(1, len(chrom_list) - 1):
        if (chrom_list[i - 1][1] == 'truck'
                and chrom_list[i][1] == 'truck'
                and chrom_list[i + 1][1] == 'truck'):
            node, _ = chrom_list[i]
            chrom_list[i] = (node, 'drone')
    return tuple(chrom_list)


def N1_relocate_truck_1_1(chrom):
    """ truck‑only relocation 1–1 """
    chrom = list(chrom)
    trucks = [i for i,(_,t) in enumerate(chrom) if t=='truck' and i!=0]
    if len(trucks) < 2:
        return tuple(chrom)
    u, v = random.sample(trucks, 2)
    node = chrom.pop(u)
    v = v if u>v else v-1  # shift after pop
    chrom.insert(v+1, node)
    return tuple(chrom)


def N2_relocate_truck_2_1_forward(chrom, rev=False):
    """ truck-only relocation 2–1
    If rev=False → move the pair (u1,u2) → (u1, u2)
    If rev=True  → move the pair (u1,u2) → (u2, u1)
    """
    chrom = list(chrom)
    trucks = [i for i,(_,t) in enumerate(chrom) if t=='truck' and i<len(chrom)-1]
    trucks = [i for i in trucks if chrom[i+1][1]=='truck']
    if not trucks:
        return tuple(chrom)
    u1 = random.choice(trucks)
    seg = chrom[u1:u1+2][::-1] if rev else chrom[u1:u1+2]    # (u1,u2) or (u2,u1)
    del chrom[u1:u1+2]
    others = [i for i, (_, t) in enumerate(chrom) if t == 'truck' and i not in (u1, u1 + 1)]
    if not others:
        return tuple(chrom)

    v = random.choice(others)
    v = v if u1>v else v-1
    chrom[v+1:v+1] = seg
    return tuple(chrom)


def N3_relocate_truck_2_1_reverse(chrom):
    return N2_relocate_truck_2_1_forward(chrom, rev=True)


def N4_swap_truck_1_1(chrom):
    chrom = list(chrom)
    trucks = [i for i,(_,t) in enumerate(chrom) if t=='truck' and i!=0]
    if len(trucks)<2:
        return tuple(chrom)
    i,j = random.sample(trucks,2)
    chrom[i],chrom[j] = chrom[j],chrom[i]
    return tuple(chrom)


def N5_swap_truck_2_1(chrom):
    chrom = list(chrom)
    pairs = [i for i,(_,t) in enumerate(chrom[:-1])
             if t=='truck' and chrom[i+1][1]=='truck']
    if not pairs:
        return tuple(chrom)
    u = random.choice(pairs)
    others = [k for k,(_,t) in enumerate(chrom) if t=='truck' and k not in (u,u+1)]
    if not others:
        return tuple(chrom)
    v = random.choice(others)
    seg = chrom[u:u+2]
    del chrom[u:u+2]
    v = v if u>v else v-1
    chrom.insert(v+1, seg[1])
    chrom.insert(v+2, seg[0])
    return tuple(chrom)


def N6_swap_truck_2_2(chrom):
    chrom=list(chrom)
    pairs=[i for i,(_,t) in enumerate(chrom[:-1])
           if t=='truck' and chrom[i+1][1]=='truck']
    if len(pairs)<2:
        return tuple(chrom)
    p,q = random.sample(pairs,2)
    # cut out segments of length 2
    seg1, seg2 = chrom[p:p+2], chrom[q:q+2]
    chrom[p:p+2], chrom[q:q+2] = seg2, seg1
    return tuple(chrom)


def N7_2opt_truck(chrom, cross=False):
    chrom = list(chrom)
    trucks = [i for i,(_,t) in enumerate(chrom)]
    if len(trucks) < 4:
        return tuple(chrom)
    i, j = sorted(random.sample(range(1, len(chrom)-1), 2))
    if cross:
        chrom[i:j+1] = reversed(chrom[i:j+1])
    else:
        chrom = chrom[:i]+chrom[j+1:]+chrom[i:j+1]
    return tuple(chrom)


def N8_2opt_truck_reverse(chrom):
    return N7_2opt_truck(chrom, cross=True)


def _find_drone_triplets(chrom):
    """⟨launch i, drone j, land k⟩"""
    trips=[]
    stack=[]
    for idx,(coord,t) in enumerate(chrom):
        if t=='drone':
            if stack:
                trips.append((stack[-1],idx))
        elif t=='truck':
            stack.append(idx)
    # land – first truck-node after drone-node
    clean=[]
    for l_idx, d_idx in trips:
        k_idx = next((x for x in range(d_idx+1,len(chrom)) if chrom[x][1]=='truck'), None)
        if k_idx: clean.append((l_idx,d_idx,k_idx))
    return clean


def N9_swap_drone_truck(chrom):
    chrom=list(chrom)
    drones=[i for i,(_,t) in enumerate(chrom) if t=='drone']
    trucks=[i for i,(_,t) in enumerate(chrom) if t=='truck' and i!=0]
    if not drones or not trucks: return tuple(chrom)
    d,u = random.choice(drones), random.choice(trucks)
    chrom[d], chrom[u] = chrom[u], chrom[d]
    return tuple(chrom)


def N10_swap_i_j(chrom):
    chrom=list(chrom)
    trips=_find_drone_triplets(chrom)
    if not trips: return tuple(chrom)
    i,j,k = random.choice(trips)
    chrom[i],chrom[j] = chrom[j],chrom[i]
    return tuple(chrom)


def N11_swap_j_k(chrom):
    chrom=list(chrom)
    trips=_find_drone_triplets(chrom)
    if not trips: return tuple(chrom)
    i,j,k = random.choice(trips)
    chrom[j],chrom[k] = chrom[k],chrom[j]
    return tuple(chrom)


def N12_swap_i_k(chrom):
    chrom=list(chrom)
    trips=_find_drone_triplets(chrom)
    if not trips:
        return tuple(chrom)
    i,j,k = random.choice(trips)
    chrom[i],chrom[k] = chrom[k],chrom[i]
    return tuple(chrom)


def N13_drone_insertion(chrom):
    """
    Makes an existing node a drone node, ensuring that there are no other drones between the selected launch and land truck nodes (i, k).
    """
    chrom = list(chrom)
    trucks = [idx for idx, (_, t) in enumerate(chrom) if t == 'truck']

    # search for all pairs (i,k) without drones between them
    valid_pairs = []
    for a in range(len(trucks) - 1):
        for b in range(a + 1, len(trucks)):
            i, k = trucks[a], trucks[b]
            if all(chrom[p][1] == 'truck' for p in range(i + 1, k)):
                valid_pairs.append((i, k))

    if not valid_pairs:
        return tuple(chrom)

    i, k = random.choice(valid_pairs)

    # do not move the node between i and k, but just skip it
    # candidates = [p for p in range(i + 1, k) if chrom[p][1] == 'truck']  # nodes between i and k
    # if not candidates:
    #     return tuple(chrom)
    # j = random.choice(candidates)

    # select any node that will become a drone
    j = random.randint(1, len(chrom) - 1)
    node, _ = chrom[j]
    chrom[j] = (node, 'drone')
    # if the new drone is outside (i,k), move it inside
    if not (i < j < k):
        chrom.pop(j)
        insert_pos = random.randint(i + 1, k - 1)
        chrom.insert(insert_pos, (node, 'drone'))
    return tuple(chrom)


def N15_drone_swap_1_1(chrom):
    chrom=list(chrom)
    drones=[i for i,(_,t) in enumerate(chrom) if t=='drone']
    if len(drones)<2: return tuple(chrom)
    a,b = random.sample(drones,2)
    chrom[a],chrom[b] = chrom[b],chrom[a]
    return tuple(chrom)


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


def type_aware_order_crossover(p1: tuple, p2: tuple, all_nodes: set | None = None,
                               variant: str | None = None):
    """
    TOX1 / TOX2
    """
    if len(p1) != len(p2):
        raise ValueError("Parents must be equal‑length")

    n = len(p1)
    depot = p1[0]
    cut1, cut2 = sorted(random.sample(range(1, n), 2))
    variant  = variant or random.choice(("TOX1", "TOX2"))
    all_nodes = all_nodes or {g[0] for g in p1[1:]}

    off = [None] * n
    off[0] = depot

    # copy segment
    # TOX1
    if variant == "TOX1":
        chosen_type = random.choice(("truck", "drone"))
        for i in range(cut1, cut2):
            # only nodes of the selected type are copied from the first parent p1 to the same positions of the child off
            if p1[i][1] == chosen_type:
                off[i] = p1[i]
    # TOX2
    else:
        # copy the segment [cut1:cut2] from the first parent in off
        off[cut1:cut2] = p1[cut1:cut2]
    # collect the coordinates of all nodes already placed in off
    used = {g[0] for g in off if g}
    src = [g for g in p2[1:] if g[0] not in used] + \
          [g for g in p1[1:] if g[0] not in used] + \
          [(coord, 'truck') for coord in all_nodes if coord not in used]

    idx = 0
    p2_type = {g[0]: g[1] for g in p2}
    # all cells that remain None are filled with elements from the src list
    for i in range(1, n):
        if off[i] is None:
            coord = src[idx][0]
            typ = p2_type.get(coord, src[idx][1])  # type from P2
            off[i] = (coord, typ)
            idx += 1

    if variant == "TOX2":
        p1_type = {g[0]: g[1] for g in p1}
        p2_type = {g[0]: g[1] for g in p2}
        for i in range(cut1, cut2):
            # type is changed to the one that the corresponding node in p2 had
            coord = off[i][0]
            # within segment [cut1:cut2]
            if cut1 <= i < cut2:
                # type from P2
                off[i] = (coord, p2_type.get(coord, off[i][1]))
            # outside segment [cut1:cut2]
            else:
                # type from P1
                off[i] = (coord, p1_type.get(coord, off[i][1]))
    return tuple(off)

def euclidean_distance(coord1, coord2):
    """
    Calculates the Euclidean distance between two points (lat, lon)
    """
    x1, y1 = map(float, coord1)
    x2, y2 = map(float, coord2)
    return math.hypot(x2 - x1, y2 - y1)


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


def mutate_agent(agent_genome, max_mutations=3):
    if len(agent_genome) < 3:
        return agent_genome
    genome_list = list(agent_genome)
    num_mut = random.randint(1, max_mutations)

    for _ in range(num_mut):
        i1 = random.randint(1, len(genome_list) - 1)
        i2 = random.randint(1, len(genome_list) - 1)
        if i1 != i2:
            (n1, t1) = genome_list[i1]
            (n2, t2) = genome_list[i2]
            if t1 == t2:  # only if type matches
                genome_list[i1], genome_list[i2] = genome_list[i2], genome_list[i1]
    return tuple(genome_list)


def shuffle_mutation(agent_genome):
    if len(agent_genome) <= 2:
        return agent_genome
    g_list = list(agent_genome)
    start = random.randint(1, len(g_list) - 1)
    length = random.randint(1, 3)
    sub = g_list[start:start + length]
    g_list = g_list[:start] + g_list[start + length:]
    insert_index = random.randint(1, len(g_list))
    g_list = g_list[:insert_index] + sub + g_list[insert_index:]

    return tuple(g_list)

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
            offspring1 = type_aware_order_crossover(p1, p2)
            logger.debug(f"{offspring1}")
            offspring1 = repair_chromosome(offspring1, [get_node(gene) for gene in initial_tsp_agent])
            logger.debug(f"{offspring1}")
            logger.info("start offspring2")
            offspring2 = type_aware_order_crossover(p2, p1)
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
            if len(recent_statuses) > 10:
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
    places = [(50.127177521308965, 8.667720581286744),  # Goethe Uni Westend
              (50.173571700260545, 8.630701738961589),  # Goethe Uni Riedberg
              (50.11988130693042, 8.652139750598623),  # Goethe Uni Bockenheim
              # (50.0967062213481, 8.661503898480328),  # Goethe Klinikum
              ]

    generations = 5000  # number of iterations
    population_size = 10  # number of agents
    # dm = DMRequest(places)

    optimal_route_ga = run_genetic(places, generations, population_size)
    logger.info(f"Optimal route: {optimal_route_ga}")

    best_chromosome = optimal_route_ga[0]
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
