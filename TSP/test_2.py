import math
from loguru import logger
from itertools import combinations
import folium
import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
from scipy.spatial.distance import cdist
import random
from folium.features import DivIcon


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
    w1 = 2.0
    w2 = 2.0
    action_trace = {}  # i -> best action ('MT' or 'LL')
    n = len(chromosome)
    logger.debug(chromosome)
    full_seq = [0] + [abs(x) for x in chromosome] + [n + 1]  # virtual end node
    logger.debug(f"full_seq {full_seq}")
    pos = {node: idx for idx, node in enumerate(full_seq)}
    logger.debug(f"dict node:idx = {pos}")

    node_types = {abs(x): 'truck' if x >= 0 else 'drone' for x in chromosome}
    logger.debug(f"nodes_types {node_types}")
    all_nodes_to_serve = set(i for i in range(1, n + 1))  # всё, кроме депо
    logger.debug(all_nodes_to_serve)
    # truck-узлы в порядке следования во full_seq
    truck_nodes = [0]
    drone_nodes = []
    #for i in range(1, n + 1):  # n+1 virtual node -> 3=0
    #    if node_types.get(i) == "truck":
    #        truck_nodes.append(i)
    for node in full_seq[1:-1]:  # without 0 and (n+1)
        if node_types.get(node) == 'truck':
            truck_nodes.append(node)
        if node_types.get(node) == "drone":
            drone_nodes.append(node)
    truck_nodes.append(n + 1)  # add virtual end node e.g. (0')=(3)
    logger.debug(f"truck_nodes {truck_nodes}")
    logger.debug(f"drone_nodes {drone_nodes}")
    last_real_truck = truck_nodes[-2]  # предпоследний элемент, т.к. [-1] == n+1 (0′)

    # cumulative time on truck route
    pos_truck = {node: idx for idx, node in enumerate(truck_nodes)}  # for each truck node: its position in truck_nodes
    logger.debug(f"pos truck: {pos_truck}")  # pos truck: {0: 0, 1: 1, 3: 2, 4: 3, 6: 4}
    prefix = [0.0]  # prefix[k] – total time from depot to truck_nodes[k]
    for p, q in zip(truck_nodes, truck_nodes[1:]):  # consecutive pairs
        prefix.append(prefix[-1] + truck_time[p][q])
        # truck_time[p][q] - time of direct movement along the matrix between neighbors
        # prefix[-1] - already accumulated time to node p
    logger.debug(f"prefix: {prefix}")

    def tau(a, b):  # a,b ∈ truck_nodes,  pos_truck[a] < pos_truck[b]
        """Returns the actual time of movement of the truck between nodes a and b in the already specified order.
        """
        return prefix[pos_truck[b]] - prefix[pos_truck[a]]

    # DP
    # C[i] — minimum time from truck node i to the end
    C = {}
    C[n + 1] = 0  # virtual end of the route
    C[truck_nodes[-1]] = 0  # C(end) = 0
    logger.debug(f"C init {C}")

    logger.debug(len(truck_nodes) - 1)
    logger.debug(range(len(truck_nodes) - 1))
    S = {}  # S[i] – множество клиентов, обслуженных, если грузовик стоит сейчас в i
    S[n + 1] = set()  # в фиктивной 0′ уже всё обслужено
    # move truck from i to j
    #for idx in reversed(range(len(truck_nodes) - 1)):  # im, …, i1, 0 from the end of the route to the beginning including the depot at the end
    for idx in range(len(truck_nodes) - 2, -1, -1):
        best_mt = None
        best_ll = None
        served_MT = set()
        served_LL = set()

        logger.debug(f"idx {idx}")  # i_idx индекс текущего узла в списке truck_nodes
        i = truck_nodes[idx]  # the truck can start moving from truck_nodes[idx]
        logger.debug(f"truck_nodes[{idx}]={i}")
        CMT = float('inf')  # Move Truck
        # CLL = float('inf')  # Launch and Land

        # MT
        logger.info(f"Start MT")
        CMT_best = float('inf')
        # all truck nodes after i
        i_full_idx = full_seq.index(i)  # позиция текущего truck-узла в full_seq
        logger.debug(f"current truck node in full seg: {i_full_idx}")
        d_idx = next((k for k in range(i_full_idx + 1, len(full_seq))
                      if node_types.get(full_seq[k]) == 'drone'),
                     len(full_seq))  # позиция d(i) либо len==0′
        # T()
        T_set = [u for u in truck_nodes  # именно T(i)
                 if pos[i] < pos[u] < d_idx]
        logger.debug(f"Tset: {T_set}")
        for j in T_set:
        #for j in truck_nodes[idx + 1:]:  # все truck-узлы правее
            # cand = S[j] | set()
            cand = S.get(j, set()).copy()
            logger.debug(f"truck_nodes[{idx + 1:}]={j}")
            # j – номер truck-узла-кандидата
            #if j == n + 1 and any(  # n+1 – это 0′
            #        t not in (n + 1,)  # пропустить сам 0′
            #        for t in truck_nodes[idx + 1:-1]  # все truck после i и ДО 0′
            #):
            #    continue  # ещё есть необслуженные truck, 0′ запрещён
            if j == n + 1 and truck_nodes[idx + 1:-1]:
                continue  # ещё есть необслуженные truck, 0′ запрещён
            # Если мы пытаемся сделать ход MT прямо в 0′, но справа от i остаются ещё непосещённые truck-клиенты, такой ход запрещаем
            #if full_seq[j] == full_seq[-1] and any(
            #        full_seq[k] in node_types and node_types[full_seq[k]] == 'truck'
            #        for k in range(i + 1, j)
            #):
            #    continue  # skip jump to the end ???

            # i, j - сами номера узлов, поэтому к матрицам времени обращаемся напрямую
            # t_time = truck_time[i][j]  # время пути грузовика i→j
            t_time = tau(i, j)  # время пути грузовика i→j
            # t_time = truck_time[full_seq[i]][full_seq[j]]
            logger.debug(f"time between {i} and {j} = {t_time}")
            logger.debug(f"C[{j}]={C.get(j, float('inf'))}")  # minimum time from j to the end of the route
            CMT = min(CMT, t_time + C.get(j, float('inf')))
            logger.debug(f"C_MT= {CMT}")

            if (CMT < CMT_best or
                    abs(t_time + C[j] - CMT_best) < 1e-9 and
                    len(cand) > len(served_MT)):
                CMT_best = CMT
                best_mt = ('MT', i, j, CMT)
                served_MT = cand  # cand может быть ∅

        # LL
        logger.info(f"Start LL")
        CLL_best_time = float('inf')
        CLL_best_action = None  # ('LL', i, deliver, k, cost)
        # find d(i)
        try:
            d_idx = next(j for j in range(i_full_idx + 1, len(full_seq))
                         if node_types.get(full_seq[j]) == 'drone')
        except StopIteration:
            # справа от i нет drone-узлов  ⇒  LL недоступен
            CLL_best_time = float('inf')
            CLL_best_action = None
            logger.debug("no drone after i → skip LL")
            d_idx = None

        if d_idx is not None:
            deliver = full_seq[d_idx]

        #d_idx = next(j for j in range(idx + 1, len(full_seq))
        #             if node_types.get(full_seq[j]) == 'drone')
        #deliver = full_seq[d_idx]  # номер узла d(i)
            logger.debug(f"d: {deliver}")
            # find d⁺(i) (next drone node after d(i))
            try:
                dplus_idx = next(j for j in range(d_idx + 1, len(full_seq))
                             if node_types.get(full_seq[j]) == 'drone')
            except StopIteration:
                dplus_idx = len(full_seq) - 1  # это 0′ в вашем full_seq
            logger.debug(f"next d: {dplus_idx}")
            # form E⁺(i)
            candidate_k = [u for u in truck_nodes
                           if d_idx < pos[u] <= dplus_idx
                           and (u != n + 1 or i == last_real_truck)]
            logger.debug(f"E+: {candidate_k}")
            # loop for k ∈ E⁺(i)
            CLL = float('inf')
            for k in candidate_k:
                # cand = S[k] | {deliver}
                cand = S.get(k, set()).union({deliver})
                drone_leg = drone_time[i][deliver]
                consec_penalty = 0.0
                cur_d = deliver
                while pos[cur_d] + 1 < pos[k] and node_types.get(full_seq[pos[cur_d] + 1]) == 'drone':
                    nxt = full_seq[pos[cur_d] + 1]
                    consec_penalty += w1 * drone_time[cur_d][nxt]
                    cur_d = nxt
                drone_leg += consec_penalty + drone_time[cur_d][k]
                # штраф за превышение дальности
                drone_leg_pen = drone_leg + w2 * max(0.0, drone_leg - drone_range)

                # d_flight = (drone_time[i][deliver] + drone_time[deliver][k])
                t_drive = tau(i, k)
                logger.debug(f"C[{k}]={C.get(k)}")
                total = max(drone_leg_pen, t_drive) + C.get(k, float('inf'))
                logger.debug(f"total: {total}")
                CLL = min(CLL, total)  # save the pair of nodes and the min time
                logger.debug(f"CLL={CLL}")

                if cand and (total < CLL_best_time or
                             abs(total - CLL_best_time) < 1e-9 and
                             len(cand) > len(served_LL)):
                    CLL_best_time = total
                    CLL_best_action = ('LL', i, deliver, k, total)
                    served_LL = cand

        CLL = CLL_best_time
        logger.debug(f"best time for LL: {CLL_best_time}")
        logger.debug(f"best action for LL: {CLL_best_action}")
        best_ll = CLL_best_action  # чтобы action_trace[i] был корректным, переопределяется внутри каждого truck-узла

        logger.info(f"best values: {best_mt}, {best_ll}")  # ('MT', 0, 2, 28.5), ('LL', 0, 1, 3, 356.98)
        # C[i] = min(CMT, CLL)
        logger.debug(f"compare: {CMT} vs {CLL}")

        if best_ll and (CLL < CMT or
                        abs(CLL - CMT) < 1e-9 and
                        len(served_LL) > len(served_MT)):
            C[i] = CLL
            S[i] = served_LL
            action_trace[i] = best_ll
        elif best_mt:
            C[i] = CMT
            S[i] = served_MT
            action_trace[i] = best_mt
        else:
            # ни MT, ни LL: узел недостижим — Type-1 infeasible
            C[i] = float('inf')
            # action_trace[i] НЕ заполняем
            continue                        # берём следующий i

    # calculate the optimal route
    if 0 not in action_trace or math.isinf(C[0]):
        return [], float('inf')  # хромосома infeasible-1

    route = []  # actions ('MT', i, k, cost) / ('LL', i, d, k, cost)
    cur = 0  # start with depo
    while cur != n + 1:  # till depo
        act = action_trace[cur]
        route.append(act)
        cur = act[2] if act[0] == 'MT' else act[3]  # next truck node k

    makespan = C[0]  # best makespan

    logger.debug(f"optimal route: {route}")
    logger.debug(f"best_makespan: {makespan}")
    return route, makespan


def visualize_route(places, route):
    # depo as centre
    m = folium.Map(location=places[0], zoom_start=14)
    n_last = len(places) - 1
    # marker for each node (as given at the beginning)
    for idx, (lat, lon) in enumerate(places):
        if idx == n_last:
            continue
        folium.Marker(
            [lat, lon],
            tooltip=f"node {idx}",
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f"""
                            <div style="background:blue; 
                                       color:white; 
                                       text-align:center; 
                                       border-radius:10px; 
                                       width:20px; height:20px;
                                       line-height:20px;">
                                       {idx}
                            </div>"""
            )
        ).add_to(m)

    truck_nodes = [0]  # start at the depo
    for act in route:
        next_node = act[2] if act[0] == "MT" else act[3]   # MT: j  |  LL: k
        truck_nodes.append(next_node)

    # if the route does not explicitly go to the depo -> add route to the depo
    if truck_nodes[-1] != len(places) - 1:
        truck_nodes.append(len(places) - 1)

    truck_points = [places[i] for i in truck_nodes]

    # truck route
    folium.PolyLine(
        truck_points,
        color="blue",
        weight=4,
        tooltip="Truck path"
    ).add_to(m)
    # labels for truck route
    for a, b in zip(truck_nodes, truck_nodes[1:]):
        folium.PolyLine(
            [places[a], places[b]],
            color="blue",
            weight=4,
            opacity=0,  # invisible line
            tooltip=f"Truck {a}->{b}"
        ).add_to(m)

    # drone route
    for action in route:
        if action[0] == "LL":
            _, launch, deliver, land, _ = action
            drone_points = [places[launch], places[deliver], places[land]]
            folium.PolyLine(
                drone_points,
                color="green",
                weight=2.5,
                dash_array="5,10",
                tooltip=f"Drone {launch}->{deliver}->{land}"
            ).add_to(m)

    m.save("route_map.html")
    return True


def generate_initial_population(n, size):
    base = list(range(1, n + 1))
    logger.debug(base)
    population = []
    for _ in range(size):
        chrom = random.sample(base, n)
        chrom = [g if random.random() > 0.3 else -g for g in chrom]
        population.append(chrom)
    logger.debug(f"init population: {population}")
    return population


def evaluate(chrom, truck_time, drone_time, drone_range):
    """Compute fitness of the chromosome.
    Returns fitness, feasibility and route"""
    route, cost = join_algorithm(chrom, truck_time, drone_time, drone_range)
    logger.debug(f"Route: {route}")
    logger.debug(f"cost: {cost}")

    if any(chrom[i] < 0 and chrom[i+1] < 0 for i in range(len(chrom)-1)):
        feas = 1                         # Type 1
    elif any(a[0]=='LL' and a[4] > drone_range for a in route):
        feas = 2                         # Type 2
    else:
        feas = 0                         # feasible
    logger.debug(f"route: {route}, feasibility: {feas}, fitness: {cost}")
    return cost, feas, route


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
    local_moves = [local_search_l1,
                   local_search_l2,
                   local_search_l3,
                   local_search_l4,
                   local_search_l5,
                   local_search_l6,
                   local_search_l7
                   ]  # todo more from N
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
    Choose three consecutive truck nodes and convert the middle one to a drone node.
    """
    # if depo and then 2 nodes are truck nodes, but we don't consider depo ??
    chrom = chromosome[:]
    n = len(chrom)
    # find three consecutive truck nodes
    for i in range(n - 2):
        if chrom[i] > 0 and chrom[i+1] > 0 and chrom[i+2] > 0:
            chrom[i+1] = -chrom[i+1]  # convert the middle one to a drone node
            return chrom
    return chrom


def local_search_l2(chromosome):
    """Remove a random drone node and randomly locate it between two consecutive truck nodes as a drone node.
    """
    drone_indices = [i for i, x in enumerate(chromosome) if x < 0]
    truck_positions = [i for i, x in enumerate(chromosome) if x > 0]
    if not drone_indices or len(truck_positions) < 2:
        return chromosome
    # drone node
    d_idx = random.choice(drone_indices)
    # remove drone node
    drone_node = chromosome.pop(d_idx)
    # insert drone node between truck nodes
    insert_pos = random.randint(1, len(chromosome) - 1)
    chromosome.insert(insert_pos, drone_node)
    return chromosome


def local_search_l3(chromosome):
    """Choose a truck node and a drone node randomly and swap them while keeping the type of each position.
    """
    truck_indices = [i for i, x in enumerate(chromosome) if x > 0]
    drone_indices = [i for i, x in enumerate(chromosome) if x < 0]
    if not truck_indices or not drone_indices:
        return chromosome
    # choose truck and drone node
    t_idx = random.choice(truck_indices)
    d_idx = random.choice(drone_indices)
    # swap
    chromosome[t_idx], chromosome[d_idx] = chromosome[d_idx], chromosome[t_idx]
    return chromosome


def local_search_l4(chromosome):
    """Randomly select two arcs from the truck tour and swap them.
    The truck sequence as well as the drone sequence between the two arcs will be reversed.
    """
    truck_positions = [i for i, x in enumerate(chromosome) if x > 0]
    if len(truck_positions) < 4:
        return chromosome
    # select two indices
    i1, i2 = sorted(random.sample(truck_positions, 2))
    # reverse arcs
    chromosome[i1:i2+1] = reversed(chromosome[i1:i2+1])
    return chromosome


def local_search_l5(chromosome):
    """Randomly choose two drone nodes and swap them while promoting their type to be truck nodes.
    """
    drone_indices = [i for i, x in enumerate(chromosome) if x < 0]
    if len(drone_indices) < 2:
        return chromosome
    # select two indices
    i1, i2 = sorted(random.sample(drone_indices, 2))
    # swap drone nodes and convert to truck nodes
    chromosome[i1], chromosome[i2] = abs(chromosome[i2]), abs(chromosome[i1])
    return chromosome


def local_search_l6(chromosome):
    """Randomly choose two drone nodes, swap them and convert one of them to truck node.
    """
    drone_indices = [i for i, x in enumerate(chromosome) if x < 0]
    if len(drone_indices) < 2:
        return chromosome
    # select two indices
    i1, i2 = random.sample(drone_indices, 2)
    # swap drone nodes
    chromosome[i1], chromosome[i2] = chromosome[i2], chromosome[i1]
    # convert one of them to truck with probability 50%
    if random.random() < 0.5:
        chromosome[i1] = abs(chromosome[i1])
    else:
        chromosome[i2] = abs(chromosome[i2])
    return chromosome


def local_search_l7(chromosome):
    """Randomly choose a drone node d and a drone tuple ⟨i,j,k⟩,
    change j to truck node and insert d as either ⟨i,d,j,k⟩ or ⟨i,j,d,k⟩.
    """
    chrom = chromosome[:]
    drone_indices = [i for i, g in enumerate(chrom) if g < 0]
    # must be at least 2 drones
    if len(drone_indices) < 2:
        return chrom

    # choose drone node d and remove it
    d_idx = random.choice(drone_indices)
    d_node = chrom[d_idx]
    chrom.pop(d_idx)

    # choose the second drone node j and convert it to truck
    remaining_drone_indices = [i for i, g in enumerate(chrom) if g < 0]
    if not remaining_drone_indices:
        chrom.insert(d_idx, d_node)  # if no drones anymore
        return chrom
    j_idx = random.choice(remaining_drone_indices)
    j_node = chrom[j_idx]
    chrom[j_idx] = abs(j_node)  # convert to truck

    # insert d either before j or after j
    if random.random() < 0.5:
        insert_pos = j_idx
    else:
        insert_pos = j_idx + 1
    chrom.insert(insert_pos, d_node)
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
    # repair always!
    for i, g in enumerate(chrom):
        if abs(g) in violating_nodes and g < 0:
            #if random.random() < p_repair:
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
                route, fitness = join_algorithm(child,
                                                truck_time_matrix,
                                                drone_time_matrix,
                                                drone_range)
                # fitness, feasible, route = evaluate(child, truck_time_matrix, drone_time_matrix, drone_range)

                feasible_pop.append((child, fitness))
                logger.debug(f"fitness: {fitness}, route: {route}")
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
    logger.debug(f"{best_solution}, {best_route}, {best_fitness}")
    return best_solution, best_route, best_fitness


if __name__ == "__main__":
    drone_speed = 20  # m/s
    truck_speed = 10
    drone_range = float('inf')

    places = [(50.149, 8.666),  # idx=0 = 6
              (50.148, 8.616),  # idx=1
              (50.130, 8.668),  # idx=2
              (50.146, 8.777),  # idx=3
              (50.160, 8.750),  # idx=4
              (50.164, 8.668),  # idx=5
              ]

    n = len(places)  # (без учёта 0′)
    logger.info(f"For {n} points.")
    chrom, route, fitness = genetic_algorithm(places)
    logger.info(f"Finally: chrom={chrom}, route={route}, fitness={fitness}")
    if route:
        visualize_route(places, route)
    else:
        logger.error(f"no optimal route")


