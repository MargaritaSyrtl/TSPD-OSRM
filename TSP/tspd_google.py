from loguru import logger
import folium
from folium.features import DivIcon
from branca.element import Template, MacroElement
import random
import math
import subprocess, pathlib
from DMRequest_google import DMRequest
from pydantic_settings import BaseSettings
import time

class Settings(BaseSettings):
    api_key: str

    class Config:
        env_file = ".env"


settings = Settings()


def build_time_matrices_from_dm(places, drone_speed, dm_data):
    """
    truck_time - from Google
    drone_time - direct flight
    """

    distance_dict = dm_data['waypoints_distances']
    n = len(places) - 1
    truck_time = [[0] * (n + 2) for _ in range(n + 2)]
    drone_time = [[0] * (n + 2) for _ in range(n + 2)]

    for i in range(n + 1):
        for j in range(n + 1):
            dist = euclidean_distance(places[i], places[j])
            drone_time[i][j] = dist / drone_speed
            key = frozenset([places[i], places[j]])  # coordinates
            road_dist = distance_dict.get(key)  # meters
            if road_dist is not None:
                truck_time[i][j] = road_dist / truck_speed
            else:
                truck_time[i][j] = float('inf')

    return truck_time, drone_time


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
    all_nodes_to_serve = set(i for i in range(1, n + 1))
    logger.debug(all_nodes_to_serve)
    # truck nodes in order of succession in full_seq
    truck_nodes = [0]
    drone_nodes = []
    # for i in range(1, n + 1):  # n+1 virtual node -> 3=0
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
    last_real_truck = truck_nodes[-2]  # penultimate element, [-1] == n+1 (0′)

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

    def has_truck_between(a_idx, b_idx):
        return any(node_types.get(full_seq[t]) == 'truck'
                   for t in range(a_idx + 1, b_idx))

    # DP
    # C[i] — minimum time from truck node i to the end
    C = {}
    C[n + 1] = 0  # virtual end of the route
    C[truck_nodes[-1]] = 0  # C(end) = 0
    logger.debug(f"C init {C}")

    logger.debug(len(truck_nodes) - 1)
    logger.debug(range(len(truck_nodes) - 1))
    S = {}  # S[i] – the number of nodes served if the truck is now in i
    S[n + 1] = set()  # in the virtual 0' everything has already been served
    # move truck from i to j
    for idx in range(len(truck_nodes) - 2, -1, -1):
        best_mt = None
        best_ll = None
        served_MT = set()
        served_LL = set()

        logger.debug(f"idx {idx}")  # i_idx index of the current node in the truck_nodes
        i = truck_nodes[idx]  # the truck can start moving from truck_nodes[idx]
        logger.debug(f"truck_nodes[{idx}]={i}")
        CMT = float('inf')  # Move Truck
        # CLL = float('inf')  # Launch and Land

        # MT
        logger.info(f"Start MT")
        CMT_best = float('inf')
        # all truck nodes after i
        i_full_idx = full_seq.index(i)  # position of current truck node in full_seq
        logger.debug(f"current truck node in full seg: {i_full_idx}")
        d_idx = next((k for k in range(i_full_idx + 1, len(full_seq))
                      if node_types.get(full_seq[k]) == 'drone'),
                     len(full_seq))  # position d(i) or len==0′
        # T()
        T_set = [u for u in truck_nodes
                 if pos[i] < pos[u] < d_idx
                 and not has_truck_between(pos[i], pos[u])
                 ]
        logger.debug(f"Tset: {T_set}")
        for j in T_set:  # truck nodes to the right
            cand = S.get(j, set()).copy()
            cand.add(j)
            logger.debug(f"truck_nodes[{idx + 1:}]={j}")
            # j – truck node-candidate
            if j == n + 1 and truck_nodes[idx + 1:-1]:
                continue  # if there are unserviced nodes, 0′ forbidden

            # i, j - node numbers
            # t_time = truck_time[i][j]
            t_time = tau(i, j)  # truck travel time i→j
            logger.debug(f"time between {i} and {j} = {t_time}")
            logger.debug(f"C[{j}]={C.get(j, float('inf'))}")  # minimum time from j to the end of the route
            CMT = min(CMT, t_time + C.get(j, float('inf')))
            logger.debug(f"C_MT= {CMT}")

            if (CMT < CMT_best or
                    abs(t_time + C[j] - CMT_best) < 1e-9 and
                    len(cand) > len(served_MT)):
                CMT_best = CMT
                best_mt = ('MT', i, j, CMT)
                served_MT = cand

        # LL
        logger.info(f"Start LL")
        CLL_best_time = float('inf')
        CLL_best_action = None  # ('LL', i, deliver, k, cost)
        # find d(i)
        try:
            d_idx = next(j for j in range(i_full_idx + 1, len(full_seq))
                         if node_types.get(full_seq[j]) == 'drone')
        except StopIteration:
            # there is no drone nodes right from the i
            CLL_best_time = float('inf')
            CLL_best_action = None
            logger.debug("no drone after i → skip LL")
            d_idx = None

        # If there is at least one truck client between i and deliver,
        # LL from i is impossible - they must be served first
        if d_idx is not None and any(node_types.get(full_seq[t]) == "truck"
                                     for t in range(i_full_idx + 1, d_idx)):
            d_idx = None

        if d_idx is not None:
            deliver = full_seq[d_idx]
            logger.debug(f"d: {deliver}")

            # find d⁺(i) (next drone node after d(i))
            try:
                dplus_idx = next(j for j in range(d_idx + 1, len(full_seq))
                                 if node_types.get(full_seq[j]) == 'drone')
            except StopIteration:
                dplus_idx = len(full_seq) - 1  # 0′ in full_seq
            logger.debug(f"next d: {dplus_idx}")

            # form E⁺(i)
            candidate_k = [u for u in truck_nodes
                           if d_idx < pos[u] <= dplus_idx
                           and not has_truck_between(d_idx, pos[u])
                           and (u != n + 1 or i == last_real_truck)]
            logger.debug(f"E+: {candidate_k}")
            # loop for k ∈ E⁺(i)
            CLL = float('inf')
            for k in candidate_k:
                cand = S.get(k, set()).union({deliver})
                cand.add(k)
                drone_leg = drone_time[i][deliver]
                consec_penalty = 0.0
                cur_d = deliver
                # penalties
                # w1 if consecutive drone nodes
                while pos[cur_d] + 1 < pos[k] and node_types.get(full_seq[pos[cur_d] + 1]) == 'drone':
                    nxt = full_seq[pos[cur_d] + 1]
                    consec_penalty += w1 * drone_time[cur_d][nxt]
                    cur_d = nxt
                drone_leg += consec_penalty + drone_time[cur_d][k]
                # w2 if flight range exceeded
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
        best_ll = CLL_best_action

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
            # Type-1 infeasible
            C[i] = float('inf')
            continue  # next i

    # calculate the optimal route
    if 0 not in action_trace or math.isinf(C[0]):
        return [], float('inf'), 0  # Type-1 infeasible

    total_time = 0.0
    route = []  # actions ('MT', i, k, cost) / ('LL', i, d, k, cost)
    cur = 0  # start with depo
    while cur != n + 1:  # till depo
        act = action_trace[cur]
        route.append(act)
        if act[0] == 'MT':
            cur = act[2]
            total_time += act[3]
        else:
            cur = act[3]  # next truck node k
            total_time += act[4]

    makespan = C[0]  # best makespan

    logger.debug(f"optimal route: {route}")
    logger.debug(f"best_makespan: {makespan}")
    logger.debug(f"total time for the route: {total_time}")
    return route, makespan, total_time


def visualize_route(places, route, fitness, dm_data):

    places = places[:]  # copy
    places.append(places[0])
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
        next_node = act[2] if act[0] == "MT" else act[3]  # MT: j  |  LL: k
        truck_nodes.append(next_node)

    # if the route does not explicitly go to the depo -> add route to the depo
    if truck_nodes[-1] != len(places) - 1:
        truck_nodes.append(len(places) - 1)

    geom_dict = dm_data['waypoints_geometries']

    for a, b in zip(truck_nodes, truck_nodes[1:]):
        key = frozenset([places[a], places[b]])
        path = geom_dict.get(key, [places[a], places[b]])
        #folium.PolyLine(
        #    path,
        #    color="blue",
        #    weight=4,
        #    tooltip=f"Truck {a}→{b}"
        #).add_to(m)

        ratio = dm_data["waypoints_traffic"].get(key, 1.0)
        if ratio < 1.1:
            color = "green"  # green
        elif ratio < 1.5:
            color = "#FFC107"  # orange
        else:
            color = "#E74C3C"  # red
        folium.PolyLine(
            path,
            color=color,
            weight=4,
            tooltip=f"Truck {a}→{b}, traffic ratio: {ratio:.2f}"
        ).add_to(m)

    # drone route
    for action in route:
        if action[0] == "LL":
            _, launch, deliver, land, _ = action
            drone_points = [places[launch], places[deliver], places[land]]
            folium.PolyLine(
                drone_points,
                color="#1E90FF",
                weight=2.5,
                dash_array="5,10",
                tooltip=f"Drone {launch}->{deliver}->{land}"
            ).add_to(m)

    # create legend with route
    # truck-route in chronological order
    truck_nodes = [0]
    for act in route:
        nxt = act[2] if act[0] == "MT" else act[3]  # MT: j  |  LL: k
        if nxt != truck_nodes[-1]:  # remove duplicates
            truck_nodes.append(nxt)
    depot_ids = {0, len(places) - 1}
    if truck_nodes[-1] not in depot_ids:
        truck_nodes.append(0)

    # drones: concatenate all "launch-deliver-land", remove consecutive-repetitions
    drone_nodes = []

    def _push(n):
        if not drone_nodes or drone_nodes[-1] != n:
            drone_nodes.append(n)

    for act in route:
        if act[0] == "LL":
            _, launch, deliver, land, _ = act
            _push(launch)
            _push(deliver)
            _push(land)

    # convert total time to hours with minutes and seconds
    hours, remainder = divmod(fitness, 3600)
    minutes, seconds = divmod(remainder, 60)
    # hours, minutes – int; seconds not int
    if hours > 0:
        time = f"{int(hours)}h:{int(minutes)}min"
    else:
        time = f"{int(minutes)}min"

    # HTML
    truck_str = " → ".join("0" if n == n_last else str(n) for n in truck_nodes)
    drone_str = " → ".join("0" if n == n_last else str(n) for n in drone_nodes)
    legend_html = f"""
        <div style="
             position: fixed;
             bottom: 30px; right: 30px;
             z-index: 9999;
             background: rgba(255,255,255,0.9);
             padding: 10px 14px;
             border: 2px solid #999;
             border-radius: 6px;
             box-shadow: 3px 3px 6px rgba(0,0,0,0.25);
             font-size: 14px; line-height: 1.5;">
          <b>Optimal route:&nbsp;</b><br>
          <span style="color:#000000; font-weight:600;">
            Truck route: &nbsp;{truck_str}
          </span><br>
          <span style="color:#000000; font-weight:600;">
            Drone route: &nbsp;{drone_str}
          </span><br>
          <span style="color:#000000; font-weight:600;">
            Travel time: &nbsp;{time}
          </span>
        </div>"""

    macro = MacroElement()
    macro._template = Template(f"{{% macro html(this, kwargs) %}}{legend_html}{{% endmacro %}}")
    m.get_root().add_child(macro)

    m.save("route_google.html")
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
    route, cost, total_time = join_algorithm(chrom, truck_time, drone_time, drone_range)
    logger.debug(f"Route: {route}")
    logger.debug(f"cost: {cost}")

    if any(chrom[i] < 0 and chrom[i + 1] < 0 for i in range(len(chrom) - 1)):
        feas = 1  # Type 1
    elif any(a[0] == 'LL' and a[4] > drone_range for a in route):
        feas = 2  # Type 2
    else:
        feas = 0  # feasible
    logger.debug(f"route: {route}, feasibility: {feas}, fitness: {cost}, total_time: {total_time}")
    return cost, feas, route, total_time


def tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range, k=5):
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
    segment = [g for g in parent1[i1:i2 + 1] if type_of_node(g) == segment_type]
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
    segment = parent1[i1:i2 + 1]
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
    best_cost, _, _, total_time = evaluate(best_chrom, truck_time, drone_time, drone_range)
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
        new_cost, _, _, new_total_time = evaluate(new_chrom, truck_time, drone_time, drone_range)
        if new_cost < best_cost:
            best_cost = new_cost
            best_chrom = new_chrom
            improved = True
            total_time = new_total_time
        attempts += 1
    logger.debug(f"chrom was: {chromosome}")
    logger.debug(f"chrom ist: {best_chrom}")
    return best_chrom, best_cost, total_time


def local_search_l1(chromosome):
    """
    Choose three consecutive truck nodes and convert the middle one to a drone node.
    """
    # if depo and then 2 nodes are truck nodes, but we don't consider depo ??
    chrom = chromosome[:]
    n = len(chrom)
    # find three consecutive truck nodes
    for i in range(n - 2):
        if chrom[i] > 0 and chrom[i + 1] > 0 and chrom[i + 2] > 0:
            chrom[i + 1] = -chrom[i + 1]  # convert the middle one to a drone node
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
    chromosome[i1:i2 + 1] = reversed(chromosome[i1:i2 + 1])
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
    route, _, _ = join_algorithm(chrom, truck_time, drone_time, drone_range)
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
            # if random.random() < p_repair:
            chrom[i] = abs(g)
    logger.debug(f"repaired chromosome: {chromosome}")
    return chrom


def genetic_algorithm(places, drone_range, generations, population_size, mu_value, ItNI, truck_speed, drone_speed, dm_data):
    with open("mutations.txt", "w", encoding="utf-8") as file:
        # init TSP with LKH
        tsp_file = write_tsplib(places, name="demo", fname="demo.tsp")
        par_file = write_par(tsp_file, "demo.par", runs=1)
        # lkh_tour = run_lkh(par_file, exe_path="/usr/local/bin/LKH")
        lkh_tour = run_lkh(par_file, exe_path="LKH")  # in TSP/
        # logger.info(f"LKH tour: {lkh_tour}")
        tsp_tour = rotate_to_start(lkh_tour, start=0)
        # init TSP tour: [0, 5, 1, 7, 6, 9, 2, 3, 4, 8]
        file.write(f"init TSP tour: {tsp_tour}\n")

        feasible_pop = []
        infeasible_1_pop = []
        infeasible_2_pop = []

        # init TSPD with heuristics -> needs to be exact partition
        places = places[:]  # copy
        places.append(places[0])
        n = len(places) - 1
        logger.info(f"For {n} points.")
        logger.info(f"{places}")


#        # init time matrix
#        truck_time_matrix = [[0] * (n + 2) for _ in range(n + 2)]
#        drone_time_matrix = [[0] * (n + 2) for _ in range(n + 2)]
#        for i in range(n + 1):
#            for j in range(n + 1):
#                dist = euclidean_distance(places[i], places[j])
#                truck_time_matrix[i][j] = dist / truck_speed
#                drone_time_matrix[i][j] = dist / drone_speed
        truck_time_matrix, drone_time_matrix = build_time_matrices_from_dm(places, drone_speed, dm_data)

        # form ω0
        omega0 = partition_tsp_to_tspd(tsp_tour, truck_time_matrix, drone_time_matrix, drone_range)
        file.write(f"omega: {omega0}\n")
        # generate subpopulations
        subpops = generate_initial_population_from_tac(omega0, mu_value, truck_time_matrix, drone_time_matrix,
                                                       drone_range)
        file.write(f"subpops: {subpops}\n")
        # merge subpopulations into one list
        population = [ch for lst in subpops.values() for ch, _ in lst]  # init population
        file.write(f"population: {population}\n")

        fitnesses = []
        best_fitness = float('inf')  # min makespan
        best_solution = None  # chromosome that gave the best result
        best_route = None  # list of actions (MT/LL) for the best chromosome
        improved = False
        best_total_time = 0
        no_improve_count = 0

        # Evaluate initial population
        for chrom in population:
            logger.debug(f"chrom: {chrom}")
            file.write(f"chrom: {chrom}\n")
            fit, feas, route, total_time = evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range)
            fitnesses.append(fit)
            # set best fitness for the init population
            if feas == 0 and fit < best_fitness:
                best_fitness = fit
                best_solution = chrom
                best_route = route
                best_total_time = total_time
        file.write(f"init fitnesses: {str(fitnesses)}\n")

        # Generate and evaluate offsprings
        for g in range(generations):

            # improved = False  # ??here
            # tournament selection
            p1 = tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range)
            logger.info(f"first parent: {p1}")  # [3, -1, 2]
            p2 = tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range)
            while p1 == p2:
                p2 = tournament_selection(population, fitnesses, truck_time_matrix, drone_time_matrix, drone_range)
            logger.info(f"second parent: {p2}")  # [2, -1, 3]
            file.write(f"parent1: {str(p1)}\n")
            file.write(f"parent2: {str(p2)}\n")
            # if still p1==p2 ?

            # crossover -> new child
            child = tox1(p1, p2) if random.random() < 0.5 else tox2(p1, p2)
            logger.debug(f"child: {child}")  # child: [-1, 2, -3]
            file.write(f"child after crossover: {str(child)}\n")
            # mutation
            if random.random() < 0.5:
                child = sign_mutation(child)
            else:
                child = tour_mutation(child)
            logger.debug(f"child after mutation: {child}")  # child after mutation: [-1, 2, 3]
            file.write(f"child after mutation: {str(child)}\n")

            # evaluate child after all mutations
            fitness, feasible, route, total_time = evaluate(child, truck_time_matrix, drone_time_matrix, drone_range)
            # if infeasible -> repair
            if feasible in [1, 2]:
                child = repair(child, truck_time_matrix, drone_time_matrix, drone_range, p_repair=0.5)
                fitness, feasible, route, total_time = evaluate(child, truck_time_matrix, drone_time_matrix, drone_range)
                # if infeasible -> penalty
                if feasible in [1, 2]:
                    fitness += 99999  # todo adjust penalties
            # if feasible
            if feasible == 0:
                # local search only for feasible chrom
                child, fitness, total_time = local_search(child, truck_time_matrix, drone_time_matrix, drone_range)
                logger.debug(f"child after local search: {child}")
                file.write(f"child after local search: {str(child)}\n")
                fitness, feasible, route, total_time = evaluate(child, truck_time_matrix, drone_time_matrix, drone_range)

                feasible_pop.append((child, fitness))
                logger.debug(f"fitness: {fitness}, route: {route}")
                # save the best solutions
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = child
                    best_route = route
                    improved = True
                    best_total_time = total_time

            # if infeasible
            if feasible == 1:
                infeasible_1_pop.append((child, fitness))
            if feasible == 2:
                infeasible_2_pop.append((child, fitness))

            # update population
            population.append(child)
            population = sorted(population,
                                key=lambda chrom: evaluate(chrom, truck_time_matrix, drone_time_matrix, drone_range)[0])
            population = population[:population_size]  # keep best

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if no_improve_count >= ItNI:
                break

    logger.debug(f"{best_solution}, {best_route}, {best_fitness}, {best_total_time}")

    return best_solution, best_route, best_fitness, best_total_time


def run_lkh(par_file, exe_path="./LKH"):
    proc = subprocess.run([exe_path, par_file],
                          stdout=subprocess.PIPE,
                          text=True, check=True)
    logger.info(proc.stdout)
    tour = read_tour("result.tour")
    return tour


def read_tour(tour_file):
    """
    Reads SECTION in form:
    TOUR_SECTION
    1 4 3 2 5 6 0
    -1
    EOF
    """
    lines = pathlib.Path(tour_file).read_text().splitlines()
    start = lines.index("TOUR_SECTION") + 1
    seq = []
    for s in lines[start:]:
        v = int(s.split()[0])
        if v == -1: break
        seq.append(v - 1)  # TSPLIB numbers from 1
    return seq


def write_par(tsp_file, par_file="problem.par", runs=10, time_limit=30):
    txt = f"""PROBLEM_FILE = {tsp_file}
OUTPUT_TOUR_FILE = result.tour
RUNS = {runs}
TIME_LIMIT = {time_limit}
"""
    pathlib.Path(par_file).write_text(txt)
    return par_file


def haversine(a, b):
    R = 6371000
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return round(
        2 * R * math.asin(
            math.sqrt(
                math.sin(dlat / 2) ** 2 +
                math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            )))


def write_tsplib(coords, name="demo", fname="demo.tsp"):
    n = len(coords)
    lines = [
        f"NAME : {name}",
        "TYPE : TSP",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        "EDGE_WEIGHT_SECTION"
    ]
    for i in range(n):
        lines.append(" ".join(str(haversine(coords[i], coords[j]))
                              for j in range(n)))
    lines.append("EOF")
    pathlib.Path(fname).write_text("\n".join(lines))
    return fname


def rotate_to_start(tour, start=0):
    k = tour.index(start)
    return tour[k:] + tour[:k]


# Exact-partition as simple heuristics
def partition_tsp_to_tspd(tour, truck_time, drone_time, drone_range, k_drone=2, w1=2.0, w2=2.0):
    """
    tour: without 0′
    each k-th client → drone if the total depot-deliver-next ≤ drone_range
    return: Type-Aware Chromosome (TAC) without depot and 0′ node
    """
    tac = []
    last_was_drone = False
    for idx, node in enumerate(tour[1:]):
        is_drone_candidate = (idx + 1) % k_drone == 0 and not last_was_drone
        if is_drone_candidate:
            prev_n = tour[idx]
            next_n = tour[idx + 2] if idx + 2 < len(tour) else tour[0]
            flight = (drone_time[prev_n][node] + drone_time[node][next_n])
            penalty = w1 * 0 + w2 * max(0, flight - drone_range)
            if flight + penalty <= drone_range:
                tac.append(-node)  # create a drone node
                last_was_drone = True
                continue
        tac.append(+node)  # leave the node as truck
        last_was_drone = False
    return tac  # omega0 in article


# To diversify the population and explore the solution space effectively,
# new individuals are generated by applying specific modifying operators to existing solutions
def element_wise_mod(chrom, p_flip=0.1, p_swap=0.1):
    """Random modification of individual elements within a chromosome
    """
    chrom = chrom[:]  # copy
    for i in range(len(chrom)):
        r = random.random()
        if r < p_flip:
            chrom[i] = -chrom[i]  # change type
        elif r < p_flip + p_swap and i > 0:
            chrom[i], chrom[i - 1] = chrom[i - 1], chrom[i]  # swap
    return chrom


def sequence_mod(chrom):
    """Modifying subsequences within the chromosome
    """
    chrom = chrom[:]
    i1, i2 = sorted(random.sample(range(len(chrom)), 2))
    mode = random.choice(["reverse", "flip", "shuffle"])
    seg = chrom[i1:i2 + 1]
    if mode == "reverse":
        seg = list(reversed(seg))
    elif mode == "flip":
        seg = [-g for g in seg]
    else:  # shuffle
        random.shuffle(seg)
    chrom[i1:i2 + 1] = seg
    return chrom


def classify(route, fitness, drone_range, chrom):
    """0 – feasible, 1 – type-1, 2 – type-2"""
    if fitness == float('inf'):
        return 1
    if any(a[0] == 'LL' and a[4] > drone_range for a in route):
        return 2
    if any(chrom[i] < 0 and chrom[i + 1] < 0 for i in range(len(chrom) - 1)):
        return 1
    return 0


def generate_initial_population_from_tac(omega0, µ, truck_time, drone_time, drone_range):
    """
    Forms three subpopulations Ωᶠ, Ω¹_inf, Ω²_inf (feasible, Type 1, Type 2).
    First compute ω₀, then "modifies" existing chromosomes until each subpopulation contains µ individuals.
    Returns subpops: {0:[(chrom,cost), ...], 1:[...], 2:[...]}"""
    subpops = {0: [], 1: [], 2: []}

    def push(ch, cost, typ):
        subpops[typ].append((ch, cost))

    # ω0
    r0, c0, t0 = join_algorithm(omega0, truck_time, drone_time, drone_range)
    push(omega0, c0, classify(r0, c0, drone_range, omega0))

    # generate until there are µ individuals in each subpopulation
    while any(len(lst) < µ for lst in subpops.values()):
        base = random.choice(sum(subpops.values(), []))[0]
        child = element_wise_mod(base)
        if random.random() < 0.5:
            child = sequence_mod(child)
        # decode chromosome with join
        route, cost, total_time = join_algorithm(child, truck_time, drone_time, drone_range)
        # -> feas, infeas1, infeas2
        typ = classify(route, cost, drone_range, child)
        # add to the corresponding subpopulation
        push(child, cost, typ)
    return subpops


if __name__ == "__main__":
    start = time.time()

    truck_speed = 10  # m/s
    drone_speed = 2 * truck_speed
    drone_range = 3000  # m
    # parameters:
    # mu_value - min size of each subpop = 15
    # lambda_value -  "offspring pool" (added on top of µ before "survivor selection") = 25
    # ItNI - number of iterations with no improvements for stopping the GA = 2500
    # generations - number of iterations in GA
    mu_value = 15
    lambda_value = 25
    population_size = mu_value + lambda_value
    ItNI = 100
    # controls the search depth of one GA run
    # how many times it will generate and select new generations in an attempt to improve solutions
    generations = 600

    places = [(50.08907396096527, 8.670714912636585),   # 0
              (50.12413060964201, 8.607552521857166),   # 1
              (50.13104153062146, 8.716872044360008),   # 2
              (50.10572906849683, 8.757866865298572),   # 3
              (50.114456044438604, 8.675334053958041),  # 4
              (50.10392126972894, 8.631731759275732),   # 5
              (50.139217216992726, 8.676705648840892),  # 6
              (50.121404901636176, 8.66130128708773),   # 7
              (50.102612763576104, 8.6767882194423),    # 8
              (50.12705083884542, 8.692123319126726),   # 9
              (50.12705083884542, 8.692123319126726)    # 10
              ]
    n = len(places)  # without 0′
    logger.info(f"For {n} points.")
    dm = DMRequest(places, settings.api_key)
    dm_data = dm.get_response_data_ga()

    # chrom, route, fitness = genetic_algorithm(places, drone_range, generations,
    # population_size, mu_value, ItNI truck_speed, drone_speed)
    # logger.info(f"Finally: chrom={chrom}, route={route}, fitness={fitness}")
    # visualize_route(places, route)

    best_route = None
    best_fitness = float('inf')
    list_of_fitnesses = []
    # used to start the GA multiple times to increase the chances of finding a good solution,
    # since the GA is stochastic (random) and with different initial populations it can come to different solutions
    for i in range(0, 3):
        chrom, route, fitness, total_time = genetic_algorithm(places, drone_range, generations,
                                                  population_size, mu_value, ItNI,
                                                  truck_speed, drone_speed, dm_data)
        logger.info(f"Finally: chrom={chrom}, route={route}, fitness={fitness}")
        list_of_fitnesses.append(fitness)
        if fitness < best_fitness:
            best_fitness = fitness
            best_route = route
            visualize_route(places, best_route, best_fitness, dm_data)

    # convert total time to hours with minutes and seconds
    hours, remainder = divmod(best_fitness, 3600)
    minutes, seconds = divmod(remainder, 60)
    # hours, minutes – int; seconds not int
    time_str = f"{int(hours):02d}:{int(minutes):02d}"
    
    logger.debug(f"fitness: {best_fitness}, route {best_route}, time: {time_str}")
    logger.debug(f"list of fitness: {list_of_fitnesses}")
    end = time.time()
    elapsed = int(end - start)
    if elapsed < 60:
        logger.info(f"Running time: {elapsed}sec")
    else:
        minutes = elapsed // 60
        seconds = elapsed % 60
        logger.info(f"Running time: {minutes}min {seconds}sec")


