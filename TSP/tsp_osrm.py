import math
from loguru import logger
import folium
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
            cand = S[j] | set()
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
            candidate_k = [
                u for u in truck_nodes
                if d_idx < pos[u] <= dplus_idx  # O(1) вместо O(n)
                and (u != n + 1 or i == last_real_truck)
            ]
            logger.debug(f"E+: {candidate_k}")

            ########
            # candidate_k уже содержит { …, n+1 } при необходимости
            #if (n + 1) in candidate_k:
            #    still_unserved = any(
            #        # ни сам n+1, ни уже проеханные узлы i не считаются
            #        node_types.get(t) == 'truck' and t not in S.get(i, set())
            #        for t in truck_nodes[pos_truck[i] + 1: -1]  # все truck между i и 0′
             #   )
             #   if still_unserved:
             #       candidate_k.remove(n + 1)
            ######

            # loop for k ∈ E⁺(i)
            CLL = float('inf')
            for k in candidate_k:
                cand = S[k] | {deliver}
                d_flight = (drone_time[i][deliver] +
                        drone_time[deliver][k])
                if d_flight > drone_range:  # узел не в E⁺(i)
                    continue
                # t_drive = truck_time[i][k]
                t_drive = tau(i, k)
                logger.debug(f"C[{k}]={C.get(k)}")
                total = max(d_flight, t_drive) + C.get(k, float('inf'))
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
        else:
            C[i] = CMT
            S[i] = served_MT
            action_trace[i] = best_mt

    # calculate the optimal route
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


def visualize_route_old(places, route):
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


if __name__ == "__main__":
    drone_speed = 20  # m/s
    truck_speed = 10

    places = [(50.149, 8.666),  # idx=0 = 6
              (50.148, 8.616),  # idx=1
              (50.130, 8.668),  # idx=2
              (50.146, 8.777),  # idx=3
              (50.160, 8.750),  # idx=4
              (50.164, 8.668),  # idx=5
              # (50.177, 8.456),
              #(50.200, 8.600)
              ]

    places.append(places[0])  # depo

    generations = 1  # number of iterations
    population_size = 2  # number of agents

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

    logger.debug(f"truck_time_matrix {truck_time_matrix}")
    logger.debug(f"drone_time_matrix {drone_time_matrix}")

    chromosome1 = [4, -3, -2, 5, -1]

    optimal_route, makespan = join_algorithm(chromosome1, truck_time_matrix, drone_time_matrix, drone_range=float('inf'))
    logger.debug(optimal_route)
    logger.debug(makespan)
    visualize_route(places, optimal_route)
