import math
from loguru import logger
from itertools import combinations
import folium


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

    # кумулятивное время по truck-маршруту
    pos_truck = {node: idx for idx, node in enumerate(truck_nodes)}
    logger.debug(f"pos truck: {pos_truck}")
    prefix = [0.0]  # prefix[k] – совокупное время от депо до truck_nodes[k]
    for p, q in zip(truck_nodes, truck_nodes[1:]):
        prefix.append(prefix[-1] + truck_time[p][q])
    logger.debug(f"prefix: {prefix}")

    def tau(a, b):  # a,b ∈ truck_nodes,  pos_truck[a] < pos_truck[b]
        return prefix[pos_truck[b]] - prefix[pos_truck[a]]

    # DP
    # C[i] — minimum time from truck node i to the end
    C = {}
    C[n + 1] = 0  # virtual end of the route
    C[truck_nodes[-1]] = 0  # C(end) = 0
    logger.debug(f"C init {C}")

    logger.debug(len(truck_nodes) - 1)
    logger.debug(range(len(truck_nodes) - 1))
    all_options = []
    # move truck from i to j
    #for idx in reversed(range(len(truck_nodes) - 1)):  # im, …, i1, 0 from the end of the route to the beginning including the depot at the end
    for idx in range(len(truck_nodes) - 2, -1, -1):
        best_mt = None
        best_ll = None
        logger.debug(f"idx {idx}")  # i_idx индекс текущего узла в списке truck_nodes
        i = truck_nodes[idx]  # the truck can start moving from truck_nodes[idx]
        logger.debug(f"truck_nodes[{idx}]={i}")
        CMT = float('inf')  # Move Truck
        # CLL = float('inf')  # Launch and Land

        # MT
        logger.info(f"Start MT")
        CMT_best = float('inf')
        # all truck nodes after i

        for j in truck_nodes[idx + 1:]:  # все truck-узлы правее
            served_mt = {}
            logger.debug(f"truck_nodes[{idx + 1:}]={j}")
            # j – номер truck-узла-кандидата
            #if j == n + 1 and any(  # n+1 – это 0′
            #        t not in (n + 1,)  # пропустить сам 0′
            #        for t in truck_nodes[idx + 1:-1]  # все truck после i и ДО 0′
            #):
            #    continue  # ещё есть необслуженные truck, 0′ запрещён
            if j == n + 1 and truck_nodes[idx + 1:-1]:
                continue  # ещё есть необслуженные truck, 0′ запрещён
            # Если мы пытаемся сделать ход MT прямо в 0′, но справа от i остаются ещё непосещённые truck-клиенты, такой ход запрещаем («continue»).
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
            if CMT < CMT_best:
                CMT_best = CMT
                # best_mt = ('MT', full_seq[i], full_seq[j], CMT)
                logger.debug(f"best mt={best_mt}")  # ('MT', 0, 2, 14.25)
                best_mt = ('MT', i, j, CMT)
                # all_options.append((CMT, best_mt))
        if best_mt:
            all_options.append((CMT_best, best_mt))
            #served_mt[j] = CMT
            #logger.debug(f"served_mt={served_mt}")

        # LL
        logger.info(f"Start LL")
        CLL_best_time = float('inf')
        CLL_best_action = None  # ('LL', i, deliver, k, cost)

        i_full_idx = full_seq.index(i)  # позиция текущего truck-узла в full_seq
        logger.debug(f"current truck node in full seg: {i_full_idx}")
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
            ]
            logger.debug(f"E+: {candidate_k}")
            # loop for k ∈ E⁺(i)
            CLL = float('inf')
            for k in candidate_k:
                d_flight = (drone_time[i][deliver] +
                        drone_time[deliver][k])
                if d_flight > drone_range:  # узел не в E⁺(i)
                    continue
                # t_drive = truck_time[i][k]
                t_drive = tau(i, k)
                logger.debug(f"C[{k}]={C.get(k)}")
                total = max(d_flight, t_drive) + C.get(k, float('inf'))
                logger.debug(f"total: {total}")
                CLL = min(CLL, total)  # Храним и минимальное время, и саму пару узлов
                logger.debug(f"CLL={CLL}")
                if total < CLL_best_time:  # нашли лучший вариант
                    CLL_best_time = total
                    CLL_best_action = ('LL', i, deliver, k, total)

        CLL = CLL_best_time
        logger.debug(f"best time for LL: {CLL_best_time}")
        logger.debug(f"best action for LL: {CLL_best_action}")
        if CLL_best_action is not None:
            all_options.append((CLL, CLL_best_action))  # Попадает только один — действительно лучший LL для текущего i
        best_ll = CLL_best_action  # чтобы action_trace[i] был корректным, переопределяется внутри каждого truck-узла, поэтому всегда актуален

        logger.info(f"best values: {best_mt}, {best_ll}")  # ('MT', 0, 2, 28.5), ('LL', 0, 1, 3, 356.98)
        # C[i] = min(CMT, CLL)
        logger.debug(f"compare: {CMT} vs {CLL}")
        if CMT <= CLL:
            C[i] = CMT
            action_trace[i] = best_mt
        else:
            C[i] = CLL
            action_trace[i] = best_ll
        # logger.debug(f"C[{i}] at the end: {C[i]}")
        logger.debug(f"all_options: {all_options}")

    # calculate the optimal route
    # --- после завершения DP-цикла ------------------------------------------
    route = []  # список действий ('MT', i, k, cost)  или  ('LL', i, d, k, cost)
    cur = 0  # начинаем с депо
    while cur != n + 1:  # пока не дошли до 0′
        act = action_trace[cur]  # лучшее действие из узла cur
        route.append(act)
        cur = act[2] if act[0] == 'MT' else act[3]  # следующий truck-узел k

    makespan = C[0]  # оптимальный makespan уже посчитан

    logger.debug(f"optimal route: {route}")
    logger.debug(f"best_makespan: {makespan}")
    return route, makespan


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


if __name__ == "__main__":
    drone_speed = 20  # m/s
    truck_speed = 10

    places = [(50.149, 8.666),  # idx=0
              (50.145, 8.616),  # idx=1
              (50.147, 8.668),  # idx=2
              (50.146, 8.777),
              #(50.155, 7.777)
              ]

    places.append(places[0])  # depo
    # places.append((places[0][0] + 0.00001, places[0][1]))  # move depo by 1 meter

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

    chromosome1 = [1, -2, 3]

    optimal_route, makespan = join_algorithm(chromosome1, truck_time_matrix, drone_time_matrix, drone_range=float('inf'))
    logger.debug(optimal_route)
    logger.debug(makespan)
    visualize_route(places, optimal_route)

