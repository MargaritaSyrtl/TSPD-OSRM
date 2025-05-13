import math
from loguru import logger


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
    n = len(chromosome)
    logger.debug(chromosome)
    full_seq = [0] + [abs(x) for x in chromosome] + [n + 1]  # virtual end node
    logger.debug(f"full_seq {full_seq}")

    node_types = {abs(x): 'truck' if x >= 0 else 'drone' for x in chromosome}
    logger.debug(f"nodes_types {node_types}")
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
    #
    # move truck from i to j
    for idx in reversed(range(len(truck_nodes) - 1)):  # from the end of the route to the beginning including the depot at the end
        logger.debug(f"idx {idx}")
        i = truck_nodes[idx]  # the truck can start moving from truck_nodes[idx]
        logger.debug(f"truck_nodes[{idx}]={i}")
        CMT = float('inf')  # Move Truck
        CLL = float('inf')  # Launch and Land

        # MT
        logger.info(f"Start MT")
        # all truck nodes after i
        for j in truck_nodes[idx + 1:]:
            logger.debug(f"truck_nodes[{idx + 1:}]={j}")
            if full_seq[j] == full_seq[-1] and any(
                    full_seq[k] in node_types and node_types[full_seq[k]] == 'truck'
                    for k in range(i + 1, j)
            ):
                continue  # skip jump to end if there are unvisited truck nodes between i and j

            t_time = truck_time[full_seq[i]][full_seq[j]]
            logger.debug(f"time between {i} and {j} = {t_time}")
            logger.debug(f"C[{j}]={C.get(j, float('inf'))}")  # minimum time from j to the end of the route

            CMT = min(CMT, t_time + C.get(j, float('inf')))
            logger.debug(f"C_MT= {CMT}")

        # LL
        logger.info(f"Start LL")
        # the first drone node after i
        for d in range(i + 1, len(full_seq) - 1):
            logger.debug(f"d={d}")
            if node_types.get(d, '') != 'drone':
                continue

            deliver = full_seq[d]
            logger.debug(f"deliver {deliver}")
            # land nodes - all truck stations after delivery
            for k in truck_nodes:
                if k <= d:
                    continue
                land = full_seq[k]
                logger.debug(f"land {land}")

                logger.debug(drone_time[full_seq[i]][deliver])
                logger.debug(drone_time[deliver][land])
                d_flight = drone_time[full_seq[i]][deliver] + drone_time[deliver][land]
                logger.debug(f"d flight {d_flight}")
                t_drive = truck_time[full_seq[i]][land]
                logger.debug(f"t drive {t_drive}")

                if d_flight <= drone_range:
                    logger.debug(f"C[{k}]={C.get(k)}")
                    total = max(d_flight, t_drive) + C.get(k, float('inf'))
                    logger.debug(f"total={total}")
                    CLL = min(CLL, total)
                    logger.debug(f"CLL={CLL}")
            break  # только первый drone после i, как в статье??

        C[i] = min(CMT, CLL)
        logger.debug(f"C[{i}] at the end: {C[i]}")

    # todo все узлы должны обслуживаться
    return C[0]


if __name__ == "__main__":
    drone_speed = 20  # m/s
    truck_speed = 10

    places = [(50.147, 8.666),  # idx=0
              (50.145, 8.616),  # idx=1
              (50.147, 8.668),  # idx=2
              ]

    places.append(places[0])   # depo

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

    chromosome = [-1, 2]
    #chromosome = [{(50.147, 8.666): "truck",
    #               (50.145, 8.616): "truck",
    #               (50.147, 8.668): "drone"}]

    makespan = join_algorithm(chromosome, truck_time_matrix, drone_time_matrix, drone_range=float('inf'))
    logger.debug(f"makespan: {makespan}")



