import random
from tsp import *


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



