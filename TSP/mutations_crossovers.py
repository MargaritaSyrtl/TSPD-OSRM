import random

def crossover(p1: tuple, p2: tuple, all_nodes: set | None = None, variant: str | None = None):
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
