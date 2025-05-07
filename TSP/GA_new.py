import random
from createHTML import create_optimal_route_html


def compute_fitness(solution, waypoints_distances, drone_speed_ratio=2.0):
    """
    Считает время маршрута с учётом грузовика и дрона:
    - 'truck_time' накапливается, когда узлы обслуживаются грузовиком.
    - 'drone_time' накапливается, когда узлы обслуживаются дроном.
    - speed_ratio дрона учитывается для ускорения (dist / drone_speed_ratio).
    Итог: fitness = max(truck_time, drone_time).
    """
    truck_time = 0.0
    drone_time = 0.0

    for index in range(len(solution)):
        waypoint1 = solution[index - 1]  # (node, type)
        waypoint2 = solution[index]  # (node, type)

        node1, type1 = waypoint1
        node2, type2 = waypoint2

        # Извлекаем расстояние из словаря, где ключ -- frozenset({node1, node2}).
        dist = waypoints_distances.get(frozenset([node1, node2]), 0.0)

        # Если оба узла обслуживает грузовик
        if type1 == 'truck' and type2 == 'truck':
            truck_time += dist
        # Если оба узла обслуживает дрон
        elif type1 == 'drone' and type2 == 'drone':
            drone_time += dist / drone_speed_ratio
        else:
            # Переход между грузовиком и дроном
            # Предположим, что грузовик продолжает движение
            truck_time += dist

    return max(truck_time, drone_time)


def generate_random_agent(waypoints):
    """
    Генерирует маршрут, где часть узлов обслуживается грузовиком ('truck'), а часть -- дроном ('drone').
    Первый узел (waypoints[0]) всегда 'truck'.
    """
    first_node = waypoints[0]
    remaining_nodes = list(waypoints[1:])
    random.shuffle(remaining_nodes)

    typed_nodes = []

    # для каждого узла случайно выбираем 'truck' или 'drone'
    for node in remaining_nodes:
        if random.random() < 0.5:
            typed_nodes.append((node, 'drone'))
        else:
            typed_nodes.append((node, 'truck'))

    # Первый узел делаем грузовиком
    # Возвращаем кортеж [(node0, 'truck'), (node1, 'drone'/'truck'), ...]
    return tuple([(first_node, 'truck')] + typed_nodes)


def mutate_agent(agent_genome, max_mutations=3):
    """
    Простая точечная мутация: перестановка узлов внутри одного и того же типа, чтобы
    не превращать 'truck' в 'drone' и наоборот.
    """
    if len(agent_genome) <= 2:
        return agent_genome

    genome_list = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)

    for _ in range(num_mutations):
        i1 = random.randint(1, len(genome_list) - 1)  # не трогаем индекс 0 (первый узел - truck)
        i2 = random.randint(1, len(genome_list) - 1)

        # Меняем местами элементы, только если у них одинаковый тип
        if i1 != i2:
            node1, type1 = genome_list[i1]
            node2, type2 = genome_list[i2]
            if type1 == type2:
                genome_list[i1], genome_list[i2] = genome_list[i2], genome_list[i1]

    return tuple(genome_list)


def shuffle_mutation(agent_genome):
    """
    Шаффл-мутация: берем случайный подотрезок и переносим в другое место.
    При этом не изменяем тип нулевого узла (индекс 0).
    """
    if len(agent_genome) <= 2:
        return agent_genome

    genome_list = list(agent_genome)

    start_index = random.randint(1, len(genome_list) - 1)
    length = random.randint(2, 20)
    subset = genome_list[start_index:start_index + length]

    # Удаляем подотрезок
    genome_list = genome_list[:start_index] + genome_list[start_index + length:]

    # Вставляем подотрезок в новое место (не затрагивая index=0)
    insert_index = random.randint(1, len(genome_list))
    genome_list = genome_list[:insert_index] + subset + genome_list[insert_index:]

    return tuple(genome_list)


def generate_random_population(population_size, waypoints):
    """
    Создает начальную популяцию типо-осведомленных агентов.
    """
    return [generate_random_agent(waypoints) for _ in range(population_size)]


def early_stopping(fitness_scores, threshold=0.01, patience=50):
    if len(fitness_scores) < patience:
        return False
    recent_scores = fitness_scores[-patience:]
    return max(recent_scores) - min(recent_scores) < threshold


def run_genetic_algorithm(
        places,
        waypoints_distances,
        generations,
        population_size,
        threshold=0.01,
        patience=50,
        drone_speed_ratio=2.0
):
    """
    Переработанный ГА:
    - хромосомы: [(node, 'truck'), (node, 'drone'), ...]
    - fitness считает max(truck_time, drone_time)
    - мутации, shuffle-мутации
    - без DP и локального поиска
    """
    fitness_scores = []
    current_best_fitness = float('inf')
    population_subset_size = int(population_size // 10)
    generations_10pct = int(generations // 10)
    current_best_genome = []

    # 1) Генерируем популяцию
    population = generate_random_population(population_size, places)

    # 2) Итерации
    for generation in range(generations):
        population_fitness = {}
        for agent_genome in population:
            if agent_genome not in population_fitness:
                population_fitness[agent_genome] = compute_fitness(
                    agent_genome,
                    waypoints_distances,
                    drone_speed_ratio
                )

        # Сортируем популяцию по возрастанию fitness
        sorted_population = sorted(population_fitness, key=population_fitness.get)

        new_population = []
        # Берем топ 10% и порождаем от них потомков
        for rank, agent_genome in enumerate(sorted_population[:population_subset_size]):
            # Сохраняем лучшего
            if (generation % generations_10pct == 0 or generation == generations - 1) and rank == 0:
                current_best_genome = agent_genome

            # Проверяем, не лучший ли это из всех, что мы видели
            if population_fitness[agent_genome] < current_best_fitness:
                current_best_fitness = population_fitness[agent_genome]
                # HTML вывод маршрута
                # create_optimal_route_html(agent_genome, current_best_fitness)

            # (1) Добавляем копию
            new_population.append(agent_genome)

            # (2) Две копии с обычной мутацией
            for _ in range(2):
                new_population.append(mutate_agent(agent_genome, 3))

            # (3) Семь копий с шаффл-мутацией
            for _ in range(7):
                new_population.append(shuffle_mutation(agent_genome))

        # Обрезаем, чтобы популяция оставалась нужного размера
        population = new_population[:population_size]

        # Запоминаем лучший fitness в этом поколении
        best_fitness_this_gen = min(population_fitness.values())
        fitness_scores.append(best_fitness_this_gen)

        # Проверка на раннюю остановку
        if early_stopping(fitness_scores, threshold, patience):
            break

    return current_best_genome
