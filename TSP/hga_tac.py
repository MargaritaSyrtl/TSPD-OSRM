import random


def compute_fitness(solution, waypoints_distances, drone_speed_ratio):
    """
    Вычисляет общее время доставки с учетом дрона и грузовика
    """
    truck_time = 0.0
    drone_time = 0.0

    for index in range(len(solution)):
        waypoint1 = solution[index - 1]
        waypoint2 = solution[index]

        if waypoint1 > 0 and waypoint2 > 0:  # Оба узла обслуживаются грузовиком
            truck_time += waypoints_distances[frozenset([waypoint1, waypoint2])]
        elif waypoint1 < 0 and waypoint2 < 0:  # Оба узла обслуживаются дроном
            drone_time += waypoints_distances[frozenset([abs(waypoint1), abs(waypoint2)])] / drone_speed_ratio
        else:  # Переход между грузовиком и дроном
            truck_time += waypoints_distances[frozenset([abs(waypoint1), abs(waypoint2)])]

            return max(truck_time, drone_time)  # Оптимизируем по максимальному времени

        def generate_random_agent(waypoints):
            """
        Генерирует случайный маршрут с учетом грузовика и дрона
        """
            first_node = waypoints[0]  # Фиксируем первый узел
            remaining_nodes = list(waypoints[1:])
            random.shuffle(remaining_nodes)

            # Примерное разделение на дрона и грузовик
            drone_nodes = {node if random.random() > 0.5 else -node for node in remaining_nodes}
            return tuple([first_node] + list(drone_nodes))

        def mutate_agent(agent_genome, max_mutations=3):
            """
        Точечная мутация, изменяющая маршрут
        """
            agent_genome = list(agent_genome)
            num_mutations = random.randint(1, max_mutations)

            for _ in range(num_mutations):
                swap_index1 = random.randint(1, len(agent_genome) - 1)
                swap_index2 = random.randint(1, len(agent_genome) - 1)

                if swap_index1 != swap_index2:
                    agent_genome[swap_index1], agent_genome[swap_index2] = agent_genome[swap_index2], agent_genome[
                        swap_index1]

            return tuple(agent_genome)

        def tox_crossover(parent1, parent2):
            """
        Type-Aware Order Crossover (TOX)
        """
            size = len(parent1)
            cut1, cut2 = sorted(random.sample(range(1, size), 2))
            offspring = [None] * size

            # Копируем сегмент из первого родителя
            offspring[cut1:cut2] = parent1[cut1:cut2]

            # Заполняем остальными узлами из второго родителя
            parent2_idx = 0
            for i in range(size):
                if offspring[i] is None:
                    while parent2[parent2_idx] in offspring:
                        parent2_idx += 1
                    offspring[i] = parent2[parent2_idx]

            return tuple(offspring)

        def generate_random_population(population_size, waypoints):
            """
        Создает начальную популяцию
        """
            return [generate_random_agent(waypoints) for _ in range(population_size)]

        def run_genetic_algorithm(places, waypoints_distances, generations, population_size, drone_speed_ratio):
            """
        Запуск HGA-TAC
        """
            population = generate_random_population(population_size, places)

            for generation in range(generations):
                population_fitness = {agent: compute_fitness(agent, waypoints_distances, drone_speed_ratio) for agent in
                                      population}

                sorted_population = sorted(population, key=population_fitness.get)
                new_population = []

                for parent1, parent2 in zip(sorted_population[:population_size // 2],
                                            sorted_population[1:population_size // 2]):
                    offspring1 = tox_crossover(parent1, parent2)
                    offspring2 = tox_crossover(parent2, parent1)

                    new_population.extend([mutate_agent(offspring1), mutate_agent(offspring2)])

                population = new_population[:population_size]  # Обновляем популяцию

            return sorted_population[0]  # Лучший найденный маршрут
