import random
#from createHTML import create_optimal_route_html, get_route_from_ranking


def compute_fitness_type_aware(solution, distances_truck, distances_drone):
    """
    Berechnet die Fitness eines Chromosoms, das aus (Stadt, Typ)-Tupeln besteht.
    Nutzt unterschiedliche Distanz-Tabellen für Truck vs. Drone.
    Je kleiner der Fitness-Wert, desto besser (hier repräsentiert er die Gesamtdistanz).
    """
    total_distance = 0.0
    n = len(solution)

    for i in range(n):
        # Nimm das aktuelle Gen
        current_city, current_type = solution[i]
        # Nimm das vorige Gen (Kreis-Tour, also -1 => letztes Element)
        prev_city, prev_type = solution[i - 1]

        # Entscheide, welche Distanzmatrix verwendet wird:
        # Wir könnten entweder annehmen, dass für den Sprung i->i-1
        # der Typ des aktuellen Ortes oder des vorherigen Ortes zählt.
        # In manchen TSP-Drone-Formulierungen nimmt man nur den Typ des
        # "fahrenden" Fahrzeugs. Hier ein simples Beispiel:
        # Falls current_type = "Drone", wird distances_drone genommen.
        # Sonst distances_truck. (Man kann natürlich auch beide kombinieren o.Ä.)
        if current_type == "Drone":
            total_distance += distances_drone.get(frozenset([prev_city, current_city]), 999999)
        else:
            total_distance += distances_truck.get(frozenset([prev_city, current_city]), 999999)

    return total_distance


def generate_random_agent_type_aware(waypoints):
    """
    Erzeugt einen zufälligen 'type-aware' Chromosomen:
    - Jede Stadt: (StadtIndex, Typ)
    - Erster Knoten bleibt fix 'Truck' (optional), damit er z.B. unser Startpunkt ist.
    """
    # Variante: Du fixierst den ersten Knoten als (waypoints[0], "Truck").
    # Den Rest mischt du inkl. zufälligem Type.
    first_node = (waypoints[0], "Truck")
    remaining_nodes = list(waypoints[1:])
    random.shuffle(remaining_nodes)

    agent = [first_node]
    for city in remaining_nodes:
        # Weise per Zufall "Truck" oder "Drone" zu
        vehicle_type = random.choice(["Truck", "Drone"])
        agent.append((city, vehicle_type))

    return tuple(agent)


def mutate_agent_type_aware(agent_genome, max_mutations=3, flip_type_probability=0.3):
    """
    Mutation für 'type-aware' Chromosomen:
      1. Tausche gelegentlich die Reihenfolge zweier Knoten (außer dem ersten).
      2. Flippe mit gewisser Wahrscheinlichkeit den Typ (Truck <-> Drone).
    """
    if len(agent_genome) <= 2:
        return agent_genome

    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)

    for _ in range(num_mutations):
        # 1. Swap
        swap_index1 = random.randint(1, len(agent_genome) - 1)
        swap_index2 = swap_index1
        while swap_index2 == swap_index1:
            swap_index2 = random.randint(1, len(agent_genome) - 1)

        agent_genome[swap_index1], agent_genome[swap_index2] = (
            agent_genome[swap_index2],
            agent_genome[swap_index1],
        )

        # 2. Flip type mit gewisser Wahrscheinlichkeit
        for i in range(1, len(agent_genome)):
            if random.random() < flip_type_probability:
                city, t = agent_genome[i]
                new_type = "Drone" if t == "Truck" else "Truck"
                agent_genome[i] = (city, new_type)

    return tuple(agent_genome)


def shuffle_mutation_type_aware(agent_genome):
    """
    'Shuffle Mutation' für den TSP-Weg (exklusive des ersten Knotens).
    Erweitert für Type-Aware:
      - Der Typ bleibt dabei an der Stadt „kleben“.
        Wir verschieben also Einträge vom Typ (Stadt, vehicle_type).
    """
    agent_genome = list(agent_genome)
    start_index = random.randint(1, len(agent_genome) - 1)
    length = random.randint(2, 20)

    genome_subset = agent_genome[start_index : start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length :]

    insert_index = random.randint(1, len(agent_genome))
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]

    return tuple(agent_genome)


def generate_random_population_type_aware(population_size, waypoints):
    """
    Erzeugt eine anfängliche Population aus type-aware Chromosomen.
    """
    random_population = []
    for _ in range(population_size):
        random_population.append(generate_random_agent_type_aware(waypoints))
    return random_population


def early_stopping(fitness_scores, threshold=0.01, patience=50):
    """
    überprüft, ob die Verbesserung zu gering ist.
    """
    if len(fitness_scores) < patience:
        return False
    recent_scores = fitness_scores[-patience:]
    return max(recent_scores) - min(recent_scores) < threshold


def run_genetic_algorithm_type_aware(places, distances_truck, distances_drone, generations,
                                     population_size, threshold=0.01, patience=50):
    """
    Hier werden zwei Distanzmatrizen genutzt: für Truck und Drone.
    """

    # Historie der Bestwerte, um early_stopping zu ermöglichen
    fitness_scores = []
    current_best_distance = float("inf")
    current_best_genome = None

    # Teilmengen-Parameter so wie vorher
    population_subset_size = population_size // 10
    generations_10pct = generations // 10

    # Initialpopulation
    population = generate_random_population_type_aware(population_size, places)

    for generation in range(generations):
        # Fitness aller Individuen berechnen
        population_fitness = {}
        for agent_genome in population:
            if agent_genome not in population_fitness:
                population_fitness[agent_genome] = compute_fitness_type_aware(
                    agent_genome, distances_truck, distances_drone
                )

        # Sortieren nach kleinstem Fitnesswert (Distanz)
        sorted_by_fitness = sorted(population_fitness, key=population_fitness.get)

        # Neue Population erzeugen
        new_population = []

        # Geh die Top 10% durch
        for rank, agent_genome in enumerate(sorted_by_fitness[:population_subset_size]):
            best_in_rank_distance = population_fitness[agent_genome]

            # Alle X% Generationen und am Ende: speichere aktuellen besten Genom
            if (generation % generations_10pct == 0 or generation == generations - 1) and rank == 0:
                current_best_genome = agent_genome

            # Falls wir einen „neuen“ Bestwert haben
            if best_in_rank_distance < current_best_distance:
                current_best_distance = best_in_rank_distance
                # Beispiel: Hier eventuell eine HTML-Visualisierung
                #create_optimal_route_html(
                #    agent_genome,
                #    current_best_distance,
                #    filename="best_route_type_aware.html"
                #)

            # 1. Unveränderte Kopie hinzufügen
            new_population.append(agent_genome)

            # 2. Zwei Offspring mit Punkt-Mutation
            for _ in range(2):
                mutated = mutate_agent_type_aware(agent_genome, max_mutations=3)
                new_population.append(mutated)

            # 3. Sieben Offspring mit Shuffle-Mutation
            for _ in range(7):
                shuffled = shuffle_mutation_type_aware(agent_genome)
                new_population.append(shuffled)

        # Population ersetzen
        population = new_population

        # Besten Fitness-Wert dieser Generation zur Historie hinzufügen
        best_fitness = min(population_fitness.values())
        fitness_scores.append(best_fitness)

        # Early Stopping
        if early_stopping(fitness_scores, threshold, patience):
            break

    return current_best_genome, current_best_distance



