from createHTML import create_optimal_route_html, get_route_from_ranking
import random
from concurrent.futures import ThreadPoolExecutor
import os

num_threads = os.cpu_count()


def compute_fitness_parallel(agent_genome, waypoints_distances):
    """ Parallel fitness calculation for one agent """
    solution_fitness = 0.0
    for index in range(len(agent_genome)):
        waypoint1 = agent_genome[index - 1]
        waypoint2 = agent_genome[index]
        solution_fitness += waypoints_distances[frozenset([waypoint1, waypoint2])]
    return agent_genome, solution_fitness


def compute_fitness(solution, waypoints_distances):
    """ Calculates the total distance traveled on this current tour. GA will favor road trips with shorter total distances traveled
    """
    solution_fitness = 0.0

    for index in range(len(solution)):
        waypoint1 = solution[index - 1]
        waypoint2 = solution[index]
        solution_fitness += waypoints_distances[frozenset([waypoint1, waypoint2])]

    return solution_fitness


def generate_random_agent(waypoints):
    """
        Creates a random road trip from all waypoints
    """
    new_random_agent = list(waypoints)
    random.shuffle(new_random_agent)
    return tuple(new_random_agent)


def mutate_agent(agent_genome, max_mutations=3):
    """
        A point mutation swaps the order of 2 waypoints in the tour.
        We apply a number of point mutations in range of 1 - max.
    """
    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)
    # max_mutations - max number of mutations for each agent

    for mutation in range(num_mutations):
        swap_index1 = random.randint(0, len(agent_genome) - 1)
        swap_index2 = swap_index1

        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(agent_genome) - 1)

        agent_genome[swap_index1], agent_genome[swap_index2] = agent_genome[swap_index2], agent_genome[swap_index1]

    return tuple(agent_genome)


def shuffle_mutation(agent_genome):
    """
        Applies a single shuffle mutation to the given tour.

        A shuffle mutation takes a random sub-section of the tour
        and moves it to another location in the tour.
    """
    agent_genome = list(agent_genome)

    start_index = random.randint(0, len(agent_genome) - 1)
    length = random.randint(2, 20)

    genome_subset = agent_genome[start_index: start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]

    insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]

    return tuple(agent_genome)


def generate_random_population(population_size, waypoints):
    """
        Generates a list with 'pop_size' number of tours.
    """
    random_population = []
    for agent in range(population_size):
        random_population.append(generate_random_agent(waypoints))
    return random_population


def early_stopping(fitness_scores, threshold=0.01, patience=50):
    """ Checks whether the improvement in the fitness function is insignificant over several generations.
    """
    if len(fitness_scores) < patience:
        return False
    recent_scores = fitness_scores[-patience:]
    return max(recent_scores) - min(recent_scores) < threshold


def run_genetic_algorithm(places, waypoints_distances, generations, population_size, threshold=0.01, patience=50):
    """
        Core of the GA -- 'generations' and 'population_size' must be a multiple of 10.
    """
    fitness_scores = []
    current_best_distance = 1
    population_subset_size = int(population_size // 10)
    generations_10pct = int(generations // 10)
    current_best_genome = []  # init
    # Create a random population of 'population_size' number of solutions
    population = generate_random_population(population_size, places)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # For 'generations' number of repetitions...
        for generation in range(generations):
            futures = {executor.submit(compute_fitness_parallel, agent, waypoints_distances): agent for agent in population}
            population_fitness = {future.result()[0]: future.result()[1] for future in futures}

            # Take top 10% shortest tours and produce offspring from each of them
            new_population = []
            for rank, agent_genome in enumerate(
                    sorted(population_fitness, key=population_fitness.get)[:population_subset_size]):
                if (generation % generations_10pct == 0 or generation == generations - 1) and rank == 0:
                    current_best_genome = agent_genome
                    # print("Generation %d | best: %d | Unique genomes: %d" % (generation, population_fitness[agent_genome], len(population_fitness)))
                    # print(agent_genome)

                # If this is the first route found, or it is shorter than the best route we know,
                # create a html output and display it
                if population_fitness[agent_genome] < current_best_distance or current_best_distance < 0:
                    current_best_distance = population_fitness[agent_genome]
                    create_optimal_route_html(agent_genome, current_best_distance)

                # Create 1 exact copy of each of the top tours
                new_population.append(agent_genome)

                # Create 2 offspring with 1-3 point mutations
                for offspring in range(2):
                    new_population.append(mutate_agent(agent_genome, 3))

                # Create 7 offspring with a single shuffle mutation
                for offspring in range(7):
                    new_population.append(shuffle_mutation(agent_genome))

                # Replace the old population with new population of offspring
                population = new_population

                # Add the best fitness score of this generation to the list
                best_fitness = min(population_fitness.values())
                fitness_scores.append(best_fitness)

                # Early stopping check
                if early_stopping(fitness_scores, threshold, patience):
                    # print(f"Early stopping triggered at generation {generation}.")
                    break

    return current_best_genome
