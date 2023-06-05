import random
from operator import length_hint

from individual import Individual


def truncation_selection(population, retain_percentage):
    """
    Perform truncation selection on a population.

    Sorts the population based on fitness in descending order and retains a certain percentage of the top individuals.

    Args:
        population (list): The population to perform selection on.
        retain_percentage (float): The percentage of individuals to retain.

    Returns:
        tuple: A tuple containing two lists: the non-selected individuals and the selected individuals.
    """
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    length_population = len(sorted_population)
    retain_count = int(length_population * retain_percentage)
    notselected_individuals = sorted_population[retain_count:]
    selected_individuals = sorted_population[:retain_count]
    return notselected_individuals, selected_individuals


def generate_child(parent_one, parent_two):
    """
    Generate a child individual using partially mapped crossover.

    Args:
        parent_one (Individual): The first parent individual.
        parent_two (Individual): The second parent individual.

    Returns:
        Individual: The generated child individual.
    """
    child = Individual([None] * 12)
    index_list = sorted(random.sample(range(12), 4))
    numbers_to_take = sorted(random.sample(range(12), 4))

    for iterator in range(len(index_list)):
        child._genes[index_list[iterator]] = parent_one._genes[numbers_to_take[iterator]]

    for child_iterator in range(len(child._genes)):
        if child._genes[child_iterator] is None:
            for number in parent_two._genes:
                if number not in child._genes:
                    child._genes[child_iterator] = number

    return child


def partially_mapped_crossover(parent_one, parent_two):
    """
    Perform partially mapped crossover between two parent individuals.

    Args:
        parent_one (Individual): The first parent individual.
        parent_two (Individual): The second parent individual.

    Returns:
        tuple: A tuple containing two child individuals.
    """
    child_one = generate_child(parent_one, parent_two)
    child_two = generate_child(parent_two, parent_one)
    return child_one, child_two


def reproduce(selected_individuals):
    """
    Reproduce selected individuals to create offspring.

    Args:
        selected_individuals (list): The selected individuals for reproduction.

    Returns:
        list: The offspring individuals.
    """
    offspring = []

    for index in range(0, length_hint(selected_individuals) - 1, 2):
        parent_one = selected_individuals[index]
        parent_two = selected_individuals[index + 1]

        parent_one.swap_mutate(mutation_rate=100)
        parent_two.swap_mutate(mutation_rate=100)

        child_one, child_two = partially_mapped_crossover(parent_one, parent_two)

        offspring.extend([child_one, child_two])

    return offspring


def evolutionary_algorithm(population_obj):
    """
    Perform an evolutionary algorithm on a population.

    Evaluates the fitness of the population, performs truncation selection, reproduction, and generates a new population.

    Args:
        population_obj (Population): The population object containing the individuals.

    Returns:
        list: The new population after the evolutionary algorithm.
    """
    population_obj.fitness()
    population = population_obj.get_population()
    winners, ready_for_mutation_individuals = truncation_selection(population, 0.9)
    mutated_individuals = reproduce(ready_for_mutation_individuals)
    new_population = mutated_individuals + winners
    return new_population
