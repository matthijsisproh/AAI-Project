from individual import Individual

class Population():
    def __init__(self, population_size, affinities):
        """
        Initializes a Population instance.

        Args:
            population_size (int): The size of the population.
            affinities (list): The affinities used for fitness evaluation.

        Attributes:
            population_size (int): The size of the population.
            affinities (list): The affinities used for fitness evaluation.
            population (list): The list of individuals in the population.
        """
        self.population_size = population_size
        self.affinities = affinities
        self.population = [Individual() for _ in range(population_size)]

    def fitness(self):
        """
        Evaluates the fitness of each individual in the population.

        Returns:
            list: The fitness values of the individuals in the population.
        """
        fitness_values = []
        for individual in self.population:
            individual.evaluate_fitness(self.affinities)
            fitness_values.append(individual.fitness)

        return fitness_values

    def get_population(self):
        """
        Returns the list of individuals in the population.

        Returns:
            list: The individuals in the population.
        """
        return self.population
