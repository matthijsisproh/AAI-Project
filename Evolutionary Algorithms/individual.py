import random

class Individual():
    def __init__(self, genes=None):
        """
        Initialize a new individual.

        Args:
            genes (list, optional): The genes of the individual. Defaults to None.
                If not provided, the genes will be randomly sampled from the range [0, 11].

        Attributes:
            fitness (float or None): The fitness value of the individual.
            weight_products (list): List to store weight products of genes.

        """
        if genes is None:
            self._genes = random.sample(range(12), 12)  # Assuming we have 12 knights
        else:
            self._genes = genes

        self.fitness = None
        self.weight_products = []

    def copy(self):
        """
        Create a copy of the individual.

        Returns:
            Individual: A new individual with the same genes and fitness value.

        """
        copied_individual = Individual(self._genes.copy())
        copied_individual.fitness = self.fitness
        copied_individual.weight_products = self.weight_products.copy()
        return copied_individual

    def evaluate_fitness(self, affinities):
        """
        Evaluate the fitness of the individual based on affinities.

        Args:
            affinities (list of lists): A 2D list representing the affinities between genes.

        Returns:
            float: The fitness value of the individual.

        """
        total_affinity = 0
        self.weight_products = []

        for individual_index in range(len(self._genes)):
            if individual_index == len(self._genes) - 1:
                affinity = (
                    affinities[self._genes[individual_index]][self._genes[0]]
                    * affinities[self._genes[0]][self._genes[individual_index]]
                )
                self.weight_products.append(
                    (self._genes[individual_index], self._genes[0], affinity)
                )
                total_affinity += affinity
            else:
                affinity = (
                    affinities[self._genes[individual_index]][
                        self._genes[individual_index + 1]
                    ]
                    * affinities[self._genes[individual_index + 1]][
                        self._genes[individual_index]
                    ]
                )
                self.weight_products.append(
                    (
                        self._genes[individual_index],
                        self._genes[individual_index + 1],
                        affinity,
                    )
                )
                total_affinity += affinity

        self.fitness = 1 - (total_affinity / len(self._genes))  # Average affinity

        return self.fitness

    def swap_mutate(self, mutation_rate):
        """
        Perform swap mutation on the individual.

        Args:
            mutation_rate (float): The probability of mutation for each gene.

        Returns:
            Individual: A new individual with the mutated genes.

        """
        mutated_individual = self.copy()

        for i in range(len(mutated_individual._genes)):
            if random.random() < mutation_rate:
                swap_index = random.randint(0, len(mutated_individual._genes) - 1)
                mutated_individual._genes[i], mutated_individual._genes[swap_index] = (
                    mutated_individual._genes[swap_index],
                    mutated_individual._genes[i],
                )

        return mutated_individual
