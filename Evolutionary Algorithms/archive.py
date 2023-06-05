import random
import time
from typing import Any
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import json

from operator import length_hint

from individual import Individual





class GeneticAlgorithm():
    def __init__(
            self,
            population_size,
            target_fitness,
            max_runtime,
            mutation_chance
    ):
        self._population_size = population_size
        self._target_fitness = target_fitness
        self._max_runtime = max_runtime
        self._mutation_chance = mutation_chance

        self._start_time = time.time()

        self._knight_names = np.genfromtxt(
            f'{os.getcwd()}/Evolutionary Algorithms/data/RondeTafel.csv', delimiter=';', usecols=[0], skip_header=2, dtype=str)
        self._affinities = np.genfromtxt(f'{os.getcwd()}/Evolutionary Algorithms/data/RondeTafel.csv',
                                         delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], skip_header=2)
        
        self._random_population = self.generate_population()


    def generate_individual(self):
        """
        Generate a random individual.

        :return: A randomly generated individual representing the order of knights around the table.
        :rtype: list
        """
        individual = random.sample(
            range(12), 12)  # Assuming we have 12 knights

        return individual

    def generate_population(self):
        population = []

        for _ in range(self._population_size):         
            individual = Individual()
            population.append(individual)

        return population


    def calculate_fitness(self, individual):
        total_affinity = 0
        for individual_index in range(length_hint(individual)):
            if individual_index == length_hint(individual) - 1:
                total_affinity += self._affinities[individual[individual_index]][individual[0]
                                                                                 ] * self._affinities[individual[0]][individual[individual_index]]
            else:
                total_affinity += self._affinities[individual[individual_index]][individual[individual_index+1]
                                                                                 ] * self._affinities[individual[individual_index+1]][individual[individual_index]]

        fitness = 1 - (total_affinity / (length_hint(individual)))  # Average affinity

        return fitness

    def mutate(self, individual, mutation_chance):
        if random.randint(1, 1000) < mutation_chance*10:
            index_one = random.randint(0, 11)
            index_two = random.randint(0, 11)
            while index_two == index_one:
                index_two = random.randint(0, 11)
            individual[index_one], individual[index_two] = individual[index_two], individual[index_one]
        return individual

    def tournament_selection(self, population, tournament_size, num_winners):
        winners = []
        non_winners = population.copy()

        for _ in range(num_winners):
            # Selecteer willekeurig individuen voor het toernooi
            tournament_candidates = random.sample(non_winners, tournament_size)

            # Sorteer de geselecteerde individuen op basis van hun fitnesswaarden
            tournament_candidates.sort(key=lambda x: x.fitness, reverse=True)

            # Selecteer het individu met de hoogste fitnesswaarde als winnaar
            winner = tournament_candidates[0]
            
            winners.append(winner)
            non_winners.remove(winner)

        return winners, non_winners



    def truncation_selection(self, population, retain_percentage):
        # Sorteer de populatie op basis van fitness (van hoog naar laag)
        sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        
        length_population = len(sorted_population)

        # Bepaal het aantal individuen dat behouden moet worden
        retain_count = int(length_population * retain_percentage)
        
        # Selecteer de niet geselecteerde individuen
        notselected_individuals = sorted_population[retain_count:]

        # Selecteer de topindividuen voor reproductie
        selected_individuals = sorted_population[:retain_count]
        
        return notselected_individuals, selected_individuals

    def roulette_wheel_selection(self, population, num_winners):
        total_fitness = sum(individual.fitness for individual in population)

        winners = []
        non_winners = population.copy()

        for _ in range(num_winners):
            # Genereer een willekeurig getal tussen 0 en de totale fitness
            spin = random.uniform(0, total_fitness)

            accumulated_fitness = 0
            for individual in non_winners:
                accumulated_fitness += individual.fitness
                if accumulated_fitness >= spin:
                    winners.append(individual)
                    non_winners.remove(individual)
                    break

        return winners, non_winners



    def elitist_selection(self, population, num_winners):
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        winners = sorted_population[:num_winners]
        non_winners = sorted_population[num_winners:]
        return winners, non_winners


    def gen_child(self, parent_one, parent_two):
        child = Individual([None]*12)  # Assuming Individual class has a _genes attribute
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

    def partially_mapped_crossover(self, parent_one, parent_two):
        child_one = self.gen_child(parent_one, parent_two)
        child_two = self.gen_child(parent_two, parent_one)

        return child_one, child_two


   # Function for the single point crossover operator

    def single_point_crossover(self, parent_one, parent_two):
        # Select a random crossover point
        parent_one_genes = parent_one._genes
        parent_two_genes = parent_two._genes
        
        crossover_point = random.randint(1, len(parent_one_genes) - 1)

        child_one_genes = parent_one_genes[:crossover_point]
        child_two_genes = parent_two_genes[:crossover_point]

        for gene in parent_two_genes[crossover_point:]:
            if gene not in child_one_genes:
                child_one_genes.append(gene)

        if len(child_one_genes) < len(parent_one_genes):
            remaining_genes = [gene for gene in parent_one_genes if gene not in child_one_genes]
            random.shuffle(remaining_genes)
            child_one_genes.extend(remaining_genes[:len(parent_one_genes) - len(child_one_genes)])
        
        for gene in parent_one_genes[crossover_point:]:
            if gene not in child_two_genes:
                child_two_genes.append(gene)

        if len(child_two_genes) < len(parent_two_genes):
            remaining_genes = [gene for gene in parent_two_genes if gene not in child_two_genes]
            random.shuffle(remaining_genes)
            child_two_genes.extend(remaining_genes[:len(parent_two_genes) - len(child_two_genes)])

        child_one = Individual(child_one_genes)
        child_two = Individual(child_two_genes)

        return child_one, child_two


    def print_progressbar(self, start_time, end_time, max_runtime):
        # Progress bar
        percentage = round(
            (max_runtime-(end_time-start_time))/(max_runtime)*100, 2)
        print('Progress: [{}{}{}]  {}%'.format(('=' * int(percentage//10)),
                                               ('>' if percentage < 100 else ''),
                                               ('.' * int(10-(((percentage)//10))-1)),
                                               percentage),
              end='\r'
              )

    def print_progressbar_epoch(self, start_time, end_time, max_runtime):
            # Progress bar
            percentage = round(
                (max_runtime-(end_time-start_time))/(max_runtime)*100, 2)
            print('Progress: [{}{}{}]  {}%'.format(('=' * int(percentage//10)),
                                                ('>' if percentage < 100 else ''),
                                                ('.' * int(10-(((percentage)//10))-1)),
                                                percentage),
                end='\r'
                )

    def evolutionary_loop(self, population, max_generations):
        result_population = []
        for generation in range(max_generations):
            try:
                population = self.get_population()
            except:
                population = population

            fitness_values, population = self.populational_fitness(population)

            # # Select parents for reproduction
            winners, ready_for_mutation_individuals = self.truncation_selection(population, 0.9)
            
            # winners, selected_individuals = self.tournament_selection(population, 10, 5)
            
            # notselected_individuals, selected_individuals = self.roulette_wheel_selection(population, 10)
            
            # winners, selected_individuals = self.elitist_selection(population, 10)
            
            # offspring = self.reproduce(winners)
            offspring = self.reproduce(ready_for_mutation_individuals)

            
            new_population = offspring + winners

            fitness_values, new_population = self.populational_fitness(new_population)
            
            best_individual = max(new_population, key=lambda individual: individual.fitness)
            # print("Generation:", generation+1, "Best Individual:", best_individual._genes, "Fitness:", best_individual.fitness)

            # self.set_population(new_population)

            result_population.append(best_individual)            



        return result_population

    def execute(self, max_generations):
        # Evolutionary Loop
        # population = self.generate_population()
        new_population = self.evolutionary_loop(self._random_population, max_generations)
        best_individual = max(new_population, key=lambda individual: individual.fitness)
        # print("Result run: Best Individual:", best_individual._genes, "Fitness:", best_individual.fitness)

        return new_population, best_individual


    def evolutionary_algorithm(self, population):
        fitness_values, population = self.populational_fitness(population)

        winners, ready_for_mutation_individuals = self.truncation_selection(population, 0.9)
            
        offspring = self.reproduce(ready_for_mutation_individuals)
          
        new_population = offspring + winners

        return new_population
    
    def reproduce(self, selected_individuals):
        offspring = []

        for index in range(0, length_hint(selected_individuals) -1, 2):
            parent_one = selected_individuals[index]
            parent_two = selected_individuals[index + 1]

            parent_one.swap_mutate(mutation_rate=100)
            parent_two.swap_mutate(mutation_rate=100)

            child_one, child_two = self.partially_mapped_crossover(parent_one, parent_two)
            # child_one, child_two = self.single_point_crossover(parent_one, parent_two)

            offspring.extend([child_one, child_two])

        return offspring

    # Opdracht 2B C
    def analyse_per_epoch(self, max_generations):
        best_fitness_results = []
        best_fitness_results_per_epoch = []
        best_individuals = []
        print(self._knight_names)
        # population = self._random_population
        best_fitness_overall = float('-inf')
        for epoch in range(0, max_generations):
            population = self.evolutionary_algorithm(self._random_population)
            best_fitness_per_epoch = float('-inf')
            for individual in population:
                fitness = individual.evaluate_fitness(self._affinities)
                if fitness > best_fitness_overall:
                    best_fitness_overall = fitness
                    best_individuals.append(individual)
                if fitness > best_fitness_per_epoch:
                    best_fitness_per_epoch = fitness

            best_fitness_results.append(best_fitness_overall)
            best_fitness_results_per_epoch.append(best_fitness_per_epoch)
            
            # Exercise 2B
            # print("Epoch: {}, Best Fitness: {}, Maximum generations: {}".format(epoch, float(round(best_fitness_per_epoch, 5)), max_generations), end='\r')
            print("Epoch: {}, Best Fitness: {}, Maximum generations: {}".format(epoch, float(round(best_fitness_overall, 5)), max_generations), end='\r')
                  

        # Exercise 2B
        plt.plot(range(len(best_fitness_results)), best_fitness_results,
             linewidth=1, color="blue", label="Max Fitness per Epoch")
        plt.legend()
        plt.show()

        # Exercise 2C
        plt.plot(range(len(best_fitness_results_per_epoch)), best_fitness_results_per_epoch,
             linewidth=1, color="blue", label="Max Fitness per Epoch")
        plt.legend()
        plt.show()

        # Exercise 2D
        for winning_individual in best_individuals:
            print("Individual genes: {}, \nFitness: {}".format(
                winning_individual._genes, 
                winning_individual.fitness), 
                end='\n')
            
            for weight_products in winning_individual.weight_products:
                # print(weight_products)
                print("Knight: {}, Knight 2: {}, weight_product {}".format(
                    self._knight_names[weight_products[0]],
                    self._knight_names[weight_products[1]],
                    weight_products[2]
                    ))
        # Output to json
        self.output_winners_tojson(best_individuals)

    def output_winners_tojson(self, best_individuals):
        output_data = []

        for winning_individual in best_individuals:
            individual_data = {
                "genes": winning_individual._genes,
                "fitness": winning_individual.fitness,
                "weight_products": []
            }

            for weight_products in winning_individual.weight_products:
                weight_product_data = {
                    "knight1": self._knight_names[weight_products[0]],
                    "knight2": self._knight_names[weight_products[1]],
                    "weight_product": weight_products[2]
                }

                individual_data["weight_products"].append(weight_product_data)

            output_data.append(individual_data)

        # Write the output data to a JSON file
        with open("output.json", "w") as json_file:
            json.dump(output_data, json_file, indent=4)



    def populational_fitness(self, population) -> list:
        fitness_list = []
        for individual in population:
            individual.evaluate_fitness(self._affinities)
            fitness_list.append(individual.fitness)

        population = sorted(population, key=lambda individual: individual.fitness, reverse=True)

        return fitness_list, population

    def set_population(self, population):
        self._population = population

    def get_population(self):
        return self._population