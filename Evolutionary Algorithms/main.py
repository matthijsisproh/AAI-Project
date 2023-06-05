import random
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import json

from operator import length_hint

import matplotlib.pyplot as plt

from population import Population

from evolutionaryalgorithm import evolutionary_algorithm

random.seed(0)

def main():
    """
    Perform the main execution of the evolutionary algorithm.

    This function initializes the necessary variables and objects, runs the evolutionary algorithm for a given number of generations,
    and prints and saves the results.

    Returns:
        None
    """
    best_fitness_results = []
    best_fitness_results_per_epoch = []
    best_individuals = []
    best_fitness_overall = float('-inf')
    
    population_size = 1000
    max_generations = 1000

    knight_names = np.genfromtxt(
            f'{os.getcwd()}/Evolutionary Algorithms/data/RondeTafel.csv', delimiter=';', usecols=[0], skip_header=2, dtype=str)
     
    affinities = np.genfromtxt(f'{os.getcwd()}/Evolutionary Algorithms/data/RondeTafel.csv',
                                         delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], skip_header=2)

    random_population = Population(population_size, affinities)

    start_time = time.time()
    for epoch in range(0, max_generations):
        population = evolutionary_algorithm(random_population)
        best_fitness_per_epoch = float('-inf')
        for individual in population:
            fitness = individual.evaluate_fitness(affinities)
            if fitness > best_fitness_overall:
                best_fitness_overall = fitness
                best_individuals.append(individual)
            if fitness > best_fitness_per_epoch:
                best_fitness_per_epoch = fitness

        best_fitness_results.append(best_fitness_overall)
        best_fitness_results_per_epoch.append(best_fitness_per_epoch)
        
        print("Epoch: {}, Best Fitness: {}, Maximum generations: {}".format(epoch, float(round(best_fitness_overall, 5)), max_generations), end='\r')
            
    print("\nElapsed time: ", round(time.time() - start_time, 3))

    plt.plot(range(len(best_fitness_results)), best_fitness_results,
        linewidth=1, color="blue", label="Max Fitness per Epoch")
    plt.legend()
    plt.show()

    plt.plot(range(len(best_fitness_results_per_epoch)), best_fitness_results_per_epoch,
        linewidth=1, color="blue", label="Max Fitness per Epoch")
    plt.legend()
    plt.show()

    for winning_individual in best_individuals:
        print("Individual genes: {}, \nFitness: {}".format(
            winning_individual._genes, 
            winning_individual.fitness), 
            end='\n')
        
        for weight_products in winning_individual.weight_products:
            print("Knight: {}, Knight 2: {}, weight_product {}".format(
                knight_names[weight_products[0]],
                knight_names[weight_products[1]],
                weight_products[2]
                ))

    output_data = []

    for winning_individual in best_individuals:
        individual_data = {
            "genes": winning_individual._genes,
            "fitness": winning_individual.fitness,
            "weight_products": []
        }
        for weight_products in winning_individual.weight_products:
            weight_product_data = {
                "knight1": knight_names[weight_products[0]],
                "knight2": knight_names[weight_products[1]],
                "weight_product": weight_products[2]
            }
            individual_data["weight_products"].append(weight_product_data)
        output_data.append(individual_data)

    with open("output.json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)

if __name__ == "__main__":
    main()
