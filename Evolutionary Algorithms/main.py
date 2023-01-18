# Local hill climbing
from random import randint, random
from functools import reduce
from operator import add


class Solution:
    def __init__(self, value):
        self.value = value
        self.fitness = None

    def set_fitness(self, value):
        self.fitness = value

    def modify(self):
        return self
    
    def calculate_fitness(self):
        calculated_value = 1
        self.fitness = calculated_value

def individual(length, min, max):
    return[randint(min, max) for x in range(length)]

def population(count, length, min, max):
    return [individual(length, min, max) for x in range(count)]

def fitness(individual, target):
    sum = reduce(add, individual, 0)
    return abs(target-sum)

def grade(population, target):
    summed = reduce(add, (fitness(x, target) for x  in population), 0)
    return summed / len(population)

def tournament_selection(size, pool):
    probability = max(pool.propability)
    best_individual = pool # best propability
    second_individual = propability * (1 - propability)
    # and so on

def evolve(population, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [fitness(x,target), x) for x in population]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]

    #Randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # Crossover parents to create offspring

    # Mutate some individuals



    





random_solution = Solution([10,10])
list_of_solutions = [random_solution]
time = 0

while(True):
    new_solution = Solution(list_of_solutions[time].modify())
    new_solution.calculate_fitness()

    if new_solution.fitness > list_of_solutions[time].fitness:
        new_solution = list_of_solutions[time+1]

    else:
        list_of_solutions[time+1] = list_of_solutions[time]
    
    time = time + 1

