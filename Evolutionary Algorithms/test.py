# 1 random city
# loop through all cities
# en eindigen random city

# min mogelijk brandstof
# zo klein mogelijke afstand

import math

class City:
    def __init__(self, coords):
        self.coords = coords
        self.distance_from_start = None

    def set_distance(self, distance):
        self.distance_from_start = distance

cities = [ 
    [60,50],
    [100, 40],
    [90, 50],
    [20, 40],
    [60, 50],
    [70, 70],
    [20, 30],
    [90, 90],
    [30, 50],
    [10, 90]
]




start_city = cities[0] #Static

solution = cities


def calculate_total_distance(start_city, cities):
    current_city = start_city
    total_distance = 0

    for index in range(1, len(cities)):
        current_city = cities[index-1]
        next_city = cities[index]
        total_distance += math.dist(current_city, next_city)

    return total_distance*2



print(calculate_total_distance(start_city, cities))

def modify(solution):
    
    return solution

time = 0
while(True):
    new_solution = modify(solution)
    calculate_total_distance(new_solution)




        





