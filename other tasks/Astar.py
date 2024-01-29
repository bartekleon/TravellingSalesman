import random
import numpy as np
import time

class City:
    def __init__(self, letter, x, y, z):
        self.letter = letter
        self.x = x
        self.y = y
        self.z = z
        self.visited = False

    def distance_to(self, other_city):
        return np.sqrt((self.x - other_city.x) ** 2 + (self.y - other_city.y) ** 2 + (self.z - other_city.z) ** 2)

def inadmissible_greedy_pairwise_heuristic(city, unvisited_cities):
    total_distance = 0
    for u_city in unvisited_cities:
        total_distance += city.distance_to(u_city)
    return total_distance

def minimum_spanning_tree_heuristic(city, unvisited_cities):
    # Prim's algorithm for MST
    visited = {city}
    unvisited = unvisited_cities.copy()
    total_weight = 0

    while unvisited:
        min_edge = None
        min_distance = float('inf')

        for v in visited:
            for u in unvisited:
                distance = v.distance_to(u)
                if distance < min_distance:
                    min_distance = distance
                    min_edge = (v, u)

        if min_edge:
            u, v = min_edge
            visited.add(u)
            unvisited.remove(v)
            total_weight += min_distance

    return total_weight

def tsp_a_star(start_city, cities, heuristic_func):
    unvisited_cities = cities.copy()
    unvisited_cities.remove(start_city)
    current_city = start_city
    route = [start_city]
    total_distance = 0

    start_time = time.time()

    while unvisited_cities:
        # Sort unvisited cities based on heuristic values
        unvisited_cities.sort(key=lambda x: heuristic_func(current_city, [x]))
        
        next_city = unvisited_cities[0]
        route.append(next_city)
        total_distance += current_city.distance_to(next_city)
        current_city = next_city
        unvisited_cities.remove(current_city)

    total_distance += current_city.distance_to(start_city)
    route.append(start_city)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return [city.letter for city in route], total_distance, elapsed_time

cityNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
cities = []
for letter in cityNames:
    cities.append(City(letter, random.randint(-100, 100), random.randint(-100, 100), random.randint(0, 50)))

start_city = cities[0]

# Solve using the inadmissible heuristic
route_inadmissible, total_distance_inadmissible, time_inadmissible = tsp_a_star(start_city, cities, inadmissible_greedy_pairwise_heuristic)
print("Inadmissible Heuristic:")
print("Route:", route_inadmissible)
print("Total Distance:", total_distance_inadmissible)
print("Time:", time_inadmissible, "seconds")

# Solve using the admissible heuristic
route_admissible, total_distance_admissible, time_admissible = tsp_a_star(start_city, cities, minimum_spanning_tree_heuristic)
print("\nAdmissible Heuristic:")
print("Route:", route_admissible)
print("Total Distance:", total_distance_admissible)
print("Time:", time_admissible, "seconds")
