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
        

def greedy_tsp(cities):
    start_city = cities[0]
    unvisited_cities = set(cities[1:])
    tour = [start_city]
    tour_length = 0

    while unvisited_cities:
        min_distance = float('inf')
        nearest_city = None

        for next_city in unvisited_cities:
            distance = start_city.distance_to(next_city)
            if distance < min_distance:
                min_distance = distance
                nearest_city = next_city

        tour.append(nearest_city)
        tour_length += min_distance
        start_city = nearest_city
        unvisited_cities.remove(nearest_city)

    # Return to the start city to complete the tour
    tour.append(cities[0])
    tour_length += start_city.distance_to(cities[0])

    return tour, tour_length

if __name__ == "__main__":
    cityNames = ['A', 'B', 'C', 'D', 'E']
    cities = []
    for letter in cityNames:
        cities.append(City(letter, random.randint(-100, 100), random.randint(-100, 100), random.randint(0, 50)))

    full_search_start_time = time.time()
    tour, total_distance = greedy_tsp(cities)
    full_search_time = time.time() - full_search_start_time
    
    print("Tour order:", [l.letter for l in tour])
    print("Total distance:", total_distance)
    print("Time: ", full_search_time)