import random
import itertools
import time
import numpy as np
import heapq

class City:
    def __init__(self, letter, x, y, z):
        self.letter = letter
        self.x = x
        self.y = y
        self.z = z
        self.visited = False

    def distance_to(self, other_city):
        return np.sqrt((self.x - other_city.x) ** 2 + (self.y - other_city.y) ** 2 + (self.z - other_city.z) ** 2)

# Full Search (Exhaustive Search)
def full_search_tsp(cities):
    best_tour = None
    best_tour_length = float('inf')
    num_cities = len(cities)

    for perm in itertools.permutations(cities[1:]):
        tour = [cities[0]] + list(perm) + [cities[0]]
        tour_length = sum(tour[i].distance_to(tour[i + 1]) for i in range(num_cities))        
        if tour_length < best_tour_length:
            best_tour_length = tour_length
            best_tour = tour

    return best_tour, best_tour_length

# Greedy Algorithm
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

# A* Algorithm
def a_star_tsp(cities):
    num_cities = len(cities)
    start_city = cities[0]
    unvisited_cities = set(cities[1:])
    priority_queue = [(0, start_city, [start_city])]
    visited_cities = []

    while priority_queue:
        cost, current_city, path = heapq.heappop(priority_queue)

        if len(path) == num_cities:
            return path + [start_city], cost + current_city.distance_to(start_city)

        if current_city not in visited_cities:
            visited_cities.append(current_city)

            for next_city in unvisited_cities:
                new_cost = cost + current_city.distance_to(next_city)
                heuristic = next_city.distance_to(start_city)  # Using the distance to the start city as the heuristic
                total_cost = new_cost + heuristic
                heapq.heappush(priority_queue, (total_cost, next_city, path + [current_city]))

    return None, None

# ACO Algorithm
class Ant:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.tour = None
        self.tour_length = None

    def _select_next_city(self, pheromone_matrix, alpha, beta):
        current_city = self.tour[-1]
        unvisited_cities = [city for city in self.cities if city not in self.tour]
        
        probabilities = []
        total_probability = 0.0

        for city in unvisited_cities:
            current_index = self.cities.index(current_city)
            city_index = self.cities.index(city)
            pheromone = pheromone_matrix[current_index][city_index]
            distance = current_city.distance_to(city)
            attractiveness = (pheromone ** alpha) * ((1.0 / distance) ** beta)
            probabilities.append(attractiveness)
            total_probability += attractiveness

        probabilities = [p / total_probability for p in probabilities]

        selected_index = np.random.choice(len(unvisited_cities), p=probabilities)
        return unvisited_cities[selected_index]

    def find_tour(self, pheromone_matrix, alpha, beta):
        self.tour = [self.cities[0]]
        while len(self.tour) < self.num_cities:
            next_city = self._select_next_city(pheromone_matrix, alpha, beta)
            self.tour.append(next_city)

        self.tour_length = sum(self.tour[i].distance_to(self.tour[i + 1]) for i in range(self.num_cities - 1)) + self.tour[-1].distance_to(self.tour[0])

class ACO:
    def __init__(self, cities, num_ants, num_iterations, alpha, beta, evaporation_rate, initial_pheromone):
        self.cities = cities
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = initial_pheromone

        self.num_cities = len(cities)
        self.pheromone_matrix = np.full((self.num_cities, self.num_cities), initial_pheromone)
        self.city_indices = {city: index for index, city in enumerate(cities)}

    def run(self):
        best_tour = None
        best_tour_length = float('inf')

        for iteration in range(self.num_iterations):
            ants = [Ant(self.cities) for _ in range(self.num_ants)]
            for ant in ants:
                ant.find_tour(self.pheromone_matrix, self.alpha, self.beta)
                if ant.tour_length < best_tour_length:
                    best_tour_length = ant.tour_length
                    best_tour = ant.tour

            self.update_pheromone(ants)

        # Evaporate pheromone
        self.pheromone_matrix *= (1.0 - self.evaporation_rate)

        # Apply pheromone for the best tour found
        if best_tour:
            for i in range(len(best_tour) - 1):
                city1 = best_tour[i]
                city2 = best_tour[i + 1]
                index1 = self.city_indices[city1]
                index2 = self.city_indices[city2]
                self.pheromone_matrix[index1][index2] += 1.0 / best_tour_length
                self.pheromone_matrix[index2][index1] += 1.0 / best_tour_length

        # Ensure the best tour is complete
        if best_tour and best_tour[0] != best_tour[-1]:
            best_tour.append(best_tour[0])
            best_tour_length += best_tour[-2].distance_to(best_tour[0])

        # Ensure all cities are visited in the ACO result
        visited_cities = [city.letter for city in best_tour]
        for city in self.cities:
            if city.letter not in visited_cities:
                return None, None

        return best_tour, best_tour_length

    def update_pheromone(self, ants):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    pheromone_change = 0.0
                    for ant in ants:
                        current_index = ant.cities.index(ant.tour[i])
                        next_index = ant.cities.index(ant.tour[j])
                        if (current_index, next_index) in ant.tour or (next_index, current_index) in ant.tour:
                            pheromone_change += 1 / ant.tour_length
                    self.pheromone_matrix[i][j] = (1 - self.evaporation_rate) * self.pheromone_matrix[i][j] + pheromone_change


if __name__ == "__main__":
    cityNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    cities = []
    for letter in cityNames:
        cities.append(City(letter, random.randint(-100, 100), random.randint(-100, 100), random.randint(0, 50)))

    # Full Search
    full_search_start_time = time.time()
    full_search_best_tour, full_search_best_length = full_search_tsp(cities)
    full_search_time = time.time() - full_search_start_time

    # Greedy
    greedy_start_time = time.time()
    greedy_best_tour, greedy_best_length = greedy_tsp(cities)
    greedy_time = time.time() - greedy_start_time

    # A*
    # a_star_start_time = time.time()
    # a_star_best_tour, a_star_best_length = a_star_tsp(cities)
    # a_star_time = time.time() - a_star_start_time

    # ACO
    aco_start_time = time.time()
    aco = ACO(cities, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.1, initial_pheromone=1.0)
    aco_best_tour, aco_best_length = aco.run()
    aco_time = time.time() - aco_start_time

    print("Full Search Best Tour Order:", [city.letter for city in full_search_best_tour])
    print("Full Search Best Tour Length:", full_search_best_length)
    print("Full Search Time:", full_search_time)

    print("Greedy Best Tour Order:", [city.letter for city in greedy_best_tour])
    print("Greedy Best Tour Length:", greedy_best_length)
    print("Greedy Time:", greedy_time)

    # print("A* Best Tour Order:", [city.letter for city in a_star_best_tour])
    # print("A* Best Tour Length:", a_star_best_length)
    # print("A* Time:", a_star_time)

    print("ACO Best Tour Order:", [city.letter for city in aco_best_tour])
    print("ACO Best Tour Length:", aco_best_length)
    print("ACO Time:", aco_time)
