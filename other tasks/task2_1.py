import numpy as np
import pandas as pd

data = pd.read_excel("GA_task.xlsx", header=None)

def data_to_dict(data):
  order_data = {}
  for order in range(1, 51):
    start_row = data[data.iloc[:, 2*(order-1)] == "R"].index[0] + 1
    order_info = data.iloc[start_row:start_row+11, 2*(order-1):2*(order-1)+2].values
    resources = order_info[:, 0].astype(int)
    times = order_info[:, 1].astype(int)
    order_data[order] = list(zip(resources, times))
  return order_data

order_data = data_to_dict(data)

def create_gantt_chart(genotype):
  gantt = {resource: [] for resource in range(1, 11)}
  resource_time = {resource: 0 for resource in range(1, 11)}
  for order in genotype:
    steps = order_data[order]
    order_start_time = 0
    for resource, time in steps: 
      if resource_time[resource] <= order_start_time:
        start_time = order_start_time
      else:
        start_time = resource_time[resource]
      end_time = start_time + time
      gantt[resource].append((start_time, end_time, order))
      resource_time[resource] = end_time
      order_start_time = end_time
  return gantt

def evaluate(genotype):
  gantt = create_gantt_chart(genotype)
  total_time = max(gantt[resource][-1][1] if gantt[resource] else 0 for resource in gantt)
  return total_time

def initialize_population(population_size):
  return [list(np.random.permutation(range(1, 51))) for _ in range(population_size)]

def mutate(solution, mutation_rate):
  if np.random.rand() < mutation_rate:
    i, j = np.random.choice(range(len(solution)), 2, replace=False)
    solution[i], solution[j] = solution[i], solution[j]
  return solution

def crossover(parent1, parent2):
  crossover_point = np.random.randint(1, len(parent1))
  
  child1_prefix = parent1[:crossover_point]
  child2_prefix = parent2[:crossover_point]
  
  child1_suffix = [gene for gene in parent2 if gene not in child1_prefix]
  child2_suffix = [gene for gene in parent1 if gene not in child2_prefix]
  
  child1 = child1_prefix + child1_suffix
  child2 = child2_prefix + child2_suffix
  
  return child1, child2

def selection(population, selection_rate):
  sorted_population = sorted(population, key=evaluate)
  selection_size = int(selection_rate * len(sorted_population))
  sorted_population[-selection_size:] = sorted_population[:selection_size].copy()
  return sorted_population

def genetic_algorithm(population_size=100, mutation_rate=0.1, crossover_rate=0.8, max_generations=10000, selection_rate=0.25, max_stagnant_gen=500):
  population = initialize_population(population_size)
  generations = 0
  stagnant_count = 0
  best_genotype = None
  prev_best_eval = float('inf')
  best_eval = float('inf')
  
  while generations < max_generations:
    population = selection(population, selection_rate)
    
    new_population = []
    while len(new_population) < population_size:
      parent_indices = np.random.choice(len(population), 2, replace=False)
      parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
      if np.random.rand() < crossover_rate:
        child1, child2 = crossover(parent1, parent2)
        new_population += [mutate(child1, mutation_rate), mutate(child2, mutation_rate)]
      else:
        new_population += [parent1, parent2]
    
    population = new_population
    
    current_eval = evaluate(min(population, key=evaluate))
    
    if current_eval < best_eval:
      best_genotype = min(population, key=evaluate)
      best_eval = current_eval
      stagnant_count = 0
        
    if generations % 100 == 0:
      print(f"Gen {generations}: Best time = {best_eval}")
    
    if prev_best_eval == best_eval:
      stagnant_count += 1
    else:
      stagnant_count = 0
    
    if stagnant_count == max_stagnant_gen:
      print(f"Algorithm stopped. No difference after {max_stagnant_gen} steps")
      break
    
    prev_best_eval = best_eval
    generations += 1

  return best_genotype

best_solution = genetic_algorithm()
print(best_solution)
