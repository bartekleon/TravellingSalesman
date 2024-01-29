import random

# Define the objective function to minimize
def objective_function(x, y):
  return (1.5 - x - x * y)**2 + (2.25 - x + x * y ** 2)**2 + (2.625 - x + x * y ** 3)**2

# Define the Particle class
class Particle:
  def __init__(self):
    self.x = random.uniform(-4.5, 4.5)
    self.y = random.uniform(-4.5, 4.5)
    self.velocity_x = random.uniform(-1, 1)
    self.velocity_y = random.uniform(-1, 1)
    self.best_x = self.x
    self.best_y = self.y
    self.best_value = objective_function(self.x, self.y)

# Define the PSO function
def particle_swarm_optimization(num_particles, max_iterations, inertia_weight, cognitive_weight, social_weight):
  particles = [Particle() for _ in range(num_particles)]

  # Initialize global best with the position of the first particle
  global_best_x = particles[0].x
  global_best_y = particles[0].y
  global_best_value = objective_function(global_best_x, global_best_y)

  for _ in range(max_iterations):
    for particle in particles:
      # Update particle's velocity
      particle.velocity_x = (inertia_weight * particle.velocity_x +
                              cognitive_weight * random.random() * (particle.best_x - particle.x) +
                              social_weight * random.random() * (global_best_x - particle.x))
      particle.velocity_y = (inertia_weight * particle.velocity_y +
                              cognitive_weight * random.random() * (particle.best_y - particle.y) +
                              social_weight * random.random() * (global_best_y - particle.y))

      # Update particle's position
      particle.x += particle.velocity_x
      particle.y += particle.velocity_y

      # Update personal best if necessary
      value = objective_function(particle.x, particle.y)
      if value < particle.best_value:
        particle.best_x = particle.x
        particle.best_y = particle.y
        particle.best_value = value

      # Update global best if necessary
      if value < global_best_value:
        global_best_x = particle.x
        global_best_y = particle.y
        global_best_value = value

  return global_best_x, global_best_y, global_best_value

if __name__ == "__main__":
  num_particles = 50
  max_iterations = 100
  inertia_weight = 0.3
  cognitive_weight = 1.5
  social_weight = 0.6

  best_x, best_y, best_value = particle_swarm_optimization(num_particles, max_iterations, inertia_weight, cognitive_weight, social_weight)

  print(f"Best solution found: x = {best_x}, y = {best_y}")
  print(f"Minimum value found: {best_value}")
