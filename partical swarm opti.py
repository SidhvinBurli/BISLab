import random
import math
import copy
import sys

# --------- Fitness functions ---------

# Rastrigin function
def fitness_rastrigin(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitnessVal

# Sphere function
def fitness_sphere(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi * xi)
    return fitnessVal

# -------------------------

# Particle class
class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)

        # Initialize position of the particle with 0.0 value
        self.position = [0.0 for i in range(dim)]

        # Initialize velocity of the particle with 0.0 value
        self.velocity = [0.0 for i in range(dim)]

        # Initialize best particle position with 0.0 value
        self.best_part_pos = [0.0 for i in range(dim)]

        # Loop to calculate random position and velocity
        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) * self.rnd.random() + minx)

        # Compute fitness of the particle
        self.fitness = fitness(self.position)  # Current fitness

        # Initialize best position and fitness of this particle
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness  # Best fitness

# Particle Swarm Optimization function
def pso(fitness, max_iter, n, dim, minx, maxx):
    # Hyperparameters
    w = 0.729  # Inertia
    c1 = 1.49445  # Cognitive (particle)
    c2 = 1.49445  # Social (swarm)

    rnd = random.Random(0)

    # Create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

    # Compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for i in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max  # Swarm best

    # Find the best particle of the swarm and its fitness
    for i in range(n):  # Check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)

    # Main loop of PSO
    Iter = 0
    while Iter < max_iter:
        # Print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print(f"Iter = {Iter} best fitness = {best_swarm_fitnessVal:.3f}")

        for i in range(n):  # Process each particle
            # Compute new velocity for current particle
            for k in range(dim):
                r1 = rnd.random()  # Randomization
                r2 = rnd.random()

                swarm[i].velocity[k] = (
                    w * swarm[i].velocity[k] +
                    c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k]) +
                    c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k])
                )

                # Clip velocity if it's out of bounds
                if swarm[i].velocity[k] < minx:
                    swarm[i].velocity[k] = minx
                elif swarm[i].velocity[k] > maxx:
                    swarm[i].velocity[k] = maxx

            # Compute new position using new velocity
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]

            # Compute fitness of the new position
            swarm[i].fitness = fitness(swarm[i].position)

            # Check if new position is the best for the particle
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # Check if new position is the best overall
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)

        Iter += 1

    return best_swarm_pos

# Driver code for Rastrigin function
print("\nBegin particle swarm optimization on Rastrigin function\n")
dim = 3
fitness = fitness_rastrigin

print(f"Goal is to minimize Rastrigin's function in {dim} variables")
print("Function has known min = 0.0 at ", end="")
print(", ".join(["0"] * (dim-1)) + ", 0)")

num_particles = 50
max_iter = 100

print(f"Setting num_particles = {num_particles}")
print(f"Setting max_iter = {max_iter}")
print("\nStarting PSO algorithm\n")

best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nPSO completed\n")
print("\nBest solution found:")
print([f"{best_position[k]:.6f}" for k in range(dim)])
fitnessVal = fitness(best_position)
print(f"Fitness of best solution = {fitnessVal:.6f}")

print("\nEnd particle swarm for Rastrigin function\n")

# ------------------------------

# Driver code for Sphere function
print("\nBegin particle swarm optimization on Sphere function\n")
dim = 3
fitness = fitness_sphere

print(f"Goal is to minimize Sphere function in {dim} variables")
print("Function has known min = 0.0 at ", end="")
print(", ".join(["0"] * (dim-1)) + ", 0)")

num_particles = 50
max_iter = 100

print(f"Setting num_particles = {num_particles}")
print(f"Setting max_iter = {max_iter}")
print("\nStarting PSO algorithm\n")

best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nPSO completed\n")
print("\nBest solution found:")
print([f"{best_position[k]:.6f}" for k in range(dim)])
fitnessVal = fitness(best_position)
print(f"Fitness of best solution = {fitnessVal:.6f}")

print("\nEnd particle swarm for Sphere function\n")
