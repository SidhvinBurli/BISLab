import numpy as np

# Define the objective function to optimize
def objective_function(x):
    return np.sum(x**2)  # Example: Minimize the sum of squares

# Define the neighborhood structure (Moore Neighborhood)
def get_neighbors(grid, i, j):
    rows, cols = grid.shape[0], grid.shape[1]
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = (i + di) % rows, (j + dj) % cols
            neighbors.append(grid[ni, nj])
    return neighbors

# Parallel Cellular Algorithm
def parallel_cellular_algorithm(objective_function, grid_size=(5, 5), dim=2, bounds=(-5, 5), max_iterations=100):
    lower_bound, upper_bound = bounds
    rows, cols = grid_size
    
    # Initialize grid with random positions
    grid = np.random.uniform(lower_bound, upper_bound, (rows, cols, dim))
    fitness_grid = np.apply_along_axis(objective_function, 2, grid)
    
    # Main loop
    for iteration in range(max_iterations):
        new_grid = grid.copy()
        
        # Update each cell based on neighbors
        for i in range(rows):
            for j in range(cols):
                neighbors = get_neighbors(grid, i, j)
                best_neighbor = min(neighbors, key=lambda x: objective_function(x))
                
                # Update the cell's position based on its best neighbor
                new_grid[i, j] = (grid[i, j] + best_neighbor) / 2
        
        # Update grid and fitness values
        grid = np.clip(new_grid, lower_bound, upper_bound)
        fitness_grid = np.apply_along_axis(objective_function, 2, grid)
        
        # Track the best solution
        best_solution = grid[np.unravel_index(np.argmin(fitness_grid), fitness_grid.shape[:2])]
        best_fitness = objective_function(best_solution)
        
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
    
    return best_solution, best_fitness

# Parameters
grid_size = (5, 5)  # Size of the grid
dim = 2             # Dimensionality of the solution space
bounds = (-10, 10)  # Lower and upper bounds
max_iterations = 50 # Maximum number of iterations

# Run the Parallel Cellular Algorithm
best_solution, best_fitness = parallel_cellular_algorithm(objective_function, grid_size=grid_size, dim=dim, bounds=bounds, max_iterations=max_iterations)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
