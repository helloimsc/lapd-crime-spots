import random

# Function to calculate distance between two points (Placeholder)
def distance(point1, point2):
    # Assume a function or a distance matrix is available
    distances = {
        ('A', 'B'): 10, ('A', 'C'): 20, ('A', 'D'): 15, ('A', 'E'): 30,
        ('B', 'C'): 25, ('B', 'D'): 35, ('B', 'E'): 20,
        ('C', 'D'): 30, ('C', 'E'): 10,
        ('D', 'E'): 25
    }
    return distances.get((point1, point2), distances.get((point2, point1), 0))

# Fitness function: Total route distance
def fitness(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance(route[i], route[i + 1])
    return total_distance

# Initialize a population of random routes, keeping the fixed start
def initialize_population():
    population = []
    for _ in range(population_size):
        route = [0] + random.sample(points[1:], len(points) - 1)
        population.append(route)
    return population

# Tournament selection
def select_parents(population):
    parents = []
    for _ in range(len(population) // 2):
        tournament = random.sample(population, 5)
        parent = min(tournament, key=fitness)
        parents.append(parent)
    return parents

# Ordered crossover, keeping fixed start point
def crossover(parent1, parent2):
    start = 1
    end = random.randint(1, len(parent1) - 1)
    
    child1 = [fixed_start] + [None] * (len(parent1) - 1)
    child2 = [fixed_start] + [None] * (len(parent2) - 1)

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    fill_child(child1, parent2)
    fill_child(child2, parent1)
    
    return child1, child2

# Helper function to fill remaining genes in child
def fill_child(child, parent):
    current_pos = 1
    for gene in parent:
        if gene not in child:
            while child[current_pos] is not None:
                current_pos += 1
            child[current_pos] = gene

# Mutation function: randomly swap two cities (excluding the fixed start)
def mutate(route):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Genetic algorithm
def genetic_algorithm():
    population = initialize_population()
    
    for generation in range(generations):
        # Selection
        parents = select_parents(population)
        
        # Create new population through crossover
        offspring = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[(i+1) % len(parents)]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])
        
        # Apply mutation
        population = [mutate(child) for child in offspring]
        
        # Evaluate fitness and select the best routes
        population.sort(key=fitness)
        population = population[:population_size]  # Keep only top individuals
    
    # Return the best route
    best_route = min(population, key=fitness)
    return best_route, fitness(best_route)

best_route, best_distance = genetic_algorithm()
print("Best route:", best_route)
print("Total distance:", best_distance)
