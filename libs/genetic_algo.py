import numpy as np
import random
from geopy.distance import geodesic


def generate_initial_population(pop_size, num_locations):
    # Generate `pop_size` random routes by shuffling location indices
    population = []
    for _ in range(pop_size):
        route = random.sample(range(num_locations), num_locations)  # Random order of locations
        population.append(route)
    return population

def calculate_route_distance(route, locations):
    distance = 0.0
    for i in range(len(route) - 1):
        loc1 = locations[route[i]]
        loc2 = locations[route[i + 1]]
        distance += geodesic(loc1, loc2).km  # Calculate distance in kilometers
    return distance

def fitness(route, locations):
    # Fitness is the inverse of the distance, so shorter routes are "fitter"
    route_distance = calculate_route_distance(route, locations)
    return 1 / route_distance if route_distance > 0 else float('inf')


def selection(population, locations):
    selected = random.sample(population, 2)  # Select two random routes
    # Choose the one with the higher fitness (shorter route)
    return min(selected, key=lambda route: fitness(route, locations))

def crossover(parent1, parent2):
    # Choose crossover points
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    # Create offspring with part of parent1's sequence
    offspring = [None] * size
    offspring[start:end] = parent1[start:end]
    
    # Fill the remaining part from parent2, in the order they appear
    pos = end
    for waypoint in parent2:
        if waypoint not in offspring:
            offspring[pos % size] = waypoint
            pos += 1
            
    return offspring


def mutate(route, mutation_rate=0.1):
    # Swap two waypoints with a small probability (mutation_rate)
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def create_new_population(population, locations, mutation_rate):
    new_population = []
    for _ in range(len(population)):
        # Selection
        parent1 = selection(population, locations)
        parent2 = selection(population, locations)
        
        # Crossover
        offspring = crossover(parent1, parent2)
        
        # Mutation
        offspring = mutate(offspring, mutation_rate)
        
        new_population.append(offspring)
    return new_population

def genetic_algorithm(locations, pop_size=100, generations=500, mutation_rate=0.1):
    population = generate_initial_population(pop_size, len(locations))
    best_route = None
    best_fitness = float('-inf')
    
    for generation in range(generations):
        population = create_new_population(population, locations, mutation_rate)
        
        # Find the best route in the current generation
        for route in population:
            route_fitness = fitness(route, locations)
            if route_fitness > best_fitness:
                best_fitness = route_fitness
                best_route = route
                
        if generation % 10 == 0:  # Print progress every 10 generations
            print(f"Generation {generation}, Best Distance: {1 / best_fitness:.2f} km")
    
    return best_route, 1 / best_fitness


