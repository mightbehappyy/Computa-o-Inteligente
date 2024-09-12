from random import randint, uniform, choices
import math

solver: int

def create_population(individuals: int = 30, genes: int = 30, genes_limit = (-100, 100)):
    population = []
    for n in range(individuals):
        individual = []
        for c in range(genes):
            individual.append(randint(genes_limit[0], genes_limit[1]))
        population.append(individual)
        individual = []
    return population

def tournament(vector):
    selected = []
    for i in range(30):
        vector1 = vector[randint(0, len(vector)) - 1]
        vector2 = vector[randint(0, len(vector)) - 1]
        if solver_function(vector1) < solver_function(vector2):
            selected.append(vector1)
        else:
            selected.append(vector2)
    return selected

def proportional_to_fitness(vector):
    weights = []
    for c in vector:
        weights.append(1/solver_function(c))
    
    return choices(vector, weights=weights, k=len(vector))

def crossover_one_point(vector, crossover_rate):
    new_individuals = []
    while len(new_individuals) < 30:
        probability = randint(0, 100)
        if probability <= crossover_rate:

            random_individual1 = randint(0, len(vector) - 1)
            random_individual2 = randint(0, len(vector) - 1)
            point = randint(1, len(vector[random_individual1]) - 1)

            individual1 = vector[random_individual1]
            individual2 = vector[random_individual2]

            individual1_first_part = individual1[:point]
            individual1_second_part = individual1[point:]
            individual2_first_part = individual2[:point]
            individual2_second_part = individual2[point:]
                
            new_individuals.append(individual1_first_part + individual2_second_part)
            new_individuals.append(individual2_first_part + individual1_second_part)
    return new_individuals

def crossover_two_points(vector, crossover_rate):
    new_individuals = []
    while len(new_individuals) < 2:
        probability = randint(0, 100)
        if probability <= crossover_rate:
            
            random_individual1 = randint(0, len(vector) - 1)
            random_individual2 = randint(0, len(vector) - 1)

            point = randint(1, len(vector[random_individual1]) - 1)
            point2 = randint(1, len(vector[random_individual1]) - 1)

            if point == point2:
                point = randint(1, len(vector[random_individual1]) - 1)
                point2 = randint(1, len(vector[random_individual1]) - 1)
            if point > point2:
                point, point2 = point2, point
                
            individual1 = vector[random_individual1]
            individual2 = vector[random_individual2]

            individual1_first_third = individual1[:point]
            individual1_second_third = individual1[point:point2]
            individual1_last_third = individual1[point2:]

            individual2_first_third = individual2[:point]
            individual2_second_third = individual2[point:point2]
            individual2_last_third = individual2[point2:]
            
            new_individuals.append(individual1_first_third + individual2_second_third + individual1_last_third)
            new_individuals.append(individual2_first_third + individual1_second_third + individual2_last_third)
            return new_individuals

def mutation(vector, mutation_rate):
    multated_individuals = []
    for c in vector:
        for index, value in enumerate(c):
            probability = uniform(0, 100)
            if probability <= mutation_rate:
                c[index] = randint(-100, 100)
        multated_individuals.append(c)
    return multated_individuals

def get_best_individual(vector):
    best_individual = min(vector, key=solver_function)
    return [best_individual, solver_function(best_individual)]

def optimization(generations: int, 
                 individuals: int, 
                 genes: int, 
                 genes_limit:tuple, 
                 crossover_rate: int, 
                 mutation_rate: int, 
                 crossover_type: int,
                 selection_type: int
                 ):
    
    best_individuals = {}

    population = create_population(individuals=individuals, genes=genes, genes_limit=genes_limit)

    for _ in range(generations):
        if selection_type == 1:
            selected = proportional_to_fitness(population)
        else:
            selected = tournament(population)

        if crossover_type == 1:
            crossed = crossover_two_points(selected, crossover_rate=crossover_rate)
        else:
            crossed = crossover_one_point(selected, crossover_rate=crossover_rate)

        mutated = mutation(crossed, mutation_rate)
        best_individual = get_best_individual(mutated)
        population = mutated
        best_individuals[best_individual[1]] = best_individual[0]
        
    for fitness, individual in best_individuals.items():
        print(f"Fitness: {fitness}, Individual: {individual}")    



def main():
    optimization(20000, 30, 30, (-100, 100), 75, 0.9,  0, 0)
    
# 1 para esfera
# 2 para rastrigin
# 3 para rosenbrock
solver = 3
def solver_function(vector):
    resultado = 0
    if solver == 1:
        for i in vector:
            resultado += i**2

    elif solver == 2:
        for i in vector:
            numero = 2*3.1415*i
            p = (numero/180)*math.pi
            resultado+= (i**2) - (10 * math.cos(p)) + 10
    elif solver == 3:
        for i in range(0,(len(vector)-1)):
            resultado += 100*(vector[i+1] - vector[i]**2)**2 + (vector[i] - 1)**2

    return resultado
    

if __name__ == "__main__":
   main()
   