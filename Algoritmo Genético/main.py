from random import randint

def create_population():
    population = []
    for n in range(30):
        individual = []
        for c in range(30):
            individual.append(randint(-100, 100))
        population.append(individual)
        individual = []
    return population

def tournament(vector):
    selected = []
    for i in range(29):
        vector1 = vector[randint(0, len(vector)) - 1]
        vector2 = vector[randint(0, len(vector)) - 1]
        if sphere(vector1) < sphere(vector2):
            selected.append(vector1)
        else:
            selected.append(vector2)
    return selected

def crossover_one_point(vector):
    new_individuals = []
    while len(new_individuals) < 30:
        probability = randint(0, 100)
        if probability <= 75:
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

def mutation(vector):
    multated_individuals = []
    for c in vector:
        for index, value in enumerate(c):
            probability = randint(0, 100)
            if probability == 42:
                c[index] = randint(-100, 100)
        multated_individuals.append(c)
    return multated_individuals

def get_best_individual(vector):
    best_individual = min(vector, key=sphere)
    return [best_individual, sphere(best_individual)]

def sphere(vector):
  resultado = 0
  
  for i in vector:
    resultado += i**2
  
  return resultado

def main():
    best_individuals = {}
    population = create_population()
    for _ in range(10000):
        selected = tournament(population)
        crossed = crossover_one_point(selected)
        mutated = mutation(crossed)
        best_individual = get_best_individual(mutated)
        population = mutated
        best_individuals[best_individual[1]] = best_individual[0]
        
    for fitness, individual in best_individuals.items():
        print(f"Fitness: {fitness}, Individual: {individual}")

if __name__ == "__main__":
   main()