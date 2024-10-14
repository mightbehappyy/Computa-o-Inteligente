from particle import particle
import random
from math import floor, sqrt
from matplotlib import pyplot as plt
def create_initial_population(dimensions: int, limit: tuple, population_size: int):
    population = {}

    for x in range(population_size):
        individual = particle(dimensions, limit)
        individual.generate_pos_vel()
        population[x] = individual
    
    return population

def execute_pso_algorithm(dimensions: int, limit: tuple, population_size: int, iterations: int, network_or_global: int):
    best_global_position = []
    best_global_position_historic = []
    generation = []
    population = create_initial_population(dimensions, limit, population_size)
    drag_factor = 1.0
    count = 0
    for c in range(iterations):
        if c == 0:
            best_global_position = get_global_best_position(population)
        
        current_best_value = evaluate(get_best_individual(population, iterations)[1][1])
        if current_best_value < evaluate(best_global_position):
            best_global_position = get_best_individual(population, iterations)[1][1]

        for i in range(len(population)):
            individual = population[i]

            if network_or_global == 1:
                new_velocity = calculate_new_velocity(individual, get_network_best_position(population, i), i, drag_factor)
            else:
                new_velocity = calculate_new_velocity(individual, get_global_best_position(population), i, drag_factor)

            individual.set_velocity(new_velocity)
            
            new_position = sum_two_lists(individual.get_position(), individual.get_velocity())
            individual.set_position(new_position)

            
            if evaluate(individual.get_position()) < evaluate(individual.get_pbest()):
                individual.set_pbest(individual.get_position())


        best_global_position_historic.append(evaluate(best_global_position))
        generation.append(count)
        drag_factor *= 0.9999
        print(evaluate(best_global_position))
        count += 1
        if evaluate(best_global_position) == 0:
            print(best_global_position_historic)
            break

    plot_convergence_graph(best_global_position_historic, generation, count)

        



def get_geographic_local_best_position(population, position):
    test_population = population
    three_closest = []


def get_network_best_position(population, position):
    current_individual = population[position].get_position()

    if position - 1 < 0:
        neighbour_1 = population[len(population) - 1].get_position()
    else:
        neighbour_1 = population[position - 1].get_position()

    if position + 1 > len(population) - 1:
        neighbour_2 = population[0].get_position()
    else:  
        neighbour_2 = population[position + 1].get_position()

    if evaluate(current_individual) < evaluate(neighbour_1) and evaluate(current_individual) < evaluate(neighbour_2):
        return current_individual
    
    elif evaluate(neighbour_1) < evaluate(current_individual) and evaluate(neighbour_1) < evaluate(neighbour_2):
        return neighbour_1
    else:
        return neighbour_2

def get_global_best_position(population):
    best_position = population[0].get_position()
    for c in range(len(population)):
        if evaluate(population[c].get_position()) < evaluate(best_position):
            best_position = population[c].get_position()
    return best_position


def calculate_new_velocity(individual: particle, best_current_position: list, i, drag_factor):
    new_velocity = []
    for i in range(len(individual.velocity)):
        new_velocity.append(floor( drag_factor * ((individual.velocity[i] * 0.8) + 
        (2.05 * random.random() * (individual.pbest[i] - individual.position[i])) + 
        (2.05 * random.random() * (best_current_position[i] - individual.position[i])))))

    return new_velocity



def plot_convergence_graph(best_fitness_over_time, generation, totalGens):
    plt.figure(figsize=(8, 5))
    plt.plot(generation, best_fitness_over_time, label=f'Melhor fitness, gens: {totalGens}', color='blue')
    plt.title('Gráfico de convergência')
    plt.xlabel('Iteração')
    plt.ylabel('Valor do fitness')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_distance(vector1: list, vector2: list):
    return sqrt(sum((x - y) ** 2 for x, y in zip(vector1, vector2)))

def sum_two_lists(list1: list, list2:list):
    result_list = []
    for l1, l2 in zip(list1, list2):
        result_list.append(l1 + l2)
    return result_list
        
def get_best_individual(population, generation):
    individuals = {}
    for key, obj in population.items():
       individuals[generation] = evaluate(obj.get_position()), obj.get_position()
   
    return min(individuals.items(), key=lambda item: item[1])

def evaluate(vector):
  resultado = 0
  for i in vector:
    resultado += i**2
  
  return resultado

if __name__ == "__main__":
    execute_pso_algorithm(30, (-100, 100), 60, 5500, 1)