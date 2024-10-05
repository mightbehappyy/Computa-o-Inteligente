from particle import particle
import random
from math import floor, sqrt

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

    population = create_initial_population(dimensions, limit, population_size)

    for c in range(iterations):
            
        if c == 0:
            best_global_position = population[c].get_position()

        if evaluate(get_best_individual(population, iterations)[1][1]) < evaluate(best_global_position):
            best_global_position_historic.append(evaluate(best_global_position))
            best_global_position = get_best_individual(population, iterations)[1][1]

        for i in range(len(population)):
            if network_or_global == 1:            
                population[i].set_velocity(calculate_new_velocity(population[i], get_network_best_position(population, i)))
            else: 
                population[i].set_velocity(calculate_new_velocity(population[i], get_global_best_position(population)))
            population[i].set_position(sum_two_lists(population[i].get_position(), population[i].get_velocity()))
    
    print(best_global_position_historic)

# def get_geographic_local_best_position(position: particle, individual):
#     min_distance = int('inf')
#     closest_pair = (None, None)
#     for i in range(len(position)):
#         for j in range(i + 1, len(position)):
#             distance = get_distance(position[i].get_position(), position[j].get_position())
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_pair = (position[i].get_position(), position[j].get_position())

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


def calculate_new_velocity(individual: particle, best_current_position: list):
    return (
        sum_lists(multiply_list(individual.get_velocity(), 0.8), 
        multiply_list(substract_lists(individual.pbest, individual.position), 2.05 * random.random()), 
        multiply_list(substract_lists(best_current_position, individual.position), 2.05 * random.random()))
        )

def get_distance(vector1: list, vector2: list):
    return sqrt(sum((x - y) ** 2 for x, y in zip(vector1, vector2)))

def sum_lists(list1:list, list2: list, list3: list):
    result_list = []
    for l1, l2, l3 in zip(list1, list2, list3):
        result_list.append(l1 + l2 + l3)
    return result_list

def substract_lists(list1: list, list2: list):
    result_list = []
    for l1, l2 in zip(list1, list2):
        result_list.append(l1 - l2)
    return result_list

def sum_two_lists(list1: list, list2:list):
    result_list = []
    for l1, l2 in zip(list1, list2):
        result_list.append(l1 + l2)
    return result_list

def multiply_list(list, value):
    result_list = []
    for c in list:
        result_list.append(floor(c * value))
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
    execute_pso_algorithm(30, (-100, 100), 30, 10000, 2)