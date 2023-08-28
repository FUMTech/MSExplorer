

import random
import math
import json

with open('coupling.json', 'r') as json_file:
    data = json.load(json_file)['couplings']



def calculate_bmc_imc(coupling, state):
    n = len(state) 
    bmc = 0
    imc = 0
    between_max = 0
    within_max = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                modules_i = state[i] 
                modules_j = state[j]
                
                # Calculate between-module coupling
                between_sum = 0
                for m1 in modules_i:
                    for m2 in modules_j:
                        c = get_coupling(m1, m2)
                        between_sum += c
                        between_max = max(between_max, c)
                bmc += between_sum
                
                # Calculate within-module coupling                
                within_sum = 0
                for m1 in modules_i:
                    for m2 in modules_i:
                        if m1 != m2:
                            c = get_coupling(m1, m2)
                            within_sum += c
                            within_max = max(within_max, c)
                imc += within_sum 
                
    if len(state) == 1:
        # Special case - single module
        bmc = 0 
        imc = 1
    else:
        # Normal case        
        bmc =  bmc / (2 * between_max) if between_max != 0 else 0
        imc = imc / (2 * within_max) if within_max != 0 else 0
    
    return bmc, imc


def energy_function(state):
    # Initialize total cohesion and size (for MSI calculation)
    total_size = 0

    # Calculate s_avg sigma
    for microservice in state:
        classes_count = find_number_of_classes_within_microservice(microservice)
        total_size += classes_count **2 # comment later 

    # Calculate average module size and MSI
    s_avg = total_size / all_classes #sigma len(microservice)**2(means number of classes in it) /n
    s_star = all_classes / (0.1 * all_classes)  # assuming m* is 10% of total classes
    w = 0.05  # penalty factor
    MSI = math.exp(-0.5 * ((math.log(s_avg) - math.log(s_star)) / (w * math.log(all_classes))) ** 2)

    BMCI, IMCI = calculate_bmc_imc(data, state)  


    # Define how to combine cohesion and MSI into a single cost
    # This could be a simple sum, a weighted sum, a product, etc.
    cost = (IMCI/(IMCI+BMCI))**0.5 *   MSI**0.5  # Assumed alfa and beta coefficients 1.
    return 1/cost

# Assuming you've provided the 'calculate_bmc_imc' and 'energy_function' already.

def neighbor(state):
    """Generate a neighboring state by moving a class from one microservice to another."""
    # Copy the current state
    new_state = [set(s) for s in state]

    # Choose random source and destination microservices
    src, dest = random.sample(new_state, 2)

    # Move a random class from source to destination
    if src:
        moved_class = random.choice(list(src))
        src.remove(moved_class)
        dest.add(moved_class)

    return new_state

def simulated_annealing(initial_state, initial_temp, alpha, stopping_temp, stopping_iter):
    """The Simulated Annealing algorithm."""
    current_state = initial_state
    current_cost = energy_function(current_state)
    current_temp = initial_temp
    iter_count = 0

    while current_temp > stopping_temp and iter_count < stopping_iter:
        successor = neighbor(current_state)
        successor_cost = energy_function(successor)
        
        delta = successor_cost - current_cost

        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / current_temp):
            current_state, current_cost = successor, successor_cost

        current_temp *= alpha
        iter_count += 1

    return current_state, current_cost

# Example usage
all_classes = {i for i in range(len(data))}  # Assuming classes are identified by indices
initial_state = [all_classes]  # Start with all classes in a single microservice

# These parameters can be tweaked based on the problem requirements and how fast you want the annealing to be
final_state, final_cost = simulated_annealing(initial_state, initial_temp=100, alpha=0.95, stopping_temp=0.01, stopping_iter=10000)

print(final_state)
print(final_cost)
