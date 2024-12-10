import numpy as np
from settings import *
from util import *
from cprunex import *
from profiler import *

"""
from settings following will be used -
1. base_model
2. x_train, y_train, x_test, y_test
3. EPOCH
4. BATCH_SIZE
5. compilation_setting
6. constraints_specs
"""



def generate_gene_space(model) :
    gene_size = model_to_chromosome_size(model)
    gene_space = []
    for i in range(gene_size) :
        gene_space.append(np.arange(0,1,0.1))
    return gene_space



# Define the fitness function
def fitness_func_bound(ga_instance, solution, solution_idx):

    input_shape = base_model.input_shape[1:]
    new_model = custom_channel_prune(base_model,input_shape,solution,1)

    ram , flash, macc = evaluate_hardware_requirements(new_model,(x_train, y_train))

    current_specs = {"ram" : ram  , "flash" : flash  , "macc" : macc }
    isFeasible = checkFeasible(constraints_specs, current_specs)

    if not isFeasible :
        print(current_specs)
        print(solution, -1)
        return -1



    new_model.compile(optimizer = compilation_settings["optimizer"], loss=compilation_settings["loss"], metrics=compilation_settings["metrics"])
    hist =  new_model.fit(x_train, y_train,validation_data=(x_test, y_test),  epochs=EPOCH, batch_size=BATCH_SIZE , verbose = 1, class_weight=class_weight)
    max_val_acc = np.around(np.amax(hist.history[metric]), decimals=3)
    print(solution, max_val_acc, ram)

    return max_val_acc


# Define the fitness function
def fitness_func_reduce_hattrbs(ga_instance, solution, solution_idx):

    input_shape = base_model.input_shape[1:]
    new_model = custom_channel_prune(base_model,input_shape,solution,1)

    ram , flash, macc = evaluate_hardware_requirements(new_model,(x_train, y_train))

    



    new_model.compile(optimizer = compilation_settings["optimizer"], loss=compilation_settings["loss"], metrics=compilation_settings["metrics"])
    hist =  new_model.fit(x_train, y_train,validation_data=(x_test, y_test),  epochs=EPOCH, batch_size=BATCH_SIZE , verbose = 1, class_weight=class_weight)
    acc = np.around(np.amax(hist.history[metric]), decimals=3)


    max_ram = base_specs["ram"]
    normalised_ram_penalty = ram/max_ram

    max_flash = base_specs["flash"]
    normalised_flash_penalty = flash/max_flash

    max_macc = base_specs["macc"]
    normalised_macc_penalty = macc/max_macc



    h_attribute_fitness = ((1-normalised_ram_penalty) + (1-normalised_flash_penalty) + (1-normalised_macc_penalty))/3


    fitness = 0.5* acc + 0.5 * h_attribute_fitness
    print(solution , ram, macc, flash , acc, h_attribute_fitness, fitness)
    
    return fitness

import numpy as np

def generate_feasible_population(pop_size, gene_space, constraints_specs, base_model):
    """
    Generates an initial population of feasible solutions based on hardware constraints.
    
    Parameters:
    - pop_size: int, desired population size (e.g., 10).
    - gene_space: list, the space of allowable values for each gene.
    - constraints_specs: dict, hardware constraint specifications.
    - base_model: model, base model to prune.
    
    Returns:
    - population: list, feasible solutions that meet constraints.
    """
    population = []
    num_genes = len(gene_space)  # Each gene has a corresponding space in gene_space
    
    while len(population) < pop_size:
        # Generate a solution by sampling from gene_space for each gene
        solution = [np.random.choice(gene_space[i]) for i in range(num_genes)]
        
        # Evaluate hardware requirements with the generated solution
        input_shape = base_model.input_shape[1:]
        new_model = custom_channel_prune(base_model, input_shape, solution, 1)
        
        ram, flash, macc = evaluate_hardware_requirements(new_model, (x_train, y_train))
        current_specs = {"ram": ram, "flash": flash, "macc": macc}
        
        # Check if the solution is feasible based on the constraints
        if checkFeasible(constraints_specs, current_specs):
            population.append(solution)
            print(f"Feasible Solution Found: {solution} -> Specs: {current_specs}")
        else:
            print(f"Solution Discarded (Not Feasible): {solution} -> Specs: {current_specs}")
    
    return population






 
