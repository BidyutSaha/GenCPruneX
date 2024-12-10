import os
import tensorflow as tf
import warnings

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = warnings, 2 = info, 3 = errors only

# Suppress specific UserWarnings from TensorFlow and global warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore')

# Set TensorFlow logger to only log errors
tf.get_logger().setLevel('ERROR')

import numpy as np
import pygad

from genCPruneX import *
from settings import *

# Define the gene space for each hyperparameter
gene_space = generate_gene_space(base_model)

# Attach fitness function
fitness_func = fitness_func_reduce_hattrbs


# init populations
# initial_population = generate_feasible_population(pop_size=10, gene_space=gene_space, 
#                                                   constraints_specs=constraints_specs, base_model=base_model)


def on_generation(ga_instance):
    print(f"\n--- Generation {ga_instance.generations_completed} Summary ---")

    parents = ga_instance.last_generation_parents
    print("Selected Parents:")
    for i, parent in enumerate(parents):
        print(f"Parent {i}: {parent} -> Fitness: {ga_instance.last_generation_fitness[i]}")

    # Log offspring after crossover
    crossover_offspring = ga_instance.last_generation_offspring_crossover
    print("\nOffspring after Crossover:")
    for i, child in enumerate(crossover_offspring):
        print(f"Child {i}: {child}")

    # Log mutated offspring
    mutated_offspring = ga_instance.last_generation_offspring_mutation
    print("\nMutated Offspring:")
    for i, mutant in enumerate(mutated_offspring):
        print(f"Mutated Offspring {i}: {mutant}")

    # Display best solution of the generation
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Best solution in generation: {best_solution} -> Fitness: {best_solution_fitness}")




def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()\n" , offspring_crossover)

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()\n", offspring_mutation)

# Define the GA parameters
ga_instance = pygad.GA(num_generations=25,     #50
                       num_parents_mating=4,   #5
                       fitness_func=fitness_func,
                       sol_per_pop=10,         #25
                       num_genes=len(gene_space),
                       gene_space=gene_space,
                       parent_selection_type="sss",  # Stochastic sampling
                       crossover_type="single_point",
                       mutation_type="random",
                       #mutation_probability=0.3,
                       #mutation_percent_genes=90,
                       #mutation_num_genes=2;
                       keep_parents=2,         #4
                       on_generation=on_generation,
                       #initial_population=initial_population,
                       on_crossover  = on_crossover,
                       on_mutation  = on_mutation,
                       mutation_percent_genes = 20,
                       )

# Run the GA
ga_instance.run()

# After the run, get the best solution (best hyperparameters)
solution, solution_fitness, solution_idx = ga_instance.best_solution()

print(f"\nBest Solution after all generations:")
print(solution)

# Plotting the fitness over generations
ga_instance.plot_fitness()
