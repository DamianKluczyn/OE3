import logging
import pygad
import numpy as np
import benchmark_functions as bf
import random

from src.optimization.optimization import Optimization
from src.configuration.config import Config
from src.algorithms.crossover.crossover import Crossover

if __name__ == '__main__':
    config = Config()

    optimization = Optimization()
    fitness_function = optimization.optimization()

    selection_method = config.get_param("algorithm_parameters.selection_method")
    mutation_method = config.get_param("algorithm_parameters.mutation_method")

    crossover = Crossover()
    crossover_method = crossover.crossover()

    range_low = config.get_param("algorithm_parameters.range_low")
    range_high = config.get_param("algorithm_parameters.range_high")
    gene_type = config.get_param("algorithm_parameters.gene_type")

    number_of_genes = config.get_param("algorithm_parameters.number_of_genes")

    population_size = config.get_param("algorithm_parameters.population_size")
    generations = config.get_param("algorithm_parameters.number_of_generations")

    parents_mating = config.get_param("algorithm_parameters.parents_mating")

    level = logging.DEBUG
    name = 'logfile.txt'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)


    def on_generation(ga_instance):
        ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
        solution, solution_fitness, solution_idx = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness)
        ga_instance.logger.info("Best    = {fitness}".format(fitness=1. / solution_fitness))
        ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))

        tmp = [1. / x for x in
               ga_instance.last_generation_fitness]  # ponownie odwrotność by zrobić sobie dobre statystyki

        ga_instance.logger.info("Min    = {min}".format(min=np.min(tmp)))
        ga_instance.logger.info("Max    = {max}".format(max=np.max(tmp)))
        ga_instance.logger.info("Average    = {average}".format(average=np.average(tmp)))
        ga_instance.logger.info("Std    = {std}".format(std=np.std(tmp)))
        ga_instance.logger.info("\r\n")

    ga_instance = pygad.GA(
        fitness_func=fitness_function,
        parent_selection_type=selection_method,
        mutation_type=mutation_method,
        crossover_type=crossover_method,

        init_range_low=range_low,
        init_range_high=range_high,
        gene_type=int,

        num_genes=number_of_genes,
        sol_per_pop=population_size,
        num_generations=generations,
        num_parents_mating=3
    )

    ga_instance.run()

    best = ga_instance.best_solution()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1. / solution_fitness))

    # sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
    ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
    ga_instance.plot_fitness()
