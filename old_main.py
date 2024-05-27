import random

import numpy as np

from src.algorithms.inversion.inversion_mutation import InversionMutation
from src.algorithms.crossover.crossover import Crossover
from src.algorithms.mutation.mutation import Mutation
from src.algorithms.selection.selection import GeneticSelection
from src.configuration.config import Config
from src.population.population import Population
from src.algorithms.selection.elite import Elite
from src.utilities.generating_files import DataSaver
#from src.gui import gui

import time


def main_function():
    config = Config()

    start_range = config.get_param('algorithm_parameters.start_range_a')
    end_range = config.get_param('algorithm_parameters.end_range_b')
    population_size = config.get_param('algorithm_parameters.population_size')
    binary_precision = config.get_param('algorithm_parameters.binary_precision')
    number_of_variables = config.get_param('algorithm_parameters.number_of_variables')
    number_of_epochs = config.get_param('algorithm_parameters.number_of_epochs')
    fitness_function = config.get_param('algorithm_parameters.fitness_function')
    selection_method = config.get_param('algorithm_parameters.selection_method')
    selection_count = config.get_param('algorithm_parameters.selection_parameters.tournament_size')
    maximum = config.get_param('algorithm_parameters.maximization')
    use_elite = config.get_param('algorithm_parameters.elite_strategy.use_elite_strategy')
    elite_count = config.get_param('algorithm_parameters.elite_strategy.elite_count')
    crossover_prob = config.get_param('algorithm_parameters.crossover_probability')
    crossover_method = config.get_param('algorithm_parameters.crossover_method')
    mutation_prob = config.get_param('algorithm_parameters.mutation_probability')
    inversion_prob = config.get_param('algorithm_parameters.inversion_probability')
    mutation_method = config.get_param('algorithm_parameters.mutation_method')


    population = Population(population_size, number_of_variables, (start_range, end_range), binary_precision, fitness_function)
    start_time = time.time()
    plot_list = []
    plot_list_mean = []
    plot_list_std = []
    if maximum:
        max_fitness = -np.inf
        for specimen in population.get_population():
            if max_fitness < specimen.get_fitness():
                x = specimen.get_decoded_specimen()
                max_fitness = specimen.get_fitness()
    else:
        max_fitness = np.inf
        for specimen in population.get_population():
            if max_fitness >= specimen.get_fitness():
                x = specimen.get_decoded_specimen()
                max_fitness = specimen.get_fitness()


    plot_list.append([0, max_fitness])
    plot_list_mean.append([0, np.mean([x.get_fitness() for x in population.get_population()])])
    plot_list_std.append([0, np.std([x.get_fitness() for x in population.get_population()])])
    #zapisywanie epoki nr 0 i jej fitness function

    for epoch in range(number_of_epochs):

        selection = GeneticSelection(population=population.get_population(), selection_type=selection_method, tournament_size=selection_count, max=maximum)
        selected_population = selection.get_best_chromosomes()

        if use_elite:
            elites = Elite(population=selected_population, elite_count=elite_count, max=maximum)
            elite_population = elites.select_elite()
            for elite in elite_population:
                selected_population.remove(elite)

        print(f'selected: {selected_population}')

        crossover = Crossover(crossover_prob=crossover_prob, cross_method=crossover_method)
        crossed_population = selected_population.copy()
        while len(crossed_population) < population_size - elite_count:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover.cross(parent1, parent2)
            crossed_population.append(child1)
            crossed_population.append(child2)

        for i in range(len(crossed_population)):
            mutation = Mutation(mutation_rate=mutation_prob, mutation_method=mutation_method)
            crossed_population[i] = mutation.mutate(crossed_population[i])

        for i in range(len(crossed_population)):
            inversion = InversionMutation(inversion_prob=inversion_prob)
            crossed_population[i] = inversion.inversion_mutation(crossed_population[i])

        if use_elite:
            crossed_population.extend(elite_population)
        population.set_population(crossed_population)
        population.fit()
        if maximum:
            for specimen in population.get_population():
                if max_fitness < specimen.get_fitness():
                    x = specimen.get_decoded_specimen()
                    max_fitness = specimen.get_fitness()
        else:
            for specimen in population.get_population():
                if max_fitness >= specimen.get_fitness():
                    x = specimen.get_decoded_specimen()
                    max_fitness = specimen.get_fitness()




        print(f"Epoch {epoch + 1}/{number_of_epochs} completed.")
        plot_list.append([epoch+1, max_fitness])
        plot_list_mean.append([epoch+1, np.mean([x.get_fitness() for x in population.get_population()])])
        plot_list_std.append([epoch+1, np.std([x.get_fitness() for x in population.get_population()])])
    # print(f"F({x}) = {max_fitness}")

    data_saver = DataSaver()
    data_saver.plot_and_save(plot_list, selection_method + '_' + crossover_method + '_' + mutation_method)
    data_saver.plot_and_save(plot_list_mean, selection_method + '_' + crossover_method + '_' + mutation_method + '_MEAN')
    data_saver.plot_and_save(plot_list_std, selection_method + '_' + crossover_method + '_' + mutation_method + '_STD')
    data_saver.save_to_file(plot_list, selection_method + '_' + crossover_method + '_' + mutation_method)
    data_saver.save_to_file(plot_list_mean, selection_method + '_' + crossover_method + '_' + mutation_method + '_MEAN')
    data_saver.save_to_file(plot_list_std, selection_method + '_' + crossover_method + '_' + mutation_method + '_STD')


    # wyswietlic znalezione min/max (zaleznie od checkboxa) z wszystkich pokolen
    # wygenerowac wykres na podstawie txt
    end_time = time.time()
    exec_time = end_time - start_time
    # print(f"Algorithm finished in {exec_time} seconds.")
    return exec_time, x, max_fitness
