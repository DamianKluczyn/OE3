import numpy as np
import random
import benchmark_functions as bf

func = bf.Ackley(n_dimensions=2)


def fitness_fun(ga_instance, solution):
    fitness = func(solution)
    return 1. / fitness


# Tested, works
def discrete_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for i in range(len(parent1)):
            if random.uniform(0, 1) < 0.5:
                child1[i] = parent1[i]
            else:
                child1[i] = parent2[i]

            if random.uniform(0, 1) < 0.5:
                child2[i] = parent2[i]
            else:
                child2[i] = parent1[i]

        offspring.append(child1)
        offspring.append(child2)

        idx += 1

    return np.array(offspring)


# Tested works
def elite_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        parent1_temp = np.empty_like(parent1)
        parent2_temp = np.empty_like(parent2)
        # single point
        point = random.randint(1, len(parent1) - 1)
        parent1_temp[point:] = parent1[point:]
        parent1_temp[:point] = parent2[:point]
        parent2_temp[point:] = parent2[point:]
        parent2_temp[:point] = parent1[:point]

        parents_temp = [parent1_temp, parent2_temp]
        ratings = [fitness_fun(ga_instance, parent) for parent in parents_temp]

        elite_index = np.argsort(ratings)[-2:]
        new_population = parents_temp[elite_index[0]], parents_temp[elite_index[1]]

        offspring.append(new_population[0])
        offspring.append(new_population[1])

        idx += 1
    return np.array(offspring)


# Tested does not work
# Aby działało musi być na liczbach binarnych
def self_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent = parents[idx % parents.shape[0], :].copy()
        if np.random.random() >= 0.5:  # crossover prob
            offspring.append(parent)
        else:
            child = np.zeros_like(parent)
            ones_counter = sum(parent)
            ones_index = random.sample(range(len(parent)), ones_counter)
            for index in ones_index:
                child[index] = 1

            offspring.append(child)

        idx += 1

    return np.array(offspring)


# Tested, works
def binary_crossover(parents, offspring_size, ga_instance):
    offspring = []

    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        if random.random() >= 0.5:
            offspring.append(parent1)
            offspring.append(parent2)
        else:
            child1 = np.empty_like(parent1)
            child2 = np.empty_like(parent2)

            left, right = 0, len(parent1) - 1

            while left < right - 2:
                center = (left + right) // 2

                TP_1 = np.concatenate((parent1[:center], parent2[center:]))
                TP_2 = np.concatenate((parent2[:center], parent1[center:]))

                NTP_1 = np.sum(TP_1)
                NTP_2 = np.sum(TP_2)

                if NTP_1 > NTP_2:
                    left = center
                else:
                    right = center

            child1 = np.concatenate((parent1[:right], parent2[right:]))
            child2 = np.concatenate((parent2[:right], parent1[right:]))

            offspring.append(child1)
            offspring.append(child2)

        idx += 1

    return np.array(offspring)


# Tested works
def linkage_evolution_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1_segment_list = []
        child2_segment_list = []

        segments = random.randint(1, min(3, len(parent1)))
        split_points = sorted(random.sample(range(1, len(parent1)), segments - 1))
        split_points = [0] + split_points + [len(parent1)]

        for j in range(len(split_points) - 1):
            segment_start = split_points[j]
            segment_end = split_points[j + 1]

            if j % 2 == 0:
                child1_segment_list.append(parent1[segment_start:segment_end])
                child2_segment_list.append(parent2[segment_start:segment_end])
            else:
                child1_segment_list.append(parent2[segment_start:segment_end])
                child2_segment_list.append(parent1[segment_start:segment_end])

        child1 = np.concatenate(child1_segment_list)
        child2 = np.concatenate(child2_segment_list)

        offspring.append(child1)
        offspring.append(child2)

    return np.array(offspring)


# Tested works
def linear_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)
        child3 = np.empty_like(parent2)

        for j in range(len(parent1)):
            child1[j] = 0.5 * parent1[j] + 0.5 * parent2[j]
            child2[j] = 1.5 * parent1[j] - 0.5 * parent2[j]
            child3[j] = -0.5 * parent1[j] + 1.5 * parent2[j]

        best_children = sorted([fitness_fun(ga_instance, child) for child in [child1, child2, child3]])[:2]
        offspring.append(best_children)

    return np.array(offspring)

#Tested works
def linear3_crossover(parents, offspring_size, ga_instance):
    alpha = random.random()
    if alpha >= 0.5:
        beta = (2 * alpha) ** (1 / (0.5 + 1))
    else:
        beta = (1 / (2 * (1 - alpha))) ** (1 / (0.5 + 1))

    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for j in range(len(parent1)):
            child1[j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
            child2[j] = 0.5 * ((1 - beta) * parent1[j] + (1 + beta) * parent2[j])

        offspring.append(child1)
        offspring.append(child2)

    return np.array(offspring)


# Tested #Works
def arithmetic_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    alpha = np.random.uniform(0, 1)

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        parent1_genome = np.zeros_like(parent1)
        parent2_genome = np.zeros_like(parent2)

        for i in range(len(parent1)):
            parent1_genome[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
            parent2_genome[i] = (1 - alpha) * parent2[i] + alpha * parent1[i]

        offspring.append(parent1_genome)
        offspring.append(parent2_genome)

        idx += 1

    return np.array(offspring)


# Tested,works
def center_of_mass_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        center_of_mass = 0

        center_of_mass += np.sum(parent1) + np.sum(parent2)

        center_of_mass = center_of_mass / 2

        child = -1 * parent1 + 2 * center_of_mass

        offspring.append(child)
        idx += 1
    parents = np.array(offspring)
    offspring = []
    alpha = 0.2

    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for i in range(parent1.shape[0]):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            lower_bound = min_val - alpha * range_val
            upper_bound = max_val + alpha * range_val

            child1[i] = random.uniform(lower_bound, upper_bound)
            child2[i] = random.uniform(lower_bound, upper_bound)

        offspring.append(child1)
        offspring.append(child2)

        idx += 1

    return np.array(offspring)


# Tested, works
def imperfect_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        rand_temp = random.uniform(0, 1)
        point = random.randint(1, len(parent1) - 1)
        child1 = [0] * len(parent1)
        child2 = [0] * len(parent2)

        if rand_temp < 0.33:
            child1[:point - 1] = parent1[:point - 1]
            child1[point + 1:] = parent2[point + 1:]
            child1[point] = np.random.uniform(np.min(parent1), np.max(parent1))

            child2[:point - 1] = parent2[:point - 1]
            child2[point + 1:] = parent1[point + 1:]
            child2[point] = np.random.uniform(np.min(parent2), np.max(parent2))
        elif rand_temp < 0.66:
            child1[:point - 1] = parent1[:point - 1]
            child1[point + 1:] = parent2[point + 1:]
            child1[point] = 0

            child2[:point - 1] = parent2[:point - 1]
            child2[point + 1:] = parent1[point + 1:]
            child2[point] = 0
        else:
            child1[:point] = parent1[:point]
            child1[point:] = parent2[point:]

            child2[:point] = parent2[:point]
            child2[point:] = parent1[point:]

        offspring.append(child1)
        offspring.append(child2)
        idx += 1

    return np.array(offspring)


# Tested, works
def blend_crossover_alpha(parents, offspring_size, ga_instance):
    offspring = []
    alpha = 0.2

    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for i in range(parent1.shape[0]):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            lower_bound = min_val - alpha * range_val
            upper_bound = max_val + alpha * range_val

            child1[i] = random.uniform(lower_bound, upper_bound)
            child2[i] = random.uniform(lower_bound, upper_bound)

        offspring.append(child1)
        offspring.append(child2)

        idx += 1

    return np.array(offspring)


# Tested, works
def blend_crossover_beta(parents, offspring_size, ga_instance):
    offspring = []
    alpha_low = 0.2
    alpha_high = 0.3

    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for i in range(parent1.shape[0]):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            lower_bound = min_val - alpha_low * range_val
            upper_bound = max_val + alpha_high * range_val

            child1[i] = random.uniform(lower_bound, upper_bound)
            child2[i] = random.uniform(lower_bound, upper_bound)

        offspring.append(child1)
        offspring.append(child2)

        idx += 1

    return np.array(offspring)


# Tested, works
def average_crossover(parents, offspring_size, ga_instance):
    offspring = []

    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)

        for i in range(parent1.shape[0]):
            child1[i] = (parent1[i] + parent2[i]) / 2

        offspring.append(child1)

        idx += 1

    return np.array(offspring)
