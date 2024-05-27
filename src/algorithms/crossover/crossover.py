import numpy as np
import random


# Tested, does not work
# Nie wiem czemu, po prostu nie idzie
def discrete_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = parent1
        child2 = parent2

        for i in range(parent1.shape[1]):
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


def elite_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        parents_temp = single_point_crossover([parent1, parent2], offspring_size, ga_instance)
        ratings = [fitness_function(ga_instance, parent) for parent in parents_temp]

        elite_index = np.argsort(ratings)[-2:]
        new_population = parents_temp[elite_index[0]], parents_temp[elite_index[1]]

        offspring.append(new_population[0])
        offspring.append(new_population[1])

        idx += 1

    return np.array(offspring)


# Tested does not work
# Error: ValueError: Sample larger than population or is negative
# Błąd jest tu: ones_index = random.sample(range(len(parent)), ones_counter)
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
    while len(offspring) < offspring_size[0]:
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

    return np.array(offspring[:offspring_size[0]])


def linkage_evolution_crossover(self, specimen1, specimen2):
    if random.random() >= self.crossover_prob:
        self.children.append(specimen1)
        self.children.append(specimen2)
    else:
        child1_chromosomes = []
        child2_chromosomes = []

        for i in range(len(specimen1.specimen)):
            chromosome_1 = specimen1.specimen[i].chromosome
            chromosome_2 = specimen2.specimen[i].chromosome

            child1_segment_list = []
            child2_segment_list = []

            segments = random.randint(1, min(3, len(chromosome_1)))
            split_points = sorted(random.sample(range(1, len(chromosome_1)), segments - 1))
            split_points = [0] + split_points + [len(chromosome_1)]

            for j in range(len(split_points) - 1):
                segment_start = split_points[j]
                segment_end = split_points[j + 1]

                if j % 2 == 0:
                    child1_segment_list.append(chromosome_1[segment_start:segment_end])
                    child2_segment_list.append(chromosome_2[segment_start:segment_end])
                else:
                    child1_segment_list.append(chromosome_2[segment_start:segment_end])
                    child2_segment_list.append(chromosome_1[segment_start:segment_end])

            child1_chromosome = np.concatenate(child1_segment_list)
            child2_chromosome = np.concatenate(child2_segment_list)

            child1_chromosomes.append(child1_chromosome)
            child2_chromosomes.append(child2_chromosome)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)

        self.children.append(child1)
        self.children.append(child2)

# Tested, does not work, 
# Error: TypeError: The output of the crossover step is expected to be of type (numpy.ndarray) but <class 'NoneType'> found.
# IMO nie może być wywoływana inna funkcja na końcu, trzeba alpha tu wrzucić
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

    blend_crossover_alpha(parents, offspring_size, ga_instance)

# Tested, works
def blend_crossover_alpha(parents, offspring_size, ga_instance):
    offspring = []
    alpha = 0.2

    idx = 0
    while len(offspring) < offspring_size[0]:
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

    return np.array(offspring[:offspring_size[0]])

# Tested, works
def blend_crossover_beta(parents, offspring_size, ga_instance):
    offspring = []
    alpha_low = 0.2
    alpha_high = 0.3

    idx = 0
    while len(offspring) < offspring_size[0]:
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

    return np.array(offspring[:offspring_size[0]])

# Tested, works
def average_crossover(parents, offspring_size, ga_instance):
    offspring = []

    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        child1 = np.empty_like(parent1)

        for i in range(parent1.shape[0]):
            child1[i] = (parent1[i] + parent2[i]) / 2

        offspring.append(child1)

        idx += 1

    return np.array(offspring[:offspring_size[0]])
