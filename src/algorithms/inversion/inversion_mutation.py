import random
import numpy as np


class InversionMutation:
    def __init__(self, inversion_prob):
        self.inversion_prob = inversion_prob

    def inversion_mutation(self, specimen):
        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()
            mutated_chromosome = chromosome.copy()

            if random.random() < self.inversion_prob:
                start = random.randint(0, len(chromosome) - 1)
                end = random.randint(start, len(chromosome))
                mutated_chromosome[start:end] = list(reversed(chromosome[start:end]))

            specimen.get_specimen()[i].set_chromosome(np.array(mutated_chromosome))

        return specimen
