import random


class Mutation:
    def __init__(self, mutation_rate, mutation_method):
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method

    def boundary_mutation(self, specimen):

        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()
            for gene_index, gene in enumerate(chromosome):
                if random.random() < self.mutation_rate:
                    chromosome_copy = chromosome.copy()
                    chromosome_copy[gene_index] = 1 - gene
                    specimen.get_specimen()[i].set_chromosome(chromosome_copy)

        return specimen

    def one_point_mutation(self, specimen):
        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()
            if random.random() < self.mutation_rate:
                point = random.randint(0, len(chromosome) - 1)
                chromosome_copy = chromosome.copy()
                chromosome_copy[point] = 1 - chromosome_copy[point]
                specimen.get_specimen()[i].set_chromosome(chromosome_copy)

        return specimen

    def two_point_mutation(self, specimen):
        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()
            if random.random() < self.mutation_rate:
                points = random.sample(range(len(chromosome)), 2)
                points.sort()
                chromosome_copy = chromosome.copy()
                for j in range(points[0], points[1] + 1):
                    chromosome_copy[j] = 1 - chromosome_copy[j]
                specimen.get_specimen()[i].set_chromosome(chromosome_copy)

        return specimen

    def mutate(self, specimen):
        if self.mutation_method == 'boundary_mutation':
            return self.boundary_mutation(specimen)
        elif self.mutation_method == 'one_point_mutation':
            return self.one_point_mutation(specimen)
        elif self.mutation_method == 'two_point_mutation':
            return self.two_point_mutation(specimen)
