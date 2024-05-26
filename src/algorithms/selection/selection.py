import random


class GeneticSelection:
    def __init__(self, population, selection_type='best', tournament_size=2, max=False):
        self.population = population
        self.selection_type = selection_type
        self.tournament_size = tournament_size
        self.selection = []
        self.fitness_sum = sum([x.get_fitness() for x in self.population])
        self.max = max
        self.best_chromosomes = []

    def roulette_wheel_selection(self):
        while len(self.best_chromosomes) < self.tournament_size:
            if self.max:
                probabilities = [x.get_fitness() / self.fitness_sum for x in self.population]
            else:
                probabilities = [1 / (x.get_fitness() * self.fitness_sum) for x in self.population]

            cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]
            random_number = random.uniform(0, 1)

            for i, cum_prob in enumerate(cumulative_probabilities):
                if random_number <= cum_prob:
                    if self.population[i] not in self.best_chromosomes:
                        self.best_chromosomes.append(self.population[i])
                        break

    def tournament_selection(self):
        tournament_group = random.sample(self.population, self.tournament_size)
        self.best_chromosomes = sorted(tournament_group, key=lambda x: x.get_fitness(), reverse=self.max)[:self.tournament_size]

    def select_best_chromosomes(self):
        self.best_chromosomes = sorted(self.population, key=lambda x: x.get_fitness(), reverse=self.max)[:self.tournament_size]

    def select(self):
        self.best_chromosomes = []
        if self.selection_type == 'best':
            self.select_best_chromosomes()
        elif self.selection_type == 'roulette_wheel':
            self.roulette_wheel_selection()
        elif self.selection_type == 'tournament':
            self.tournament_selection()

    def get_best_chromosomes(self):
        self.select()
        return self.best_chromosomes

