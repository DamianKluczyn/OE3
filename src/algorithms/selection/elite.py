class Elite:
    def __init__(self, population, elite_count, max):
        self.population = population
        self.elite_count = elite_count
        self.max = max

    def select_elite(self):
        sorted_population = sorted(self.population, key=lambda x: x.get_fitness(), reverse=self.max)
        elite = sorted_population[:self.elite_count]
        return elite

