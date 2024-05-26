from typing import Tuple, List
from .specimen import Specimen


class Population:
    def __init__(self, population_size: int, number_of_chromosomes_per_specimen: int,
                 boundaries: Tuple[float, float], accuracy: int, fitness_function: str):
        self.population_size = population_size
        self.population = [Specimen(number_of_chromosomes_per_specimen, boundaries, accuracy, fitness_function)
                           for _ in range(population_size)]

    def get_population_size(self) -> int:
        return self.population_size

    def get_population(self) -> List[Specimen]:
        return self.population

    def set_population(self, population: List[Specimen]):
        self.population = population

    def fit(self):
        for specimen in self.population:
            specimen.calculate_fitness()

    def __str__(self) -> str:
        result = 'Population:\n'
        for specimen in self.population:
            result += f'\t{specimen.__str__()}'
        return result
