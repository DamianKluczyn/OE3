import numpy as np
from typing import Tuple
from math import log2, ceil


class Chromosome:
    def __init__(self, boundaries: Tuple[float, float], accuracy: int):
        self.boundaries = boundaries
        self.accuracy = accuracy
        self.chromosome_length = ceil(log2((boundaries[1] - boundaries[0]) * 10 ** accuracy))
        self.chromosome = np.random.choice([0, 1], size=self.chromosome_length, p=[0.5, 0.5])

    def get_boundaries(self) -> Tuple[float, float]:
        return self.boundaries

    def get_accuracy(self) -> int:
        return self.accuracy

    def get_chromosome_length(self) -> int:
        return self.chromosome_length

    def get_chromosome(self) -> np.ndarray:
        return self.chromosome

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome

    def decode_binary_chromosome(self) -> float:
        decimal = sum(value * (2 ** index) for index, value in enumerate(reversed(self.chromosome)))
        return (self.boundaries[0] + decimal *
                (self.boundaries[1] - self.boundaries[0]) / (2 ** self.chromosome_length - 1))

    def __str__(self) -> str:
        return f'Chromosome: {self.chromosome}'

