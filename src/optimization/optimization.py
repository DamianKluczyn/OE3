import numpy as np
import benchmark_functions as bf
from opfunu import cec_based
from src.configuration.config import Config


class Optimization:
    def __init__(self):
        self.config = Config()
        self.maximum = self.config.get_param('algorithm_parameters.maximization')

    def bent_cigar(self, ga_instance, solution, solution_idx):
        func = cec_based.F32013(ndim=self.config.get_param('algorithm_parameters.number_of_genes'))
        return func.evaluate(solution)

    def hypersphere(self, ga_instance, solution, solution_idx):
        func = bf.Hypersphere(n_dimensions=self.config.get_param('algorithm_parameters.number_of_genes'))
        return func(solution)

    def optimization(self):
        option = self.config.get_param('algorithm_parameters.fitness_function')
        if option == 'bent_cigar':
            return self.bent_cigar
        elif option == 'hypersphere':
            return self.hypersphere
        else:
            return 0
