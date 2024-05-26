import numpy as np
import benchmark_functions as bf
from opfunu import cec_based 
from src.configuration.config import Config


class Optimization:
    def __init__(self):
        self.config = Config()
        self.maximum = self.config.get_param('algorithm_parameters.maximization')

    def bent_cigar_function(self, x):
        func = cec_based.F32013(ndim=self.conig.get_param('algorithm_parameters.number_of_variables'))
        return 1. / func.evaluate(x) if self.maximum else func.evaluate(x)

    def hypersphere(self, x):
        func = bf.Hypersphere(n_dimensions=self.conig.get_param('algorithm_parameters.number_of_variables'))
        return func(x) if self.maximum else 1. / func(x)
