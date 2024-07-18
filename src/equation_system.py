import numpy as np


class EquationSystem:
    def __init__(self, equations):
        self.equations = equations

    def evaluate(self, x):
        return [np.abs(equation(x)) for equation in self.equations]
