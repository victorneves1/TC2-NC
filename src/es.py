import numpy as np
from numpy.random import randn, rand

# penalty evaluation function
def evaluate_penalty(individual, equation_system):
    result = equation_system.evaluate(individual)
    penalty_sum = 0
    for r in result:
        penalty_sum += max(0, np.abs(r)) ** 2
    return penalty_sum

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
    # enumerate all dimensions of the point
    for d in range(len(bounds)):
        # check if out of bounds for this dimension
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True

# evolution strategy (mu, lambda) algorithm
def es_core(objective, bounds, n_iter, step_size, mu, lam):
    """
    Evolution Strategy (mu, lambda) algorithm

    :param objective: function
    :param bounds: np.array -- min and max for each variable
    :param n_iter: int -- number of iterations
    :param step_size: float -- maximum step size
    :param mu: int -- number of parents selected
    :param lam: int -- number of children to generate
    """
    best, best_eval = None, 1e+10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    history = []
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(c) for c in population]
        # rank scores in ascending order
        ranks = np.argsort(np.argsort(scores))
        # select the indexes for the top mu ranked solutions
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        # create children from parents
        children = list()
        for i in selected:
            # check if this parent is the best solution ever seen
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                history.append(best_eval)
                print('Epoch %04d, Best: f(%s) = %.5f' % (epoch, best, best_eval), end="\r")
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + randn(len(bounds)) * step_size
                children.append(child)
        # replace population with children
        population = children
    return best, best_eval, history


def evolution_strategy(objective, lb, ub, n_iter, step_size, mu, lam, epochs=10):
    # perform the evolution strategy search
    individuals = []
    scores = []
    histories = []
    bounds = np.asarray([[lb[i], ub[i]] for i in range(len(lb))])
    for _ in range(epochs):
        individual, score, history = es_core(lambda x: evaluate_penalty(x, objective), bounds, n_iter, step_size, mu, lam)
        individuals.append(individual)
        scores.append(score)
        histories.append(history)

    best_score = min(scores)

    best_index = scores.index(best_score)
    best_individual = individuals[best_index]
    return best_index, best_individual, best_score, histories