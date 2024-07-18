import numpy as np
from numpy.random import rand, randn, randint

def in_bounds(point, bounds):
    for d in range(len(bounds)):
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True

def evaluate_comde(individual, equation_system, bounds):
    result = equation_system.evaluate(individual)
    constraints_violations = [max(0, np.abs(r)) for r in result]
    return np.sum(constraints_violations), constraints_violations

def comde_mutation(parent1, parent2, parent3, F, bounds):
    mutant = parent1 + F * (parent2 - parent3)
    for i in range(len(mutant)):
        if mutant[i] < bounds[i, 0]:
            mutant[i] = bounds[i, 0] + rand() * (parent1[i] - bounds[i, 0])
        elif mutant[i] > bounds[i, 1]:
            mutant[i] = bounds[i, 1] + rand() * (bounds[i, 1] - parent1[i])
    return mutant

def comde(objective, bounds, n_iter, pop_size, F_initial, F_final, CR_initial, CR_final, k, R, epochs=10):
    best, best_eval = None, 1e+10
    history = []

    population = [bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]) for _ in range(pop_size)]
    
    for G in range(n_iter):
        CR = CR_initial + (CR_final - CR_initial) * (1 - G / n_iter)**4
        if G / n_iter <= R:
            F = F_final + (F_initial - F_final) * (1 - G / n_iter)**k
        else:
            F = F_final

        scores = [evaluate_comde(ind, objective, bounds) for ind in population]
        scores.sort(key=lambda x: x[0])

        best_index = np.argmin([score[0] for score in scores])
        worst_index = np.argmax([score[0] for score in scores])
        best_individual = population[best_index]
        worst_individual = population[worst_index]

        new_population = []
        for i in range(pop_size):
            if rand() <= 0.5:  # Use New Directed Mutation Scheme
                r1 = randint(pop_size)
                while r1 == best_index or r1 == worst_index or r1 == i:
                    r1 = randint(pop_size)
                F_l = np.random.uniform(0.4, 0.6)
                j_rand = randint(len(bounds))
                u = np.copy(population[i])
                for j in range(len(bounds)):
                    if rand() < CR or j == j_rand:
                        u[j] = population[r1][j] + F_l * (best_individual[j] - worst_individual[j])
            else:  # Use Modified Basic Mutation Scheme
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                F_g = np.random.choice([-1.0, 0.0, 1.0])
                j_rand = randint(len(bounds))
                u = np.copy(population[i])
                for j in range(len(bounds)):
                    if rand() < CR or j == j_rand:
                        u[j] = population[r1][j] + F_g * (population[r2][j] - population[r3][j])

            if evaluate_comde(u, objective, bounds)[0] < evaluate_comde(population[i], objective, bounds)[0]:
                new_population.append(u)
            else:
                new_population.append(population[i])

        population = new_population

        for i in range(pop_size):
            current_eval = evaluate_comde(population[i], objective, bounds)[0]
            if current_eval < best_eval:
                best, best_eval = population[i], current_eval
                print('Generation %04d, Best: f(%s) = %.5f' % (G, best, best_eval), end="\r")
        history.append(best_eval)
    best_individual = [best, best_eval]    
    return best_individual, history


def comde_algorithm(
    equation_system, bounds, n_iter, pop_size, F_initial, F_final, CR_initial, CR_final, k, R, epochs=10
):

    # run the genetic algorithm 30 times and collect the results
    all_histories = []
    all_individuals = []
    for epoch in range(epochs):
        best_individual, history = comde(equation_system, bounds, n_iter, pop_size, F_initial, F_final, CR_initial, CR_final, k, R, epochs)
        all_histories.append(history)
        all_individuals.append(best_individual)

    best_results = [h[-1] for h in all_histories]
    idxmin = best_results.index(np.min(best_results))
    best_individual = all_individuals[idxmin]

    return  idxmin, best_individual, all_histories