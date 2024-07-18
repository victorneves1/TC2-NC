import random
from numpy.random import seed
import numpy as np
import pandas as pd

"""
Real-Coded Genetic Algorithm

evolution strategy (mu, lambda) of the ackley objective function


Sources:

@book{michalewicz2013genetic,
  title={Genetic algorithms+ data structures= evolution programs},
  author={Michalewicz, Zbigniew},
  year={2013},
  publisher={Springer Science \& Business Media}
}

@article{michalewicz1996evolutionary,
  title={Evolutionary algorithms for constrained parameter optimization problems},
  author={Michalewicz, Zbigniew and Schoenauer, Marc},
  journal={Evolutionary computation},
  volume={4},
  number={1},
  pages={1--32},
  year={1996},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}

@phdthesis{latre2000contribuiccoes,
  title={Contribui{\c{c}}{\~o}es {\`a} Solu{\c{c}}{\~a}o de Problemas de Escalonamento pela Aplica{\c{c}}{\~a}o Conjunta de Computa{\c{c}}{\~a}o Evolutiva e Otimiza{\c{c}}{\~a}o com Restri{\c{c}}{\~o}es},
  author={Latre, Luis Gimeno},
  year={2000},
  school={Universidade Estadual de Campinas}
}
https://www.dca.fee.unicamp.br/~vonzuben/theses/rico_mest/

"""

seed(42)  # seed the pseudorandom number generator


def generate_individual(upper_bound: list, lower_bound: list) -> np.array:
    return np.array(
        [
            random.uniform(lower_bound[i], upper_bound[i])
            for i in range(len(upper_bound))
        ]
    )


def generate_population(upper_bound: list, lower_bound: list, pop_size) -> list:
    return [generate_individual(upper_bound, lower_bound) for _ in range(pop_size)]


def evaluate_penalty(individual, equation_system):
    """
    Método dw penalidade:
    """
    result = equation_system.evaluate(individual)
    penalty_sum = 0
    for r in result:
        penalty_sum += max(0, r) **2
    return penalty_sum


def arithmetical_crossover(parent1, parent2, alpha):
    """
    Este operador é definido como uma fusão de dois vetores (cromossomos):
    se x1 e x2 são dois indivíduos selecionados para crossover, os dois filhos resultantes serão:

    x1_prime = alpha*x1 + (1 - alpha)*x2
    x2_prime = (1 - alpha)*x1 + alpha*x2

    sendo alpha um número aleatório pertencente ao intervalo [0, 1].
    (Latre, 2000; Michalewicz, 2013)
    """
    parent1, _ = parent1
    parent2, _ = parent2
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2
    return (offspring1, None), (offspring2, None)


def gaussian_mutation(individual, sigma):
    """
    Para problemas com codificação em ponto flutuante.
    Modifique todos os componentes de um cromossomo x = [x1 … xn] na forma:

    x_prime = x + N(0, sigma)

    Sendo N(0, sigma) um vetor de variáveis aleatórias independentes, com distribuição normal,
    média zero e desvio padrão sigma

    (Michalewicz & Schoenauer , 1996)
    """
    ind, fitness = individual
    return ind + np.random.normal(0, sigma, len(ind)), fitness


def roulette_wheel_selection(evaluated_population):
    total_fitness = 0
    for individual, fitness in evaluated_population:
        total_fitness += 1 / fitness
    r = random.uniform(0, total_fitness)
    acc = 0
    for individual, fitness in evaluated_population:
        acc += 1 / fitness
        if acc >= r:
            return individual, fitness
    return evaluated_population[-1]


def real_coded_genetic_algorithm_core(
    equation_system,
    lower_bound,
    upper_bound,
    pop_size,
    max_gen,
    pc,
    pm,
    alpha,
    sigma,
    elitism=True,
    epoch="",
):
    """
    Real-Coded Genetic Algorithm

    :param equation_system: EquationSystem
    :param lower_bound: list
    :param upper_bound: list
    :param pop_size: int
    :param max_gen: int
    :param pc: float - Crossover probability
    :param pm: float - Mutation probability
    :param alpha: float - Crossover parameter
    :param sigma: float - Mutation parameter
    """
    # pop_size must be even
    if pop_size % 2 != 0:
        raise ValueError("pop_size must be even")

    population = generate_population(upper_bound, lower_bound, pop_size)
    population = [
        (individual, evaluate_penalty(individual, equation_system))
        for individual in population
    ]
    population = sorted(population, key=lambda x: x[1])

    history = []
    best_individual = None
    for gen in range(max_gen):
        print(f"Epoch {epoch+1} | Generation {gen+1}    ", end="\r")

        # Keep 1 elite individual
        if elitism:
            elite = population[0]
        else:
            elite = None

        # Selection
        selected_population = []
        for _ in range(pop_size):
            selected_population.append(roulette_wheel_selection(population))
        # Crossover
        offspring_population = []
        for i in range(0, pop_size, 2):
            if random.random() < pc:
                # print(selected_population[i], selected_population[i+1])
                offspring1, offspring2 = arithmetical_crossover(
                    selected_population[i], selected_population[i + 1], alpha
                )
                offspring_population.append(offspring1)
                offspring_population.append(offspring2)
            else:
                offspring_population.append(selected_population[i])
                offspring_population.append(selected_population[i + 1])

        # Mutation
        mutated_population = []
        for individual in offspring_population:
            if random.random() < pm:
                mutated_population.append(gaussian_mutation(individual, sigma))
            else:
                mutated_population.append(individual)

        # Replace population
        population = mutated_population

        # Re Evaluate
        evaluated_population = []

        for individual, _ in population:
            fitness = evaluate_penalty(individual, equation_system)
            evaluated_population.append((individual, fitness))
        if elite:
            evaluated_population[0] = elite

        evaluated_population = sorted(evaluated_population, key=lambda x: x[1])

        best_score = evaluated_population[0][1]
        best_individual = evaluated_population[0][0]
        history.append(best_score)
        population = evaluated_population

    return best_individual, history


def real_coded_genetic_algorithm(
    equation_system,
    lower_bound,
    upper_bound,
    epochs=30,
    pop_size=100,
    max_gen=2000,
    pc=0.8,
    pm=0.1,
    alpha=0.5,
    sigma=0.1,
    elitism=True,
):

    # run the genetic algorithm 30 times and collect the results
    all_histories = []
    all_individuals = []
    for epoch in range(epochs):
        best_individual, history = real_coded_genetic_algorithm_core(
            equation_system,
            lower_bound,
            upper_bound,
            pop_size,
            max_gen,
            pc,
            pm,
            alpha,
            sigma,
            elitism,
            epoch,
        )
        all_histories.append(history)
        all_individuals.append(best_individual)

    best_results = [h[-1] for h in all_histories]
    idxmin = best_results.index(np.min(best_results))
    best_individual = all_individuals[idxmin]

    return  idxmin, best_individual, all_histories
