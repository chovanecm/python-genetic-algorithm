import pytest


def test_genalg():
    from mchgenalg import GeneticAlgorithm
    import numpy as np

    # First, define function that will be used to evaluate the fitness
    def fitness_function(genome):
        # let's count the number of one-values in the genome
        # this will be our fitness
        sum = np.sum(genome)
        return sum

    # Configure the algorithm:
    population_size = 10
    genome_length = 20
    ga = GeneticAlgorithm(fitness_function)
    ga.generate_binary_population(size=population_size, genome_length=genome_length)
    # How many pairs of individuals should be picked to mate
    ga.number_of_pairs = 5
    # Selective pressure from interval [1.0, 2.0]
    # the lower value, the less will the fitness play role
    ga.selective_pressure = 1.5
    ga.mutation_rate = 0.1

    # Run 1000 iterations of the algorithm
    ga.run(1000)

    assert (ga.get_best_genome()[1] == genome_length)
