# python-genetic-algorithm
Genetic algorithm "library"

## About the library
The "library" was created for a Czech Technical University's course "Problems and algorithms".

It supports evolving individuals composed of fixed-length binary genomes.

## Installation
Install the library:

	python setup.py install
	
## Usage:
The following example shows using the library to solve a very simple problem
of maximizing the number of one-values in a vector.


```python
from mch_genetic_algorithm import GeneticAlgorithm
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

best_genome, best_fitness = ga.get_best_genome()

# If you want, you can have a look at the population:
population = ga.population
# and the fitness of each element:
fitness_vector = ga.get_fitness_vector()
```