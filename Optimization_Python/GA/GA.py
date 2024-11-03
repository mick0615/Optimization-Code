import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

class Chromosomes:
    def __init__(self, x_bounds, y_bounds, num_x_genes, num_y_genes):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.num_x_genes = num_x_genes
        self.num_y_genes = num_y_genes
        self.genes = num_x_genes + num_y_genes
        self.x_binary = np.random.randint(2, size=num_x_genes).tolist()
        self.y_binary = np.random.randint(2, size=num_y_genes).tolist()
        self.binary = self.x_binary + self.y_binary
        self.x = self.decode(self.x_binary, x_bounds, num_x_genes)
        self.y = self.decode(self.y_binary, y_bounds, num_y_genes)

    @staticmethod
    def decode(binary, bounds, num_genes):
        lower, upper = bounds
        decimal = sum(val * (2 ** idx) for idx, val in enumerate(reversed(binary)))
        return lower + (decimal * (upper - lower)) / (2 ** num_genes - 1)
    
    def fitness(self):
        return np.exp(-0.1 * (self.x ** 2 + self.y ** 2)) + np.exp(np.cos(4 * np.pi * self.x) + np.cos(2 * np.pi * self.y))
    
class GA:
    def __init__(self, x_bounds, y_bounds, num_x_genes, num_y_genes, xy_upper, fitness_func, CR, MR, iteration, popsize, convergence_threshold, convergence_max_times):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.num_x_genes = num_x_genes
        self.num_y_genes = num_y_genes
        self.genes = num_x_genes + num_y_genes
        self.xy_upper = xy_upper
        self.fitness_func = fitness_func
        self.CR = CR
        self.MR = MR
        self.iteration = iteration
        self.popsize = popsize
        self.convergence_threshold = convergence_threshold
        self.convergence_max_times = convergence_max_times
        self.population = [Chromosomes(x_bounds, y_bounds, num_x_genes, num_y_genes) for _ in range(popsize)]
        self.best_solution = None
        self.best_fitness = -np.inf

    def select_parents(self):
        fitness = np.array([chromo.fitness() for chromo in self.population])
        parents = []
        for _ in range(2):
            Normalized_fitness = (fitness - np.min(fitness))/np.sum(fitness)
            Cum = np.cumsum(Normalized_fitness)
            rand_num = np.random.rand()
            pick = np.argmax(rand_num < Cum)
            parents.append(self.population[pick])
            fitness = np.delete(fitness, pick)
        return parents
    
    def crossover(self, parent1, parent2):
        if np.random.rand() < self.CR:
            cross_point = np.random.randint(1, self.genes)
            child1 = Chromosomes(self.x_bounds, self.y_bounds, parent1.num_x_genes, parent1.num_y_genes)
            child2 = Chromosomes(self.x_bounds, self.y_bounds, parent1.num_x_genes, parent1.num_y_genes)
            child1.binary = parent1.binary[:cross_point] + parent2.binary[cross_point:]
            child2.binary = parent2.binary[:cross_point] + parent1.binary[cross_point:]
            child1.x_binary = child1.binary[:self.num_x_genes]
            child1.y_binary = child1.binary[self.num_x_genes:]
            child2.x_binary = child2.binary[:self.num_x_genes]
            child2.y_binary = child2.binary[self.num_x_genes:]
            child1.x = child1.decode(child1.x_binary, self.x_bounds, self.num_x_genes)
            child1.y = child1.decode(child1.y_binary, self.y_bounds, self.num_y_genes)
            child2.x = child2.decode(child2.x_binary, self.x_bounds, self.num_x_genes)
            child2.y = child2.decode(child2.y_binary, self.y_bounds, self.num_y_genes)
            return child1, child2
        else:
            return parent1, parent2
    
    def mutate(self, chromosome):
        if np.random.rand() < self.MR:
            rand_Muta = np.random.randint(0, self.genes)
            chromosome.binary[rand_Muta] = 1 - chromosome.binary[rand_Muta]
            chromosome.x_binary = chromosome.binary[:self.num_x_genes]
            chromosome.y_binary = chromosome.binary[self.num_x_genes:]
            chromosome.x = chromosome.decode(chromosome.x_binary, self.x_bounds, self.num_x_genes)
            chromosome.y = chromosome.decode(chromosome.y_binary, self.y_bounds, self.num_y_genes)
        return chromosome
    
    def run(self):
        convergence_times = 0
        fitness_history = []
        best_fitness_per_iteration = []
        for iteration in range(self.iteration):
            new_population = []
            z = 0
            while z < self.popsize:
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"Time: {current_time}; Iteration: {iteration + 1}; Generate Child: {z + 1}")
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
                z += 2
            
            self.population = new_population
            best_in_population = max(self.population, key=lambda chromo: chromo.fitness())
            best_in_population_fitness = best_in_population.fitness()

            if best_in_population_fitness > self.best_fitness:
                self.best_fitness = best_in_population_fitness
                self.best_solution = best_in_population
                convergence_times = 0
            else:
                convergence_times += 1

            fitness_history.append(self.best_fitness)
            best_fitness_per_iteration.append(best_in_population_fitness)

            if convergence_times >= self.convergence_max_times:
                break

        return self.best_solution, self.best_fitness, fitness_history, best_fitness_per_iteration

# Fitness function
def fitness_function(chromosome):
    return np.exp(-0.1 * (chromosome.x ** 2 + chromosome.y ** 2)) + np.exp(np.cos(4 * np.pi * chromosome.x) + np.cos(2 * np.pi * chromosome.y))

if __name__ == "__main__":
    # Parameters
    x_bounds = (-1, 1)
    y_bounds = (-2, 1)
    xy_upper = 1
    num_x_genes = 15
    num_y_genes = 16
    CR = 0.9
    MR = 0.2
    iteration = 20
    popsize = 800
    convergence_threshold = 1e-4
    convergence_max_times = 12


    start_time = time.time()
    # Run GA
    ga = GA(x_bounds, y_bounds, num_x_genes, num_y_genes, xy_upper, fitness_function, CR, MR, iteration, popsize, convergence_threshold, convergence_max_times)
    best_solution, best_fitness, fitness_history, best_fitness_per_iteration = ga.run()
    end_time = time.time()

    # Plot figure
    print(f'Best Variable x : {best_solution.x}')
    print(f'Best Variable y : {best_solution.y}')
    print(f'Maximum of Objective Function : {best_fitness}')
    print(f'Iteration times : {len(fitness_history)}')
    print(f"Total time spent for GA: {end_time - start_time} sec")


    # Plotting the fitness over iterations
    plt.plot(fitness_history, 'xr-', label='Best record so far')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Evolution History')
    plt.xticks(np.arange(0, len(fitness_history), 1))
    plt.ylim([8.2, 8.4])
    plt.legend(loc='center right')
    plt.gca().yaxis.grid(True, alpha=0.4)
    plt.yticks(np.arange(8.2, 8.4, 0.01))
    plt.show()