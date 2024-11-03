import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

class SOS:
    def __init__(self, objective_function, x_min, x_max, y_min, y_max, population_size, iteration):
        self.objective_function = objective_function  
        self.x_min, self.x_max = x_min, x_max  
        self.y_min, self.y_max = y_min, y_max  
        self.population_size = population_size  
        self.iteration = iteration  
        self.population = self.initialize_population() 
        self.best_fitness_per_iteration = []
    
    def initialize_population(self):
        population = np.random.rand(self.population_size, 2)
        population[:, 0] = self.x_min + population[:, 0] * (self.x_max - self.x_min)
        population[:, 1] = self.y_min + population[:, 1] * (self.y_max - self.y_min)
        return population

    def fitness_function(self, individual):
        return self.objective_function(individual[0], individual[1])

    def optimize(self):
        best_fitness_overall = -np.inf  
        best_individual_overall = None
        start_time = time.time()

        for iter in range(self.iteration):
            fitness_values = np.array([self.fitness_function(individual) for individual in self.population])
            Best_variable = self.population[np.argmax(fitness_values)]

            for i in range(self.population_size):
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"Time: {current_time}; Iteration: {iter + 1}; Organism: {i + 1}")
                x_i, y_i = self.population[i]
                current_fitness = self.fitness_function(self.population[i])

                # Mutualism Phase
                j = np.random.randint(self.population_size)
                while j == i:
                    j = np.random.randint(self.population_size)

                mutVec = np.mean([self.population[i], self.population[j]], axis=0)
                BF1 = np.round(1 + np.random.rand())
                BF2 = np.round(1 + np.random.rand())  

                xNew1 = self.population[i] + np.random.rand(2) * (Best_variable - BF1 * mutVec)
                xNew2 = self.population[j] + np.random.rand(2) * (Best_variable - BF2 * mutVec)

                xNew1 = np.clip(xNew1, [self.x_min, self.y_min], [self.x_max, self.y_max])
                xNew2 = np.clip(xNew2, [self.x_min, self.y_min], [self.x_max, self.y_max])

                if self.fitness_function(xNew1) > current_fitness:
                    self.population[i] = xNew1
                    current_fitness = self.fitness_function(xNew1)
                if self.fitness_function(xNew2) > self.fitness_function(self.population[j]):
                    self.population[j] = xNew2

                # Commensalism Phase
                j = np.random.randint(self.population_size)
                while j == i:
                    j = np.random.randint(self.population_size)

                xNew3 = self.population[i] + (np.random.rand(2) * 2 - 1) * (Best_variable - self.population[j])

                xNew3 = np.clip(xNew3, [self.x_min, self.y_min], [self.x_max, self.y_max])

                if self.fitness_function(xNew3) > current_fitness:
                    self.population[i] = xNew3
                    current_fitness = self.fitness_function(xNew3)
                
                # Parasitism Phase
                j = np.random.randint(self.population_size)
                while j == i:
                    j = np.random.randint(self.population_size)

                parVec = self.population[i].copy()
                nVar = len(parVec)  
                seed = np.random.permutation(nVar)
                id = seed[:np.random.randint(1, nVar+1)] 
                parVec[id] = np.random.rand(len(id)) * (self.x_max - self.x_min) + self.x_min  

                parVec = np.clip(parVec, [self.x_min, self.y_min], [self.x_max, self.y_max])

                fvPar = self.fitness_function(parVec)
                fv_j = self.fitness_function(self.population[j])

                if fvPar > fv_j:
                    self.population[j] = parVec

            fitness_values = np.array([self.fitness_function(individual) for individual in self.population])
            Best_variable = self.population[np.argmax(fitness_values)]

            best_fitness_current = np.max(fitness_values)
            if best_fitness_current > best_fitness_overall:
                best_fitness_overall = best_fitness_current
                best_individual_overall = self.population[np.argmax(fitness_values)]

            self.best_fitness_per_iteration.append(best_fitness_overall)

        end_time = time.time()
        return best_individual_overall, best_fitness_overall, start_time, end_time
    
    def plot_fitness(self):
        plt.plot(range(1, self.iteration + 1), self.best_fitness_per_iteration, marker='x', linestyle='-', color='r')
        plt.title('Best Fitness Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

def objective_function(x, y):
    return np.exp(-0.1 * (x**2 + y**2)) + np.exp(np.cos(4 * np.pi * x) + np.cos(2 * np.pi * y))


if __name__ == "__main__":
    
    x_min = -1
    x_max = 1
    y_min = -2
    y_max = 1
    population_size = 100
    iteration = 100
    
    sos = SOS(objective_function, x_min, x_max, y_min, y_max, population_size, iteration)
    best_solution, best_value, start_time, end_time = sos.optimize()
    print(f"Best Variable x = {best_solution[0]}, Best Variable y = {best_solution[1]}，Maximum Value: {best_value}")
    print(f"Iteration times: {iteration}, Popsize: {population_size}")
    print(f"Total time spent for SOS: {end_time - start_time} sec")

    sos.plot_fitness()

