import numpy as np
import time
import matplotlib.pyplot as plt
import datetime

class Particle:
    def __init__(self, x_bounds, y_bounds, xy_upper, velocity_max, object_func):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.xy_upper = xy_upper
        self.velocity_max = np.array(velocity_max)
        self.object_func = object_func
        
        self.position = np.array([np.random.uniform(*x_bounds), np.random.uniform(*y_bounds)])
        self.velocity = np.array([np.random.uniform(-self.velocity_max[0], self.velocity_max[0]), 
                                  np.random.uniform(-self.velocity_max[1], self.velocity_max[1])])
        
        # Constraint: Pre-censoring
        while np.sum(self.position) > self.xy_upper:
            self.position = np.array([np.random.uniform(*x_bounds), np.random.uniform(*y_bounds)])
        
        self.best_position = self.position.copy()
        self.fitness = self.object_func(*self.position)
    
    def update_velocity(self, pBest, gBest, w, c1, c2):
        r1, r2 = np.random.rand(2)
        cognitive_component = c1 * r1 * (pBest - self.position)
        social_component = c2 * r2 * (gBest - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component

        # Damping limit for velocity
        self.velocity = np.clip(self.velocity, -self.velocity_max, self.velocity_max)
    
    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, [self.x_bounds[0], self.y_bounds[0]], [self.x_bounds[1], self.y_bounds[1]])
         
        # Constraint Handling: Adhere strategy
        if np.sum(self.position) > self.xy_upper:
            gap = (np.sum(self.position) - self.xy_upper) / 2
            self.position -= gap
        
        fitness = self.object_func(*self.position)
        if fitness > self.fitness:
            self.fitness = fitness
            self.best_position = self.position.copy()

        
class PSO:
    def __init__(self, x_bounds, y_bounds, xy_upper, velocity_max, c1, c2, w_bounds, object_func, num_particle, iteration, convergence_threshold, convergence_max_times):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.xy_upper = xy_upper
        self.velocity_max = np.array(velocity_max)
        self.c1 = c1
        self.c2 = c2
        self.w_bounds = w_bounds
        self.object_func = object_func
        self.num_particle = num_particle
        self.iteration = iteration
        self.convergence_threshold = convergence_threshold
        self.convergence_max_times = convergence_max_times
        
        self.particle =  [Particle(x_bounds, y_bounds, xy_upper, velocity_max, object_func) for _ in range(num_particle)]
        
        # Initialize pBest, gBest positions and fitnesses
        self.pBest_position = [particle.position.copy() for particle in self.particle]
        self.pBest_fitness = [particle.fitness for particle in self.particle]

        self.gBest_position = max(self.particle, key=lambda p: p.fitness).best_position
        self.gBest_fitness = max(p.fitness for p in self.particle)
        
        self.best_record = [self.gBest_fitness]
        self.convergence_times = 0

    def optimize(self):
        start_time = time.time()

        for iteration in range(self.iteration):
            w = self.w_bounds[0] - (iteration / self.iteration) * (self.w_bounds[0] - self.w_bounds[1])

            for particle in self.particle:
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"Time: {current_time}; Iteration: {iteration + 1}; particle: {self.particle.index(particle) + 1}")
                particle.update_velocity(self.pBest_position[self.particle.index(particle)], self.gBest_position, w, self.c1, self.c2)
                particle.update_position()

                # Update pBest for each particle
                if particle.fitness > self.pBest_fitness[self.particle.index(particle)]:
                    self.pBest_fitness[self.particle.index(particle)] = particle.fitness
                    self.pBest_position[self.particle.index(particle)] = particle.position.copy()

                # Update gBest
                if particle.fitness > self.gBest_fitness:
                    self.gBest_fitness = particle.fitness
                    self.gBest_position = particle.position.copy()

            self.best_record.append(self.gBest_fitness)

            if abs(self.best_record[-1] - self.best_record[-2]) < self.convergence_threshold:
                self.convergence_times += 1
                if self.convergence_times >= self.convergence_max_times:
                    break
            else:
                self.convergence_times = 0
        end_time = time.time()

        print(f"Best Variable x: {self.gBest_position[0]}")
        print(f"Best Variable y: {self.gBest_position[1]}")
        print(f"Maximum of Objective Function: {self.gBest_fitness}")
        print(f"Iteration times: {iteration + 1}")
        print(f"Total time spent for PSO: {end_time - start_time} sec")

    def plot_result(self):
        plt.figure()
        plt.plot(self.best_record, 'xr-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('Fitness Evolution')
        plt.ylim([8.2, 8.4])
        plt.legend(['Best record so far'], loc = 'center right')
        plt.gca().yaxis.grid(True, alpha=0.4)
        plt.xticks(np.arange(0, len(self.best_record), 1))
        plt.yticks(np.arange(8.2, 8.4, 0.01))
        plt.show()


def fitness_function(x, y):
    return np.exp(-0.1 * (x**2 + y**2)) + np.exp(np.cos(4 * np.pi * x) + np.cos(2 * np.pi * y))

if __name__ == "__main__":
    x_bounds = (-1, 1)
    y_bounds = (-2, 1)
    xy_upper = 1
    particle_count = 800
    c1 = 3
    c2 = 3
    velocity_max = [0.5 * (x_bounds[1] - x_bounds[0]), 0.5 * (y_bounds[1] - y_bounds[0])]
    w_bounds = [1, 0]
    iteration = 50
    convergence_threshold = 1e-4
    convergence_max_times = 10

    pso = PSO(x_bounds, y_bounds, xy_upper, velocity_max, c1, c2, w_bounds, fitness_function, particle_count, iteration, convergence_threshold, convergence_max_times)
    pso.optimize()
    pso.plot_result()
