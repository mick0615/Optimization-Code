import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def obj_dis(permutation, distance_matrix):
    total_distance = 0
    for i in range(len(permutation) - 1):
        total_distance += distance_matrix[permutation[i] - 1, permutation[i + 1] - 1]
    total_distance += distance_matrix[permutation[-1] - 1, permutation[0] - 1]
    return total_distance

def switch(permutation, distance_matrix, value1, value2):
    index1 = np.where(permutation == value1)[0][0]
    index2 = np.where(permutation == value2)[0][0]
    permutation[index1], permutation[index2] = permutation[index2], permutation[index1]
    switch_distance = obj_dis(permutation, distance_matrix)
    return permutation, switch_distance

class Tabu_Search:
    def __init__(self, distance_matrix, iteration, tenure, aspiration_criterion, swap_move, convergence_threshold, convergence_max_times):
        self.distance_matrix = distance_matrix
        self.iteration = iteration
        self.tenure = tenure
        self.aspiration_criterion = aspiration_criterion
        self.swap_move = swap_move
        self.convergence_threshold = convergence_threshold
        self.convergence_max_times = convergence_max_times
        self.node = len(distance_matrix)
        self.permutation = np.random.permutation(range(1, self.node+1))
        self.distance = obj_dis(self.permutation, self.distance_matrix)
        self.tabu_list = np.zeros((len(distance_matrix), len(distance_matrix)))
        self.pre_tabu = [[0,0] for i in range(self.tenure+1)]
        self.best_record_permutation = self.permutation.copy()
        self.best_record_distance = self.distance

    def run(self):
        convergence_times = 0
        fitness_history = []
        for iteration in range(self.iteration):
            Swap, Swap_Obj = [], []
            while len(Swap) < self.swap_move:
                pair = np.random.choice(self.permutation, size=2, replace=False)
                pair.sort()
                
                if list(pair) not in Swap:
                    Swap.append(list(pair))
                    _, distance = switch(self.permutation.copy(), self.distance_matrix, pair[0], pair[1])
                    Swap_Obj.append(distance)
            
            Penalty_Swap_Obj = Swap_Obj.copy()
            for i, move in enumerate(Swap):
                Penalty_Swap_Obj[i] += self.tabu_list[move[1]-1, move[0]-1]
                
            sort_pair = sorted(zip(Penalty_Swap_Obj, Swap), key=lambda x: x[0])    
            Penalty_Swap_Obj = [val[0] for val in sort_pair]
            Swap = [val[1] for val in sort_pair]

            pick_swap = None
            for i, move, Penalty_move_obj in zip(range(self.swap_move), Swap, Penalty_Swap_Obj):
                if self.tabu_list[move[0]-1, move[1]-1] == 0:
                    pick_swap = Swap[i]
                    break
                elif (Penalty_move_obj - self.distance) > self.aspiration_criterion:
                    pick_swap = Swap[i]
                    break

            if pick_swap:
                self.permutation, self.distance = switch(self.permutation.copy(), self.distance_matrix, pick_swap[0], pick_swap[1])
            
            # short-term-memory
            if len(self.pre_tabu) >= self.tenure + 1:
                self.pre_tabu = self.pre_tabu[1:]
            self.pre_tabu.append([pick_swap[0]-1, pick_swap[1]-1])
            for i in range(len(self.pre_tabu)):
                self.tabu_list[self.pre_tabu[i][0], self.pre_tabu[i][1]] = len(self.pre_tabu) - i
            # long-term-memory
            self.tabu_list[pick_swap[1]-1, pick_swap[0]-1] += 1

            if self.distance < self.best_record_distance:
                self.best_record_distance = self.distance
                self.best_record_permutation = self.permutation.copy()
                convergence_times = 0
            else:
                convergence_times += 1
            
            fitness_history.append(self.best_record_distance)

            if convergence_times >= self.convergence_max_times:
                break

        return self.best_record_distance, self.best_record_permutation, fitness_history

if __name__ == "__main__":
    # Parameters
    excel_path = os.path.join(os.getcwd(), 'Tabu', 'SA TS Problems.xlsx')
    dis = np.array(pd.read_excel(excel_path, sheet_name='Q1', header=None))
    dis = dis[1:, 1:]
    iteration = 200
    tenure = 3
    aspiration_criterion = 5
    swap_move = 15
    convergence_threshold = 0
    convergence_max_times = 20

    start_time = time.time()
    # Run Tabu_Search
    Tabu = Tabu_Search(dis, iteration, tenure, aspiration_criterion, swap_move, convergence_threshold, convergence_max_times)
    best_record_distance, best_record_permutation, fitness_history = Tabu.run()
    end_time = time.time()

    # Plot figure
    print(f'Best Permutation : {best_record_permutation}')
    print(f'Minimum Distance : {best_record_distance}')
    print(f'Iteration times : {len(fitness_history)}')
    print(f"Total time spent for Tabu_Search: {end_time - start_time} sec")

    # Plotting the fitness over iterations
    plt.plot(fitness_history, 'xr-', label='Best record so far')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Evolution History')
    plt.xticks(np.arange(0, len(fitness_history), 1))
    plt.legend(loc='center right')
    plt.gca().yaxis.grid(True, alpha=0.4)
    plt.show()


    
         
    


