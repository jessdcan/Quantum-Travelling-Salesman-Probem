import math
import random
import matplotlib.pyplot as plt
import timeit


class SA_NN(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

    # find the TSP using the Nearest Neighbor approach
    def nearest_neighbor(self):
        curr_node = random.choice(self.nodes)  # start from a random node
        solution = [curr_node]

        # each iteration remove the current node from the array of free nodes
        free_nodes = set(self.nodes)
        free_nodes.remove(curr_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.calculate_distance(
                curr_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            curr_node = next_node

        cur_fit = self.total_path_distance(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        end = timeit.default_timer()
        print("Best fitness obtained for nearest neighbor: ", self.best_fitness)

        return solution, cur_fit

    # Calulcates the distance between two nodes
    def calculate_distance(self, node1, node2):
        coord1, coord2 = self.coords[node1], self.coords[node2]
        distance = math.sqrt((coord1[0] - coord2[0])
                             ** 2 + (coord1[1] - coord2[1]) ** 2)
        return distance

    # Calulcates the total distance of the current path
    def total_path_distance(self, solution):
        total_distance = 0
        for i in range(self.N):
            total_distance += self.calculate_distance(
                solution[i % self.N], solution[(i + 1) % self.N])
        return total_distance

    # Plot the greedy solution.
    def plot_greedy_solution(self):
        solution, _ = self.nearest_neighbor()
        plotTSP([solution], self.coords)

    # Probability of accepting if the candidate is worse than the current solution.
    # Dependant on current temperature and difference between current and candidate.

    def p_accept(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    # If candidate is better than current, accept with probability = 1.
    # If candidate is worse, except with probability found in p_accept()

    def accept(self, candidate):
        candidate_fitness = self.total_path_distance(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    # Perform simulated annealing algorithm
    def simulated_annealing(self):
        start = timeit.default_timer()
        # Initialize with the nearest neighbor solution.
        self.cur_solution, self.cur_fitness = self.nearest_neighbor()

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1

           # self.fitness_list.append(self.cur_fitness)
        end = timeit.default_timer()
        print("Simulated Annealing with Nearest Neighbor Inital Approach")
        print("Time taken: ", end - start)
        print("Best fitness obtained: ", self.best_fitness)
        # improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        # print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    # def simulated_annealing(self):
    #     start = time.time()
    #     # Initialize with a random solution.
    #     self.cur_solution = list(range(self.N))
    #     self.cur_fitness = self.total_path_distance(self.cur_solution)

    #     print("Starting annealing.")
    #     while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
    #         candidate = list(self.cur_solution)  # Make a copy of the current solution
    #         l = random.randint(2, self.N - 1)
    #         i = random.randint(0, self.N - l)
    #         candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
    #         self.accept(candidate)
    #         self.T *= self.alpha
    #         self.iteration += 1

    #     end = time.time()
    #     print("Time taken: ", end - start)

    #     print("Best fitness obtained: ", self.best_fitness)

    # Runs a simulated annealing algorithm n time, with random initial solutions.

    def batch_anneal(self, n=10):
        for i in range(1, n + 1):
            print(f"Iteration {i}/{n} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.nearest_neighbor()
            self.simulated_annealing()

    # Plot the route
    def visualize_routes(self):
        plotTSP([self.best_solution], self.coords)

    # Plot the learning
    def plot_learning(self):
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()


def plotTSP(paths, points, num_iters=1):
    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list

    """

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    x = []
    y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads (there should be a reasonable default for     this, WTF?)
    a_scale = float(max(x))/float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []
            yi = []
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                      head_width=a_scale, color='r',
                      length_includes_head=True, ls='dashed',
                      width=0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                          head_width=a_scale, color='r', length_includes_head=True,
                          ls='dashed', width=0.001/float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=a_scale,
              color='g', length_includes_head=True)
    for i in range(0, len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width=a_scale,
                  color='g', length_includes_head=True)

    # Set axis too slitghtly larger than the set of x and y
    # plt.xlim(min(x)*1.1, max(x)*1.1)
    # plt.ylim(min(y)*1.1, max(y)*1.1)
    plt.show()
