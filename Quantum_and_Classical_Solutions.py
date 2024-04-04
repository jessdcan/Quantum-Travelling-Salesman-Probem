#!/usr/bin/env python
# coding: utf-8

# In[3]:


# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_algorithms.optimizers import SPSA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Options, Session
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.converters import QuadraticProgramToQubo
from itertools import permutations
import time


# In[4]:


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


# In[66]:


n = 3 #number of nodes in the TSP
num_qubits = n**2
tsp = Tsp.create_random_instance(n, seed=123)
adj_matrix = nx.to_numpy_array(tsp.graph)
print("distance\n", adj_matrix)

colors = ["r" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
draw_graph(tsp.graph, colors, pos)


# In[53]:


qp = tsp.to_quadratic_program()
# print(qp.prettyprint())


# In[54]:


qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
qubitOp, offset = qubo.to_ising()


# In[55]:


QiskitRuntimeService.save_account(channel="ibm_quantum", token="9ef8aefb2a8761e877e2f93f4bc957eb2e692dd0f54890189f494ff1eb147d9f6202033185155b894e9a1d9efb8fa8b6c7130df9a8498123e9d40a14729f9afc",overwrite=True)
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.backend("ibm_kyoto")


# In[56]:


hamiltonian = qubitOp
# QAOA ansatz circuit
ansatz = QAOAAnsatz(hamiltonian, reps=2)


# In[57]:


target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
ansatz_isa = pm.run(ansatz)


# In[46]:


hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)


# In[58]:


def cost_func(params, ansatz, hamiltonian, estimator):
    energy = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    return energy


# In[59]:


session = Session(backend=backend)
# Configure estimator
estimator = Estimator(session=session)
estimator.options.default_shots = 10000
# Configure sampler
sampler = Sampler(session=session)
sampler.options.default_shots = 10000


# In[60]:


x0 = 2 * np.pi * np.random.rand(ansatz_isa.num_parameters)


# In[61]:


res = minimize(cost_func, x0, args=(ansatz_isa, hamiltonian_isa, estimator), method="COBYLA")


# In[ ]:


# Assign solution parameters to ansatz
qc = ansatz.assign_parameters(res.x)
# Add measurements to the circuit
qc.measure_all()
qc_isa = pm.run(qc)


# In[ ]:


result = sampler.run([qc_isa]).result()


# In[ ]:


prob_distribution_routes = result.quasi_dists[0]


# In[ ]:


dictionary_routes = dict(sorted(distribution.binary_probabilities().items(),key=lambda item: item[1], reverse=True))


# The above line is used to sort the binary outcomes of `prob_distribution_routes`, which represents the probabilities of different solutions to the Traveling Salesman Problem (TSP), ranked on their likelihood of being ideal. In the context of the TSP, each binary string represents a potential route for the graph theory Hamiltonian.

# In[ ]:


arrays = []

# Iterate over the keys of the dictionary
for key in my_dict.keys():
    # Convert the key to a numpy array of integers and append it to the list
    arrays.append(np.array([int(bit) for bit in key]))

# Convert the list of arrays to a numpy array
result = np.array(arrays)

print(result)


# In[ ]:


x = result[0]


# In[ ]:


print("feasible:", qubo.is_feasible(x))


# In[ ]:


z = tsp.interpret(x)
print("solution:", z)


# In[124]:


# Function to generate a random instance of TSP
def generate_tsp_instance(n, seed=123):
    tsp = Tsp.create_random_instance(n, seed=seed)
    return tsp

# Function to calculate the total distance of a tour
def calculate_total_distance(tour, adj_matrix):
    total_distance = 0
    n = len(tour)
    for i in range(n - 1):
        total_distance += adj_matrix[tour[i], tour[i + 1]]
    total_distance += adj_matrix[tour[-1], tour[0]]  # Close the loop
    return total_distance

# Function to solve TSP using brute force
def brute_force_tsp(adj_matrix):
    start_time = time.time()
    n = len(adj_matrix)
    best_order = None
    best_distance = np.inf
    for order in permutations(range(n)):
        total_distance = calculate_total_distance(order, adj_matrix)
        if total_distance < best_distance:
            best_distance = total_distance
            best_order = order
    end_time = time.time()
    time_taken = end_time - start_time
    return best_distance, best_order, time_taken

# Generate TSP instance
n = 9  # Number of nodes in the TSP
tsp = generate_tsp_instance(n)
adj_matrix = nx.to_numpy_array(tsp.graph)
print("Distance Matrix:\n", adj_matrix)

# Solve TSP using brute force
best_distance, best_order, time_taken = brute_force_tsp(adj_matrix)
print("Best order from brute force:", best_order, "with total distance:", best_distance)
print("Time taken:", time_taken, "seconds")

# Plot the TSP solution
colors = ["r" for node in tsp.graph.nodes]
pos = nx.spring_layout(tsp.graph)  # Get node positions for plotting
draw_tsp_solution(tsp.graph, best_order, colors, pos)


# In[110]:


# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to compute the tour length given a tour
def tour_length(tour, adj_matrix):
    length = 0
    for i in range(len(tour) - 1):
        length += adj_matrix[tour[i]][tour[i+1]]
    length += adj_matrix[tour[-1]][tour[0]]  # Close the loop
    return length

# Function to find the nearest neighbor of a given node
def find_nearest_neighbor(node, visited, adj_matrix):
    min_distance = np.inf
    nearest_neighbor = None
    for neighbor in range(len(adj_matrix)):
        if neighbor != node and neighbor not in visited:
            distance = adj_matrix[node][neighbor]
            if distance < min_distance:
                min_distance = distance
                nearest_neighbor = neighbor
    return nearest_neighbor

# Function to solve TSP using K-nearest neighbor algorithm
def tsp_knn(adj_matrix):
    n = len(adj_matrix)
    start_node = 0  
    tour = [start_node]
    visited = set([start_node])

    while len(visited) < n:
        current_node = tour[-1]
        nearest_neighbor = find_nearest_neighbor(current_node, visited, adj_matrix)
        tour.append(nearest_neighbor)
        visited.add(nearest_neighbor)

    tour_length_knn = tour_length(tour, adj_matrix)
    return tour, tour_length_knn

n = 11  # Number of nodes in the TSP
tsp = Tsp.create_random_instance(n, seed=123)
adj_matrix = nx.to_numpy_array(tsp.graph)
print("Distance Matrix:\n", adj_matrix)

# Start timing
start_time = time.time()

# Solve TSP using K-nearest neighbor algorithm
tour_knn, tour_length_knn = tsp_knn(adj_matrix)

# End timing
end_time = time.time()

# Print the tour and its length
print("Tour (K-nearest neighbor):", tour_knn)
print("Tour length (K-nearest neighbor):", tour_length_knn)

# Print time taken
print("Time taken:", end_time - start_time, "seconds")

# Plot the graph and the tour
colors = ["r" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
draw_graph(tsp.graph, colors, pos)
nx.draw_networkx_edges(tsp.graph, pos, edgelist=[(tour_knn[i], tour_knn[i+1]) for i in range(len(tour_knn)-1)], edge_color='r')
nx.draw_networkx_edges(tsp.graph, pos, edgelist=[(tour_knn[-1], tour_knn[0])], edge_color='r')  # Close the loop
plt.title("TSP Solution using K-nearest neighbor")
plt.show()


# In[99]:


# Function to generate a random initial tour starting from node 0
def generate_initial_tour(n):
    tour = np.arange(1, n)  # Nodes other than 0
    np.random.shuffle(tour)  # Shuffle the nodes
    tour = np.insert(tour, 0, 0)  # Insert node 0 at the beginning
    return tour


# Function to accept or reject a move based on the Metropolis criterion
def accept_move(delta, temperature):
    return np.random.rand() < np.exp(-delta / temperature)

# Function to perform simulated annealing to solve TSP
def tsp_simulated_annealing(adj_matrix, initial_temperature=100, cooling_rate=0.99, num_iterations=1000):
    n = len(adj_matrix)
    current_tour = generate_initial_tour(n)
    current_length = tour_length(current_tour, adj_matrix)
    best_tour = current_tour.copy()
    best_length = current_length

    temperature = initial_temperature
    for _ in range(num_iterations):
        new_tour = np.random.permutation(n)
        new_length = tour_length(new_tour, adj_matrix)
        delta = new_length - current_length

        if delta < 0 or accept_move(delta, temperature):
            current_tour = new_tour.copy()
            current_length = new_length
            if current_length < best_length:
                best_tour = current_tour.copy()
                best_length = current_length

        temperature *= cooling_rate

    return best_tour, best_length

n = 11 # Number of nodes in the TSP
tsp = Tsp.create_random_instance(n, seed=123)
adj_matrix = nx.to_numpy_array(tsp.graph)
print("Distance Matrix:\n", adj_matrix)

# Start timing
start_time = time.time()

# Solve TSP using simulated annealing algorithm
tour_sa, tour_length_sa = tsp_simulated_annealing(adj_matrix)

# End timing
end_time = time.time()

# Print the tour and its length
print("Tour:", tour_sa)
print("Tour length:", tour_length_sa)

# Print time taken
print("Time taken:", end_time - start_time, "seconds")

# Plot the graph and the tour
colors = ["r" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
draw_graph(tsp.graph, colors, pos)
nx.draw_networkx_edges(tsp.graph, pos, edgelist=[(tour_sa[i], tour_sa[i+1]) for i in range(len(tour_sa)-1)], edge_color='b')
nx.draw_networkx_edges(tsp.graph, pos, edgelist=[(tour_sa[-1], tour_sa[0])], edge_color='b')  # Close the loop
plt.title("TSP Solution using Simulated Annealing")
plt.show()

