# Quantum Traveling Salesman Problem

**Authors:** Aidan Rabinowitz (2341197), Jess Dworcan (1924564)

## Overview

This project implements and compares multiple approaches to solve the Traveling Salesman Problem (TSP):

- **Quantum Computing**: QAOA algorithm using IBM Quantum
- **Classical Algorithms**: Brute Force, Nearest Neighbor, Simulated Annealing
- **Hybrid Approach**: Simulated Annealing with Nearest Neighbor initialization

## Project Details

The project explores both quantum and classical solutions to the TSP, a classic optimization problem. The quantum approach uses the Quantum Approximate Optimization Algorithm (QAOA) implemented on IBM's quantum hardware, while classical methods provide baseline comparisons and hybrid solutions.

### Algorithm Implementations

- **Brute Force**: Exhaustive search for optimal solution (guaranteed optimal but exponential complexity)
- **Nearest Neighbor**: Greedy heuristic approach
- **Simulated Annealing**: Metaheuristic optimization with temperature-based acceptance criteria
- **SA + NN Hybrid**: Combines nearest neighbor initialization with simulated annealing refinement

### Execution Order

The algorithms are executed in the following sequence:
1. **Quantum TSP** - QAOA algorithm (may encounter IBM backend limits)
2. **Brute Force** - Exhaustive search for optimal solution
3. **Nearest Neighbor** - Greedy heuristic approach
4. **Simulated Annealing** - Metaheuristic optimization
5. **Simulated Annealing with Nearest Neighbor** - Hybrid approach using NN as initial solution

## Setup

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- IBM Quantum account (for quantum experiments)

### Installation

```bash
pip install qiskit qiskit-optimization qiskit-algorithms qiskit-ibm-runtime
pip install matplotlib numpy networkx scipy
```

### IBM Quantum Setup

1. Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Generate an API token
3. Uncomment and configure the token in `submission.ipynb`:
   ```python
   token = "your_ibm_quantum_token"
   QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, overwrite=True)
   ```

## Files

- `submission.ipynb` - Main implementation and comparison notebook
- `SA_NN_Class.py` - Simulated Annealing with Nearest Neighbor class
- `Quantum_Project___Travelling_Salesman.pdf` - Project documentation

## Usage

Run `submission.ipynb` to execute all algorithms. The notebook displays solution plots and performance comparisons.

**Note:** IBM Quantum backend may have job call limits. If errors occur, continue running subsequent cells - the notebook will progress smoothly, with no more errors.

### Visualization

The notebook generates plots of TSP solutions throughout the execution process, showing:
- Graph representations of the problem
- Route visualizations for each algorithm
- Performance comparisons and analysis

