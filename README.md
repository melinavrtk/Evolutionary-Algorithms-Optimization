# Evolutionary Algorithms & Mathematical Optimization

This repository focuses on advanced mathematical optimization and evolutionary computation techniques. It implements both unconstrained/constrained optimization algorithms and population-based Genetic Algorithms (GAs) to solve complex engineering, regression, and combinatorial problems using Python (`scipy`, `numpy`, `scikit-opt`).

## Repository Contents

### 1. Genetic Algorithms (GAs) & Combinatorial Optimization
* **`zakharov_genetic_algorithm.py`**: Optimizes the complex multi-variable Zakharov test function using a custom-configured Genetic Algorithm (custom population size, elite count, and bounds).
* **`regression_genetic_algorithms.py`**: Solves parameter estimation for both linear models and non-linear sinusoidal functions ($y = a \cdot \sin(2\pi x + b) + c$) using GA-based error minimization.
* **`facility_location_ga.py`**: Solves the weighted distance facility location problem (Weber problem) with integer coordinate constraints on a discrete grid.
* **`knapsack_genetic_algorithm.py`**: Implements the classic 0-1 Knapsack Problem using binary variable encoding and penalty functions for capacity limits.

### 2. Classical Mathematical Optimization (MATLAB-to-Python equivalents)
* **`unconstrained_optimization.py`**: Solves unconstrained minimization and maximization problems (including the Rosenbrock banana function) using BFGS and gradient-based approaches.
* **`constrained_optimization.py`**: Handles bounded domains and equality constraints (equivalent to MATLAB's `fmincon` using SciPy's SLSQP method).
* **`nelder_mead_optimization.py`**: Implements the Nelder-Mead simplex algorithm (`fminsearch` equivalent) for robust derivative-free optimization.

## Skills & Technologies Highlighted
* **Evolutionary Computation**: Genetic Algorithms (Selection, Crossover, Mutation, Elitism).
* **Constrained & Unconstrained Optimization**: BFGS, Nelder-Mead, SLSQP, and Penalty Method handling.
* **Operations Research**: Facility Location, Knapsack Combinatorial Optimization, and Parameter Regression.
* **Scientific Visualization**: 3D surface plotting (`matplotlib` / `mpl_toolkits`) for visualizing cost functions and convergence paths.

## Acknowledgments & Context
The foundational concepts and initial Python scripts for these projects were developed as part of my undergraduate coursework at the **University of West Attica (Biomedical Engineering)**. 

The current repository represents a heavily refactored and optimized evolution of those academic assignments. The original procedural Python code has been reorganized into robust Object-Oriented pipelines, adhering to modern software engineering practices and industry standards for scalability and readability.

---
*Curated, refactored, and optimized by a final-year Biomedical Engineering student (University of West Attica), specializing in AI and Medical Data Science.*
