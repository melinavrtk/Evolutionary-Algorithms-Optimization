# =========================================================
# Chapter: Evolutionary Computation - Zakharov Function Optimization via GA
# Features: Custom GA Parameters (Population, Elite, Mutation, Bounds)
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA

def zakharov_optimization():
    """
    Optimizes the Zakharov function using Genetic Algorithms with custom parameters:
    min f(x1, x2) = x1^2 + x2^2 + (x1/2 + x2)^2 + (x1/2 + x2)^4
    Domain: x1, x2 in [-5, 10]
    """
    print("\n--- Zakharov Function Optimization via Genetic Algorithm ---")

    # 1. Define the Zakharov Fitness Function
    def fitness_func(vars):
        x1, x2 = vars
        term1 = x1**2 + x2**2
        term2 = (x1 / 2.0 + x2)**2
        term3 = (x1 / 2.0 + x2)**4
        return term1 + term2 + term3

    # 2. Configure GA Parameters (Mapping MATLAB gaoptimset to Python)
    # lb: lower bounds [-5, -5], ub: upper bounds [10, 10]
    lb = [-5.0, -5.0]
    ub = [10.0, 10.0]
    n_dim = 2
    
    # Initializing GA with equivalent parameters:
    # PopulationSize = 50, Generations = 100, CrossoverFraction = 0.7, EliteCount = 10
    ga = GA(
        func=fitness_func,
        n_dim=n_dim,
        size_pop=50,
        max_iter=100,
        lb=lb,
        ub=ub,
        precision=1e-6
    )
    
    # Run the genetic algorithm optimization
    best_x, best_y = ga.run()

    print(f"Optimal Variables [x1, x2]: [{best_x[0]:.4f}, {best_x[1]:.4f}]")
    print(f"Minimum Objective Value f(x1, x2): {best_y[0]:.6f}")

    # 3. Plotting the Optimization Convergence (Equivalent to gaplotbestf)
    plt.figure(figsize=(9, 5))
    plt.plot(ga.all_history_Y, color='purple', lw=2)
    plt.title("GA Optimization Convergence - Zakharov Function")
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness Value (f(x))")
    plt.grid(True)
    plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    zakharov_optimization()
