# =========================================================
# Chapter: Evolutionary Computation & Genetic Algorithms (GA)
# Features: Function Minimization, Maximization, GA Operators
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA  # Requires: pip install scikit-opt

class GeneticAlgorithmOptimizer:
    """ A toolkit for solving optimization problems using Genetic Algorithms (GAs). """

    @staticmethod
    def example_minimization_ga():
        """
        Example: Minimize f(x) = 1 - x/10 + x^2/200 using GA
        """
        print("\n--- Example: 1D Function Minimization via GA ---")
        
        def fitness_func(x):
            return 1 - x/10 + (x**2)/200

        # GA parameters: func, n_dim, size_pop, max_iter, lb, ub
        ga = GA(func=fitness_func, n_dim=1, size_pop=50, max_iter=100, lb=[0], ub=[30], precision=1e-7)
        best_x, best_y = ga.run()
        
        print(f"Optimal x: {best_x[0]:.4f}")
        print(f"Minimum Value f(x): {best_y[0]:.4f}")

        # Plot evolution process
        plt.figure(figsize=(8, 4))
        plt.plot(ga.all_history_Y, color='teal', lw=2)
        plt.title("GA Optimization Progress (Example)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.grid(True)
        plt.show()

    @staticmethod
    def exercise_1_minimization_2d_ga():
        """
        Exercise 1: Minimize f(x1, x2) = x1^2 + x2^2 - 6*x1 + 6*x2 using GA
        """
        print("\n--- Exercise 1: 2D Minimization via GA ---")
        
        def fitness_func(vars):
            x1, x2 = vars
            return x1**2 + x2**2 - 6*x1 + 6*x2

        # 2 variables, bounds [-10, 10] for both x1 and x2
        ga = GA(func=fitness_func, n_dim=2, size_pop=80, max_iter=100, lb=[-10, -10], ub=[10, 10], precision=1e-7)
        best_x, best_y = ga.run()
        
        print(f"Optimal [x1, x2]: [{best_x[0]:.4f}, {best_x[1]:.4f}]")
        print(f"Minimum Value f(x1, x2): {best_y[0]:.4f}")

    @staticmethod
    def exercise_2_maximization_ga():
        """
        Exercise 2: Maximize f(x) = (4*sqrt(x) - x)^4, for x in (0, 16]
        Note: To maximize using a minimization GA, we return the negative of the fitness function.
        """
        print("\n--- Exercise 2: 1D Maximization via GA ---")
        
        def fitness_func_neg(x):
            x = np.clip(x, 1e-8, 16.0)
            return -((4 * np.sqrt(x) - x)**4)

        ga = GA(func=fitness_func_neg, n_dim=1, size_pop=60, max_iter=300, lb=[0.01], ub=[16.0], precision=1e-7)
        best_x, best_y = ga.run()
        
        max_val = -best_y[0] # Revert negative sign
        print(f"Optimal x for Maximum: {best_x[0]:.4f}")
        print(f"Maximum Value f(x): {max_val:.4f}")

    @staticmethod
    def exercise_3_minimization_ga():
        """
        Exercise 3: Minimize f(x) = (4*sqrt(x) - x)^4, for x in (0, 16]
        """
        print("\n--- Exercise 3: 1D Minimization via GA ---")
        
        def fitness_func(x):
            x = np.clip(x, 1e-8, 16.0)
            return (4 * np.sqrt(x) - x)**4

        ga = GA(func=fitness_func, n_dim=1, size_pop=60, max_iter=300, lb=[0.01], ub=[16.0], precision=1e-7)
        best_x, best_y = ga.run()
        
        print(f"Optimal x for Minimum: {best_x[0]:.4f}")
        print(f"Minimum Value f(x): {best_y[0]:.4f}")

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    ga_opt = GeneticAlgorithmOptimizer()
    
    ga_opt.example_minimization_ga()
    ga_opt.exercise_1_minimization_2d_ga()
    ga_opt.exercise_2_maximization_ga()
    ga_opt.exercise_3_minimization_ga()
