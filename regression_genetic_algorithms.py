# =========================================================
# Chapter: Evolutionary Computation - Regression via Genetic Algorithms (GA)
# Features: Linear Regression & Non-Linear Sinusoidal Regression
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA

class RegressionGAOptimizer:
    """ A toolkit for solving linear and non-linear regression parameters using Genetic Algorithms. """

    @staticmethod
    def linear_regression_ga():
        """
        Problem 1: Linear Regression using GA
        Find parameters c1 and c2 in [-5, 5] for y = c1*x + c2
        Given data: x = [0, 1, 2, 3, 4], y = [1, 3, 5, 7, 9]
        """
        print("\n--- Problem 1: Linear Regression via GA ---")
        
        xt = np.array([0, 1, 2, 3, 4], dtype=float)
        yt = np.array([1, 3, 5, 7, 9], dtype=float)

        # Objective function: Sum of Squared Errors (MSE / SSE)
        def fitness_func(params):
            c1, c2 = params
            y_pred = c1 * xt + c2
            return np.sum((yt - y_pred)**2)

        # GA Configuration (Population=50, Generations=1000, Bounds=[-5, 5])
        lb = [-5.0, -5.0]
        ub = [5.0, 5.0]
        
        ga = GA(
            func=fitness_func,
            n_dim=2,
            size_pop=50,
            max_iter=1000,
            lb=lb,
            ub=ub,
            precision=1e-6
        )
        
        best_params, best_fitness = ga.run()
        c1_opt, c2_opt = best_params

        print(f"Optimal Parameters -> c1: {c1_opt:.4f}, c2: {c2_opt:.4f}")
        print(f"Final Sum of Squared Errors (Fitness): {best_fitness[0]:.6f}")

        # Plotting Results
        plt.figure(figsize=(8, 5))
        plt.scatter(xt, yt, color='blue', label='Actual Data', zorder=5)
        plt.plot(xt, c1_opt * xt + c2_opt, color='red', label=f'GA Fit: y = {c1_opt:.2f}x + {c2_opt:.2f}')
        plt.title('Linear Regression via Genetic Algorithm')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def nonlinear_regression_ga():
        """
        Problem 2: Non-Linear Regression using GA
        Model: y = a * sin(2*pi*x + b) + c
        Find parameters a, b, c given 11 data points.
        """
        print("\n--- Problem 2: Non-Linear Regression via GA ---")
        
        xt = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
        yt = np.array([0.641, 0.698, 0.678, 0.591, 0.469, 0.359, 0.302, 0.322, 0.409, 0.531, 0.641], dtype=float)

        # Objective function: Sum of Squared Errors
        def fitness_func(params):
            a, b, c = params
            y_pred = a * np.sin(2 * np.pi * xt + b) + c
            return np.sum((yt - y_pred)**2)

        # Bounds for 3 parameters [a, b, c]
        lb = [-5.0, -5.0, -5.0]
        ub = [5.0, 5.0, 5.0]
        
        ga = GA(
            func=fitness_func,
            n_dim=3,
            size_pop=50,
            max_iter=1000,
            lb=lb,
            ub=ub,
            precision=1e-6
        )
        
        best_params, best_fitness = ga.run()
        a_opt, b_opt, c_opt = best_params

        print(f"Optimal Parameters -> a: {a_opt:.4f}, b: {b_opt:.4f}, c: {c_opt:.4f}")
        print(f"Final Sum of Squared Errors (Fitness): {best_fitness[0]:.6f}")

        # Plotting Results
        x_smooth = np.linspace(0, 1, 200)
        y_smooth = a_opt * np.sin(2 * np.pi * x_smooth + b_opt) + c_opt

        plt.figure(figsize=(9, 5))
        plt.scatter(xt, yt, color='black', label='Observed Data', zorder=5)
        plt.plot(x_smooth, y_smooth, color='purple', lw=2, label='GA Non-Linear Fit')
        plt.title('Non-Linear Sinusoidal Regression via Genetic Algorithm')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    reg_ga = RegressionGAOptimizer()
    
    reg_ga.linear_regression_ga()
    reg_ga.nonlinear_regression_ga()
