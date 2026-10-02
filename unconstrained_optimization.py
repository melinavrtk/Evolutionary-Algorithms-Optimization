# =========================================================
# Chapter: Unconstrained Optimization Algorithms
# Features: Minimization, Maximization, 2D/3D Function Plotting
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

class FunctionOptimizer:
    """ A toolkit for solving unconstrained optimization problems. """
    
    @staticmethod
    def example_minimization():
        """
        Example: Find the minimum of f(x) = 1 - x/10 + x^2/200
        """
        print("\n--- Example: Simple 1D Minimization ---")
        
        # Define the objective function
        def objective_func(x):
            return 1 - x/10 + (x**2)/200

        # Initial guess (x0)
        x0 = 1.0
        
        # Perform minimization using BFGS algorithm (Python equivalent of fminunc)
        result = minimize(objective_func, x0, method='BFGS')
        
        print(f"Optimal x: {result.x[0]:.4f}")
        print(f"Minimum Value f(x): {result.fun:.4f}")
        print(f"Success: {result.success} | Iterations: {result.nit}")

        # Plotting
        x_vals = np.linspace(0, 30, 100)
        y_vals = objective_func(x_vals)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, label='f(x) = 1 - x/10 + x^2/200', color='blue')
        plt.scatter(result.x, result.fun, color='red', s=100, label=f'Min: ({result.x[0]:.2f}, {result.fun:.2f})', zorder=5)
        plt.title('1D Function Minimization')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_1_minimization_2d():
        """
        Exercise 1: Minimize f(x1, x2) = x1^2 + x2^2 - 6*x1 + 6*x2
        """
        print("\n--- Exercise 1: 2D Function Minimization ---")
        
        def objective_func(vars):
            x1, x2 = vars
            return x1**2 + x2**2 - 6*x1 + 6*x2

        # Initial guess [1, 1]
        x0 = np.array([1.0, 1.0])
        
        result = minimize(objective_func, x0, method='BFGS')
        
        print(f"Optimal [x1, x2]: [{result.x[0]:.4f}, {result.x[1]:.4f}]")
        print(f"Minimum Value f(x1, x2): {result.fun:.4f}")

        # 3D Surface Plotting (Equivalent to MATLAB's surf)
        x1_vals = np.linspace(-10, 15, 100)
        x2_vals = np.linspace(-15, 10, 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = X1**2 + X2**2 - 6*X1 + 6*X2

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
        
        # Plot the minimum point
        ax.scatter(result.x[0], result.x[1], result.fun, color='red', s=100, label='Global Minimum')
        
        ax.set_title('Exercise 1: 3D Surface Minimization')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_2_maximization():
        """
        Exercise 2: Maximize f(x) = (4*sqrt(x) - x)^4
        Note: To maximize in scipy, we minimize the negative of the function.
        """
        print("\n--- Exercise 2: 1D Function Maximization ---")
        
        # We define the NEGATIVE of the function for maximization
        def objective_func_neg(x):
            # Add a small epsilon to avoid math domain errors if solver tests negative numbers
            x = np.clip(x, 1e-8, None) 
            return -((4 * np.sqrt(x) - x)**4)

        # The actual function for plotting
        def actual_func(x):
            return (4 * np.sqrt(x) - x)**4

        x0 = 1.0
        # Bounds used to prevent negative square roots during search
        result = minimize(objective_func_neg, x0, bounds=[(0.01, 100)])
        
        max_x = result.x[0]
        max_val = -result.fun # Revert the negative sign
        
        print(f"Optimal x for Maximum: {max_x:.4f}")
        print(f"Maximum Value f(x): {max_val:.4f}")

        # Plotting
        x_vals = np.linspace(0, 10, 500)
        y_vals = actual_func(x_vals)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, label='f(x) = (4√x - x)^4', color='green')
        plt.scatter(max_x, max_val, color='red', s=100, label=f'Max: ({max_x:.2f}, {max_val:.2f})', zorder=5)
        plt.title('Exercise 2: 1D Function Maximization')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_3_minimization():
        """
        Exercise 3: Minimize f(x) = (4*sqrt(x) - x)^4
        """
        print("\n--- Exercise 3: 1D Function Minimization ---")
        
        def objective_func(x):
            x = np.clip(x, 1e-8, None)
            return (4 * np.sqrt(x) - x)**4

        x0 = 1.0
        result = minimize(objective_func, x0, bounds=[(0.01, 100)])
        
        min_x = result.x[0]
        min_val = result.fun
        
        print(f"Optimal x for Minimum: {min_x:.4f}")
        print(f"Minimum Value f(x): {min_val:.4f}")

        # Plotting over a wider range to show the minimum
        x_vals = np.linspace(0, 100, 500)
        y_vals = objective_func(x_vals)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, label='f(x) = (4√x - x)^4', color='purple')
        plt.scatter(min_x, min_val, color='red', s=100, label=f'Min: ({min_x:.2f}, {min_val:.2f})', zorder=5)
        plt.title('Exercise 3: 1D Function Minimization')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    opt = FunctionOptimizer()
    
    # Run the exercises
    opt.example_minimization()
    opt.exercise_1_minimization_2d()
    opt.exercise_2_maximization()
    opt.exercise_3_minimization()
