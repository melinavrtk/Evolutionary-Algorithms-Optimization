# =========================================================
# Chapter: Unconstrained Optimization via Nelder-Mead (fminsearch equivalent)
# Features: Rosenbrock Function, 1D/2D Minimization & Maximization
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

class NelderMeadOptimizer:
    """ Solves unconstrained optimization problems using the Nelder-Mead simplex algorithm. """

    @staticmethod
    def rosenbrock_example():
        """
        Example: Minimize the Rosenbrock function
        f(x1, x2) = 100 * (x2 - x1^2)^2 + (1 - x1)^2
        """
        print("\n--- Example: Rosenbrock Function Minimization ---")
        
        def objective_func(x):
            return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2

        x0 = np.array([-1.2, 1.0])
        # Nelder-Mead method corresponds to MATLAB's fminsearch
        result = minimize(objective_func, x0, method='Nelder-Mead')
        
        print(f"Optimal x: [{result.x[0]:.4f}, {result.x[1]:.4f}]")
        print(f"Minimum Value f(x): {result.fun:.4f}")

        # 3D Surface Plot
        xx1 = np.linspace(-2, 2, 50)
        xx2 = np.linspace(-2, 2, 50)
        X1, X2 = np.meshgrid(xx1, xx2)
        Y = 100.0 * (X2 - X1**2)**2 + (1.0 - X1)**2

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
        ax.scatter(result.x[0], result.x[1], result.fun, color='red', s=100, label='Minimum')
        
        ax.set_title("Rosenbrock Function (Nelder-Mead)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1, x2)")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_1_minimization_2d():
        """
        Exercise 1: Minimize f(x1, x2) = x1^2 + x2^2 - 6*x1 + 6*x2
        """
        print("\n--- Exercise 1: 2D Minimization ---")
        
        def objective_func(x):
            return x[0]**2 + x[1]**2 - 6*x[0] + 6*x[1]

        x0 = np.array([1.0, 1.0])
        result = minimize(objective_func, x0, method='Nelder-Mead')
        
        print(f"Optimal x: [{result.x[0]:.4f}, {result.x[1]:.4f}]")
        print(f"Minimum Value f(x): {result.fun:.4f}")

        # 3D Surface Plot
        xx1 = np.linspace(-10, 10, 50)
        xx2 = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(xx1, xx2)
        Z = X**2 + Y**2 - 6*X + 6*Y

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
        ax.scatter(result.x[0], result.x[1], result.fun, color='red', s=100, label='Minimum')
        
        ax.set_title("Exercise 1: 2D Minimization")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1, x2)")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_2_maximization():
        """
        Exercise 2: Maximize f(x) = (4*sqrt(x) - x)^4 for x in (0, 16]
        (Achieved by minimizing the negative of the function)
        """
        print("\n--- Exercise 2: 1D Maximization ---")
        
        def objective_func_neg(x):
            val = np.clip(x[0], 1e-8, 16.0)
            return -((4.0 * np.sqrt(val) - val)**4)

        def actual_func(x):
            return (4.0 * np.sqrt(x) - x)**4

        x0 = [1.0]
        result = minimize(objective_func_neg, x0, method='Nelder-Mead')
        
        max_x = result.x[0]
        max_val = -result.fun
        
        print(f"Optimal x for Maximum: {max_x:.4f}")
        print(f"Maximum Value f(x): {max_val:.4f}")

        # Plotting
        xx = np.linspace(0.01, 16, 300)
        yy = actual_func(xx)
        
        plt.figure(figsize=(8, 5))
        plt.plot(xx, yy, color='teal', label='f(x) = (4√x - x)^4')
        plt.scatter(max_x, max_val, color='red', s=100, label=f'Max: ({max_x:.2f}, {max_val:.2f})', zorder=5)
        plt.title("Exercise 2: 1D Maximization")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_3_minimization():
        """
        Exercise 3: Minimize f(x) = (4*sqrt(x) - x)^4 for x in (0, 16]
        """
        print("\n--- Exercise 3: 1D Minimization ---")
        
        def objective_func(x):
            val = np.clip(x[0], 1e-8, 16.0)
            return (4.0 * np.sqrt(val) - val)**4

        x0 = [1.0]
        result = minimize(objective_func, x0, method='Nelder-Mead')
        
        print(f"Optimal x for Minimum: {result.x[0]:.4f}")
        print(f"Minimum Value f(x): {result.fun:.4f}")

        # Plotting
        xx = np.linspace(0.01, 16, 300)
        yy = objective_func([xx])
        
        plt.figure(figsize=(8, 5))
        plt.plot(xx, yy, color='purple', label='f(x) = (4√x - x)^4')
        plt.scatter(result.x[0], result.fun, color='red', s=100, label=f'Min: ({result.x[0]:.2f}, {result.fun:.2f})', zorder=5)
        plt.title("Exercise 3: 1D Minimization")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_4_quadratic():
        """
        Exercise 4: Minimize f(x) = 1 - x/10 + x^2/200
        """
        print("\n--- Exercise 4: Quadratic Minimization ---")
        
        def objective_func(x):
            return 1.0 - x[0]/10.0 + (x[0]**2)/200.0

        x0 = [1.0]
        result = minimize(objective_func, x0, method='Nelder-Mead')
        
        print(f"Optimal x: {result.x[0]:.4f}")
        print(f"Minimum Value f(x): {result.fun:.4f}")

        # Plotting
        xx = np.linspace(-10, 20, 300)
        yy = 1.0 - xx/10.0 + (xx**2)/200.0
        
        plt.figure(figsize=(8, 5))
        plt.plot(xx, yy, color='blue', label='f(x) = 1 - x/10 + x^2/200')
        plt.scatter(result.x[0], result.fun, color='red', s=100, label=f'Min: ({result.x[0]:.2f}, {result.fun:.2f})', zorder=5)
        plt.title("Exercise 4: Quadratic Minimization")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    nm = NelderMeadOptimizer()
    
    nm.rosenbrock_example()
    nm.exercise_1_minimization_2d()
    nm.exercise_2_maximization()
    nm.exercise_3_minimization()
    nm.exercise_4_quadratic()
