# =========================================================
# Chapter: Constrained Optimization Algorithms
# Features: Bounds, Equality Constraints (fmincon equivalent)
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

class ConstrainedOptimizer:
    """ A toolkit for solving optimization problems with bounds and equality constraints. """

    @staticmethod
    def exercise_1_constrained_2d():
        """
        Exercise 1: Minimize f(x1, x2) = x1^2 + x2^2 - 6*x1 + 6*x2 (Unconstrained via minimize)
        """
        print("\n--- Exercise 1: 2D Minimization ---")
        
        def objective_func(vars):
            x1, x2 = vars
            return x1**2 + x2**2 - 6*x1 + 6*x2

        x0 = np.array([1.0, 1.0])
        result = minimize(objective_func, x0, method='BFGS')
        
        print(f"Optimal [x1, x2]: [{result.x[0]:.4f}, {result.x[1]:.4f}]")
        print(f"Minimum Value f(x1, x2): {result.fun:.4f}")

        # 3D Surface Plot
        xx1 = np.linspace(-10, 10, 50)
        xx2 = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(xx1, xx2)
        Z = X**2 + Y**2 - 6*X + 6*Y

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
        ax.scatter(result.x[0], result.x[1], result.fun, color='red', s=100, label='Minimum')
        
        ax.set_title('Exercise 1: Unconstrained 2D Minimization')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_2_bounded_1d():
        """
        Exercise 2: Minimize f(x) = (4*sqrt(x) - x)^4, with bounds x in (0, 16]
        """
        print("\n--- Exercise 2: Bounded 1D Minimization ---")
        
        def objective_func(x):
            x = np.clip(x, 1e-8, 16.0)
            return (4 * np.sqrt(x) - x)**4

        lb, ub = 0.01, 16.0
        x0 = [(lb + ub) / 2.0]
        
        # Bounded minimization (equivalent to bounds in fmincon)
        result = minimize(objective_func, x0, bounds=[(lb, ub)], method='L-BFGS-B')
        
        print(f"Optimal x: {result.x[0]:.4f}")
        print(f"Minimum Value f(x): {result.fun:.4f}")

        # Plotting
        xx = np.linspace(0.01, 16, 500)
        yy = (4 * np.sqrt(xx) - xx)**4
        
        plt.figure(figsize=(8, 5))
        plt.plot(xx, yy, label='f(x) = (4√x - x)^4', color='teal')
        plt.scatter(result.x[0], result.fun, color='red', s=100, label=f'Min: ({result.x[0]:.2f}, {result.fun:.2f})', zorder=5)
        plt.title('Exercise 2: Bounded Minimization (x ∈ (0, 16])')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def exercise_3_equality_constraint():
        """
        Exercise 3: Maximize f(x, y) = 10 - x^2 - (y-2)^2
        Subject to equality constraint: x + 2y = 5
        (Implemented via SciPy's constraints dictionary)
        """
        print("\n--- Exercise 3: Equality-Constrained Maximization ---")
        
        # To maximize f, we minimize -f
        def objective_func_neg(vars):
            x, y = vars
            return -(10 - x**2 - (y - 2)**2)

        def actual_func(x, y):
            return 10 - x**2 - (y - 2)**2

        # Equality constraint: x + 2y - 5 = 0 -> eq constraint type 'eq' means fun == 0
        eq_constraint = {
            'type': 'eq',
            'fun': lambda vars: vars[0] + 2 * vars[1] - 5
        }

        x0 = [1.0, 1.0]
        result = minimize(objective_func_neg, x0, method='SLSQP', constraints=eq_constraint)
        
        opt_x, opt_y = result.x
        max_val = -result.fun # Revert negative sign for maximum value
        
        print(f"Optimal [x, y]: [{opt_x:.4f}, {opt_y:.4f}]")
        print(f"Maximum Value f(x, y) under constraint x + 2y = 5: {max_val:.4f}")

        # 3D Surface Plot with Constraint Line
        xx1 = np.linspace(-5, 5, 50)
        xx2 = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(xx1, xx2)
        Z = actual_func(X, Y)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
        
        # Plot the optimal point
        ax.scatter(opt_x, opt_y, max_val, color='black', s=150, label='Constrained Maximum', zorder=5)
        
        ax.set_title('Exercise 3: Constrained Maximization (x + 2y = 5)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.legend()
        plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    opt = ConstrainedOptimizer()
    
    opt.exercise_1_constrained_2d()
    opt.exercise_2_bounded_1d()
    opt.exercise_3_equality_constraint()
