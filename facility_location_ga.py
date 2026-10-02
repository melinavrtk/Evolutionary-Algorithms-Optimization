# =========================================================
# Chapter: Evolutionary Computation - Facility Location Problem via GA
# Features: Integer-Constrained Genetic Algorithm for Weighted Distance Minimization
# =========================================================
import numpy as np

class FacilityLocationGA:
    """ Solves the weighted distance facility location problem using Genetic Algorithms. """

    @staticmethod
    def solve_location_problem():
        print("\n--- Facility Location Optimization via GA ---")

        # 1. Define Matrix P: [Column (X), Row (Y), Weight]
        P = np.array([
            [1, 1, 3], [2, 1, 4], [3, 1, 1], [4, 1, 2], [5, 1, 1], [6, 1, 3], [7, 1, 8],
            [1, 2, 2], [2, 2, 1], [3, 2, 3], [4, 2, 1], [5, 2, 3], [6, 2, 9], [7, 2, 7],
            [1, 3, 5], [2, 3, 1], [3, 3, 2], [4, 3, 4], [5, 3, 4], [6, 3, 9], [7, 3, 8],
            [1, 4, 4], [2, 4, 2], [3, 4, 1], [4, 4, 1], [5, 4, 2], [6, 4, 5], [7, 4, 9],
            [1, 5, 8], [2, 5, 9], [3, 5, 6], [4, 5, 3], [5, 5, 2], [6, 5, 8], [7, 5, 7],
            [1, 6, 9], [2, 6, 8], [3, 6, 5], [4, 6, 2], [5, 6, 1], [6, 6, 7], [7, 6, 9],
            [1, 7, 8], [2, 7, 9], [3, 7, 6], [4, 7, 1], [5, 7, 1], [6, 7, 8], [7, 7, 9]
        ], dtype=float)

        # 2. Objective Function: Weighted Sum of Euclidean Distances
        def objective_function(coords):
            x_eru, y_eru = coords[0], coords[1]
            dist_sum = 0.0
            for i in range(len(P)):
                dist = P[i, 2] * np.sqrt((P[i, 0] - x_eru)**2 + (P[i, 1] - y_eru)**2)
                dist_sum += dist
            return dist_sum

        # Wrapper to enforce integer coordinates for grid points (IntCon equivalent)
        def fitness_func(coords):
            x_int = np.round(coords)
            return objective_function(x_int)

        # 3. Import and Run Genetic Algorithm (using scikit-opt)
        try:
            from sko.GA import GA
            
            # Bounds: X and Y coordinates restricted between 1 and 7
            ga = GA(
                func=fitness_func,
                n_dim=2,
                size_pop=500,
                max_iter=200,
                lb=[1.0, 1.0],
                ub=[7.0, 7.0],
                precision=1e-7
            )
            
            best_vars, best_score = ga.run()
            optimal_coords = np.round(best_vars).astype(int)
            final_fval = objective_function(optimal_coords)

            print(f"Optimal Facility Coordinates [X, Y]: {optimal_coords}")
            print(f"Minimum Weighted Distance (fval): {final_fval:.4f}")

        except ImportError:
            print("[Note]: 'scikit-opt' library not found. Running exhaustive grid search instead...")
            # Fallback to precise grid search since the search space is small (7x7 = 49 points)
            best_val = float('inf')
            best_pos = (0, 0)
            for x in range(1, 8):
                for y in range(1, 8):
                    val = objective_function([x, y])
                    if val < best_val:
                        best_val = val
                        best_pos = (x, y)
            print(f"Optimal Facility Coordinates [X, Y]: {best_pos}")
            print(f"Minimum Weighted Distance (fval): {best_val:.4f}")

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    FacilityLocationGA.solve_location_problem()
