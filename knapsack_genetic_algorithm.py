# =========================================================
# Chapter: Evolutionary Computation - 0-1 Knapsack Problem via GA
# Features: Binary Integer Constraints, Penalty Function for Capacity Limit
# =========================================================
import numpy as np
from sko.GA import GA

class KnapsackGAOptimizer:
    """ Solves the 0-1 Knapsack Problem using a Genetic Algorithm with penalty handling. """

    @staticmethod
    def solve_knapsack():
        print("\n--- 0-1 Knapsack Problem via Genetic Algorithm ---")

        # 1. Problem Parameters
        No_Variables = 8
        capacity = 20.0
        r = 9.0 / 3.0  # Penalty coefficient
        
        # Item characteristics
        # Weights: [5, 4, 7, 6, 3, 7, 9, 3]
        weights = np.array([5, 4, 7, 6, 3, 7, 9, 3], dtype=float)
        # Values: [7, 11, 14, 6, 9, 10, 8, 4]
        values = np.array([7, 11, 14, 6, 9, 10, 8, 4], dtype=float)

        # 2. Objective / Fitness Function with Penalty
        def sakos_fitness(x):
            # Enforce binary integer constraints (0 or 1) by rounding during evaluation
            x_bin = np.round(x)
            
            # Weight constraint check
            total_weight = np.sum(x_bin * weights)
            g = total_weight - capacity
            pen = max(0.0, g)
            
            # Value calculation (minimizing 1/value maximizes total value)
            total_val = np.sum(x_bin * values)
            if total_val <= 0:
                f = 1e6 # High penalty for zero/negative value
            else:
                f = 1.0 / total_val
                
            # Total fitness to minimize
            y = f + r * pen
            return y

        # 3. Configure GA Parameters
        lb = [0.0] * No_Variables
        ub = [1.0] * No_Variables
        
        # PopulationSize = 200, Generations = 500, CrossoverFraction = 0.75
        ga = GA(
            func=sakos_fitness,
            n_dim=No_Variables,
            size_pop=200,
            max_iter=500,
            lb=lb,
            ub=ub,
            precision=1e-7
        )
        
        best_x, best_y = ga.run()
        
        # Extract optimal binary selection vector
        optimal_selection = np.round(best_x).astype(int)
        final_weight = np.sum(optimal_selection * weights)
        final_value = np.sum(optimal_selection * values)

        print(f"Optimal Item Selection (Binary Vector): {optimal_selection}")
        print(f"Total Weight: {final_weight} (Capacity Limit: {capacity})")
        print(f"Total Value Collected: {final_value}")

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    KnapsackGAOptimizer.solve_knapsack()
