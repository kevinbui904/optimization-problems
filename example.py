import cvxpy as cp
import numpy as np

x1 = cp.Variable(nonneg = True)
x2 = cp.Variable(nonneg = True)

# Maximize:
objective = cp.Maximize(x1 + x2)

# Subject to:
constraints = [x1 >= 0, x2 >= 0, x2 - x1 <= 1, x1 + 6*x2 <= 15, 4*x1 - x2 <= 10]

problem = cp.Problem(objective, constraints)

problem.solve()

print(problem.value)
print(x1.value, x2.value)