import cvxpy as cp
import numpy as np

# Quick discussion of the problem:
# Before reading this code, read over the problem starting on page 17 of
# the textbook.
# The modeling is kind of tricky, so I recommend making sure you fully understand
# the objective function before reading this#

# Variables:
#
# For 1 <= i <= 12, we have 4 variables:
#
# x_i represents the production in month i
# s_i represents the surplus after month i
# y_i represents an increase in production from month i-1 to month i
# z_i represents a decrease in production from month i-1 to month i
# 
# 
# #

x = cp.Variable((12), nonneg=True)
s = cp.Variable((12), nonneg=True)
y = cp.Variable((12), nonneg=True)
z = cp.Variable((12), nonneg=True)

# This creates a row vector representing the demand in each month.
# You can change this to see how best to react to different
# demand schedules
d = np.array([350, 325, 450, 640, 640, 550, 700, 670, 350, 425, 400, 650])

cost1 = np.array([50,50,50,50,50,50,50,50,50,50,50,50])
cost2 = np.array([20,20,20,20,20,20,20,20,20,20,20,20])
# cost2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Our objective function involves taking sums of each of these variables
# from i=1 to i=12. 
# Instead of using sums (which are not linear), we will
# expand each of these sums to be linear equations 
# (ie: expanding sum(i_1 => i_12) into (i_1 + i_2 +...+ i_12).
# Hope that makes sense.
#
# .T notation transposes the preceding argument. So for example,
# because cost1 and y are both row vectors, we need to transpose 
# cost1 into a column vector in order to multiply them.
# #
objective = cp.Minimize((cost1.T @ y)+(cost1.T @ z)+(cost2.T @ s))

# This is complicated. We're using linear algebra to expand a 
# recurrence relationship into a linear function.
# #
A = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]])

# These constraints are the meat of it.
# Remember that @ is matrix multiplication. 
# Try setting up these equivalences and examining them entry-by-entry.
# We're basically encoding the constraints using this matrix. It's
# weird lol. 
# #
constraints = [A @ s == d.T - x,
               A @ x == y - z]

problem = cp.Problem(objective, constraints)


print("Total production cost for the year: ", problem.solve())
print("\n")
print("Optimized production for each month: ", x.value)