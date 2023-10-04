import cvxpy as cp
import numpy as np
import random 
import matplotlib.pyplot as plt

rabbits = {"x": [], "y":[]}
weasels = {"x": [], "y":[]}
#randomly generate set for rabbits and weasels
n = 20 # num rabbits
m = 20 # num weasels
for i in range(n):
    rabbits["x"].append(random.uniform(0,10))
    rabbits["y"].append(random.uniform(5,10))
for j in range(m):
    weasels["x"].append(random.uniform(0,10))
    weasels["y"].append(random.uniform(0,5))

#converting to numpy for numerical calculations
rabbits = {"x": np.array(rabbits["x"]), "y":np.array(rabbits["y"])}
weasels = {"x": np.array(weasels["x"]), "y":np.array(weasels["y"])}

plt.plot(rabbits["x"], rabbits["y"], "o")
plt.plot(weasels["x"], weasels["y"], 'o')
plt.xticks(range(1,11))
plt.yticks(range(1,11))

a = cp.Variable()
b = cp.Variable()
delta = cp.Variable()

objective = cp.Maximize(delta)
constraints = []

# y(pi) ≥ ax(pi) + b + δ
constraints = [rabbits["y"] >= a*rabbits["x"] + b + delta, weasels["y"] <= a*weasels["x"]+ b - delta]
problem = cp.Problem(objective, constraints)
solution = problem.solve()

#for drawing the line y = ax+b
y1 = a.value * (1.0) + b.value 
y2 = a.value * (3.0) + b.value
plt.axline((1.0, y1), (3.0, y2))

#for drawing the line y = ax+b + delta
y1_upper = a.value * (1.0) + b.value  + solution
y_2_upper = a.value * (3.0) + b.value + solution
plt.axline((1.0, y1_upper), (3.0, y_2_upper), ls=":", )

#for drawing y = ax+b - delta
y1_lower = a.value * (1.0) + b.value  - solution
y_2_lower = a.value * (3.0) + b.value - solution
plt.axline((1.0, y1_lower), (3.0, y_2_lower), ls=":")

plt.show()
