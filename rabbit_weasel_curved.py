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
c = cp.Variable()
delta = cp.Variable()

objective = cp.Maximize(delta)
constraints = []

# y(pi) ≥ ax(pi)2 + bx(pi) + c + δ
constraints = [rabbits["y"] >= a*np.square(rabbits["x"]) + b*rabbits["x"] + c + delta, weasels["y"] <= a*np.square(weasels["x"])+ b*weasels["x"] + c - delta]
problem = cp.Problem(objective, constraints)
solution = problem.solve()

#for drawing purposes only
x = np.linspace(0,10)
#for drawing the line y = ax+b
y = a.value * np.square(x) + b.value*x + c.value
plt.plot(x,y)

#for drawing the line y = ax+b + delta
y_upper = a.value*np.square(x) + b.value*x + c.value + solution
plt.plot(x,y_upper, ls=":")

#for drawing y = ax+b - delta
y_lower = a.value*np.square(x) + b.value*x + c.value - solution
plt.plot(x,y_lower, ls=":")

plt.show()
