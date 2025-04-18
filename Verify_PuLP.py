from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Create a simple LP problem
prob = LpProblem("Simple_Test", LpMinimize)

# Define variables
x1 = LpVariable("x1", lowBound=0, cat="Continuous")
x2 = LpVariable("x2", lowBound=0, cat="Continuous")

# Define objective function
prob += 2 * x1 + 3 * x2, "Total_Cost"

# Define constraints
prob += x1 + x2 >= 5, "Constraint1"
prob += x1 <= 4, "Constraint2"
prob += x2 <= 3, "Constraint3"

# Solve the problem
prob.solve()

# Print results
print("Status:", prob.status)
print("Total Cost =", value(prob.objective))
print("x1 =", value(x1))
print("x2 =", value(x2))
