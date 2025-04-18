# Import required libraries
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import pandas as pd
import numpy as np

# --- Step 1: Read Data from Excel Files ---
hubs_df = pd.read_excel("Hubs_and_Focus_Cities.xlsx", sheet_name="Hubs")
focus_df = pd.read_excel("Hubs_and_Focus_Cities.xlsx", sheet_name="Focus_Cities")
centers_df = pd.read_excel("Centers.xlsx", sheet_name="Centers")
cost_hub_to_focus_df = pd.read_excel("Costs.xlsx", sheet_name="Hub_to_Focus_Costs")
cost_hub_to_center_df = pd.read_excel("Costs.xlsx", sheet_name="Hub_to_Center_Costs")
cost_focus_to_center_df = pd.read_excel("Costs.xlsx", sheet_name="Focus_to_Center_Costs")

# --- Debug: Print Raw Data ---
print("\n--- Raw Data from Hub_to_Focus_Costs ---")
print(cost_hub_to_focus_df)
print("\n--- Raw Data from Hub_to_Center_Costs (first 10 rows) ---")
print(cost_hub_to_center_df.head(10))
print("\n--- Raw Data from Focus_to_Center_Costs (first 10 rows) ---")
print(cost_focus_to_center_df.head(10))

# --- Step 2: Define Sets (Indices) ---
hub_indices = hubs_df["Hub_ID"].tolist()  # [1, 2]
focus_indices = focus_df["Focus_ID"].tolist()  # [1, 2, 3]
center_indices = centers_df["Center_ID"].tolist()  # [1, 2, ..., 65]

# --- Step 3: Define Parameters ---
hub_capacity = dict(zip(hubs_df["Hub_ID"], hubs_df["Capacity"]))
focus_capacity = dict(zip(focus_df["Focus_ID"], focus_df["Capacity"]))
center_demand = dict(zip(centers_df["Center_ID"], centers_df["Demand"]))

# --- Step 4: Clean and Debug Cost Data ---
def clean_cost_df(df, id1_name, id2_name, cost_name="Cost", df_name=""):
    print(f"\n--- Cleaning {df_name} ---")
    
    # Step 1: Print the raw cost column
    print(f"Raw {cost_name} column:\n{df[cost_name]}")
    
    # Step 2: Convert cost column to string and strip whitespace
    df[cost_name] = df[cost_name].astype(str).str.strip()
    print(f"After converting to string and stripping whitespace:\n{df[cost_name]}")
    
    # Step 3: Replace "N/A" and variations with None
    df[cost_name] = df[cost_name].replace(["N/A", "n/a", "NA", "na", "nan", "NaN", "", " "], None)
    print(f"After replacing 'N/A' and variations with None:\n{df[cost_name]}")
    
    # Step 4: Convert back to numeric, non-numeric values will become NaN
    df[cost_name] = pd.to_numeric(df[cost_name], errors="coerce")
    print(f"After converting to numeric:\n{df[cost_name]}")
    
    # Step 5: Convert NaN to None using replace
    df[cost_name] = df[cost_name].replace(np.nan, None)
    print(f"After converting NaN to None:\n{df[cost_name]}")
    
    # Step 6: Final check for invalid values (NaN or inf)
    invalid_entries = df[df[cost_name].apply(lambda x: isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    if not invalid_entries.empty:
        print(f"Warning: Found NaN/inf values in {cost_name} column after cleaning:")
        print(invalid_entries)
    else:
        print(f"No NaN/inf values found in {df_name} after cleaning.")
    
    # Convert to dictionary
    cost_dict = {(row[id1_name], row[id2_name]): row[cost_name] for _, row in df.iterrows()}
    return cost_dict

# Clean and create cost dictionaries
cost_hub_to_focus = clean_cost_df(cost_hub_to_focus_df, "Hub_ID", "Focus_ID", df_name="Hub_to_Focus_Costs")
cost_hub_to_center = clean_cost_df(cost_hub_to_center_df, "Hub_ID", "Center_ID", df_name="Hub_to_Center_Costs")
cost_focus_to_center = clean_cost_df(cost_focus_to_center_df, "Focus_ID", "Center_ID", df_name="Focus_to_Center_Costs")

# Debug: Print the cost dictionaries to inspect for NaN/inf
print("\ncost_hub_to_focus:", cost_hub_to_focus)
print("cost_hub_to_center (first 10 entries):", dict(list(cost_hub_to_center.items())[:10]))
print("cost_focus_to_center (first 10 entries):", dict(list(cost_focus_to_center.items())[:10]))

# --- Step 5: Create the LP Problem ---
prob = LpProblem("Amazon_Air_Optimization", LpMinimize)

# --- Step 6: Define Decision Variables ---
x = LpVariable.dicts("x", (hub_indices, focus_indices), lowBound=0, cat="Continuous")
y = LpVariable.dicts("y", (hub_indices, center_indices), lowBound=0, cat="Continuous")
z = LpVariable.dicts("z", (focus_indices, center_indices), lowBound=0, cat="Continuous")

# --- Step 7: Set Variables to 0 for N/A Routes ---
for i in hub_indices:
    for j in focus_indices:
        if cost_hub_to_focus.get((i, j)) is None:
            x[i][j].setInitialValue(0)
            x[i][j].fixValue()

for i in hub_indices:
    for k in center_indices:
        if cost_hub_to_center.get((i, k)) is None:
            y[i][k].setInitialValue(0)
            y[i][k].fixValue()

for j in focus_indices:
    for k in center_indices:
        if cost_focus_to_center.get((j, k)) is None:
            z[j][k].setInitialValue(0)
            z[j][k].fixValue()

# --- Step 8: Define the Objective Function ---
def is_valid_cost(cost):
    if cost is None:
        return False
    if isinstance(cost, float):
        return not np.isnan(cost) and not np.isinf(cost)
    return True

prob += (
    lpSum(cost_hub_to_focus[(i, j)] * x[i][j] for i in hub_indices for j in focus_indices if is_valid_cost(cost_hub_to_focus.get((i, j)))) +
    lpSum(cost_hub_to_center[(i, k)] * y[i][k] for i in hub_indices for k in center_indices if is_valid_cost(cost_hub_to_center.get((i, k)))) +
    lpSum(cost_focus_to_center[(j, k)] * z[j][k] for j in focus_indices for k in center_indices if is_valid_cost(cost_focus_to_center.get((j, k))))
), "Total_Cost"

# --- Step 9: Define Constraints ---
for i in hub_indices:
    prob += (
        lpSum(x[i][j] for j in focus_indices) + lpSum(y[i][k] for k in center_indices) <= hub_capacity[i],
        f"Hub_Capacity_{i}"
    )

for j in focus_indices:
    prob += (
        lpSum(x[i][j] for i in hub_indices) <= focus_capacity[j],
        f"Focus_City_Capacity_{j}"
    )

for j in focus_indices:
    prob += (
        lpSum(z[j][k] for k in center_indices) == lpSum(x[i][j] for i in hub_indices),
        f"Flow_Balance_{j}"
    )

for k in center_indices:
    prob += (
        lpSum(y[i][k] for i in hub_indices) + lpSum(z[j][k] for j in focus_indices) == center_demand[k],
        f"Center_Demand_{k}"
    )

# --- Step 10: Solve the Problem ---
prob.solve()

# --- Step 11: Print the Results ---
print("Status:", prob.status)
print("Total Cost =", value(prob.objective))

print("\nShipments from Hubs to Focus Cities:")
for i in hub_indices:
    for j in focus_indices:
        if value(x[i][j]) > 0:
            print(f"x[{i}][{j}] = {value(x[i][j])} tons (Hub {i} to Focus {j})")

print("\nShipments from Hubs to Centers:")
for i in hub_indices:
    for k in center_indices:
        if value(y[i][k]) > 0:
            print(f"y[{i}][{k}] = {value(y[i][k])} tons (Hub {i} to Center {k})")

print("\nShipments from Focus Cities to Centers:")
for j in focus_indices:
    for k in center_indices:
        if value(z[j][k]) > 0:
            print(f"z[{j}][{k}] = {value(z[j][k])} tons (Focus {j} to Center {k})")