import os
import numpy as np
import pandas as pd

# set seed for reproducibility
np.random.seed(42)

# number of employees to generate
N = 5000

# features
emp_id = [f"EMP{str(i).zfill(4)}" for i in range(N)]
names = [f"Employee_{i+1}" for i in range(N)]

age = np.random.randint(20, 60, N)
years_at_company = np.random.randint(0, 15, N)
job_satisfaction_num = np.random.randint(1, 5, N)   # 1 (very low) - 4 (very high)
job_satisfaction_map = {1: "Very Low", 2: "Low", 3: "High", 4: "Very High"}
job_satisfaction = [job_satisfaction_map[v] for v in job_satisfaction_num]

work_life_balance_num = np.random.randint(1, 5, N)  # 1 (worst) - 4 (excellent)
wlb_map = {1: "Worst", 2: "Bad", 3: "Good", 4: "Excellent"}
work_life_balance = [wlb_map[v] for v in work_life_balance_num]

monthly_income = np.random.randint(3000, 20000, N)

# Overtime: generate random number of overtime hours per week (0-12)
overtime_hours = np.random.randint(0, 13, N)
# For display, you may show: "No OT" if 0, else "{n} hours"
overtime_display = [f"{h} hours" if h > 0 else "No OT" for h in overtime_hours]

distance_raw = np.random.randint(1000, 30000, N) # in meters (1km - 30km)
distance_from_home = distance_raw / 1000  # convert to km

promotion_last_5years_num = np.random.choice([0, 1], size=N, p=[0.8, 0.2])
promotion_map = {0: "No", 1: "Yes"}
promotion_last_5years = [promotion_map[v] for v in promotion_last_5years_num]

# probability of attrition (rule-based, using numbers)
base_prob = (
    0.3*(age < 30) +
    0.2*(job_satisfaction_num < 2) +
    0.25*(overtime_hours > 6) +
    0.15*(years_at_company < 2) +
    0.2*(promotion_last_5years_num == 0)
)

# squish probabilities into [0,1]
prob_attrition = 1 / (1 + np.exp(-(base_prob - 1.5)))

# simulate attrition outcome
attrition = np.random.binomial(1, prob_attrition)

# build DataFrame
df = pd.DataFrame({
    'EmpID': emp_id,
    'Name': names,
    'Age': age,
    'YearsAtCompany': years_at_company,
    'JobSatisfaction': job_satisfaction,
    'WorkLifeBalance': work_life_balance,
    'MonthlyIncome': monthly_income,
    'OverTimeHours': overtime_display,
    'DistanceFromHome_km': distance_from_home,
    'PromotionLast5Years': promotion_last_5years,
    'Attrition': np.where(attrition == 1, 'Yes', 'No')
})

# make sure "data" folder exists
os.makedirs("data", exist_ok=True)

# save to CSV
output_path = os.path.join("data", "employees.csv")
df.to_csv(output_path, index=False)

print(f"✅ Synthetic dataset generated and saved to {output_path}")
print(df.head())