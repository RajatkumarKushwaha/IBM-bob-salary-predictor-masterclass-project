import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ── Load & train ──────────────────────────────────────────────────────────────
df = pd.read_csv("salary_data.csv")
X = df[["YearsExperience"]].values
y = df["Salary"].values

model = LinearRegression()
model.fit(X, y)

# ── Model quality metrics ─────────────────────────────────────────────────────
y_pred_train = model.predict(X)
r2  = r2_score(y, y_pred_train)
mae = mean_absolute_error(y, y_pred_train)

print("=" * 45)
print("       SALARY PREDICTION MODEL READY")
print("=" * 45)
print(f"  Formula : Salary = {model.coef_[0]:,.2f} x Years + {model.intercept_:,.2f}")
print(f"  R2 Score: {r2:.4f}  (1.0 = perfect fit)")
print(f"  Avg Error: +/-${mae:,.2f}")
print("=" * 45)
print("  Type a number to predict salary.")
print("  Type 'quit' to exit.")
print("=" * 45)

# ── Interactive loop ──────────────────────────────────────────────────────────
while True:
    user_input = input("\nEnter years of experience: ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    try:
        years = float(user_input)
        if years < 0:
            print("  Years of experience cannot be negative.")
            continue
        if years > 50:
            print("  Value seems unusually high -- proceeding anyway.")

        predicted = model.predict([[years]])[0]
        predicted = max(predicted, 0)          # salary can't be negative

        # Confidence band: +/- 1 MAE
        low  = max(predicted - mae, 0)
        high = predicted + mae

        print(f"\n  Predicted Salary : ${predicted:>12,.2f}")
        print(f"  Likely Range     : ${low:>12,.2f}  -  ${high:>12,.2f}")

    except ValueError:
        print("  Please enter a valid number (e.g. 3.5).")
