import pandas as pd

# Load the validation results CSV
results_df = pd.read_csv("results.csv")

# Get the last row
last_row = results_df.tail(1)

# Print the last row
print(last_row)

