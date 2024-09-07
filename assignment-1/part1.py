# %%
import pandas as pd
import numpy as np
import requests
from io import StringIO
import seaborn as sns
from sklearn.model_selection import train_test_split
from itertools import product
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='output_part1.log', level=logging.INFO, format='%(asctime)s - %(message)s')


# %%
def async_read_csv(url):
    response = requests.get(url, stream=True)
    response.raise_for_status() # will raise error if request is not successful
    return pd.read_csv(StringIO(response.text))
    
# Read the streamed content
df = async_read_csv("https://raw.githubusercontent.com/uy-seng/cs4375/main/assignment-1/scripts/convert_to_csv/abalone.csv")
df.head(), df.shape

# %%
df.info()

# %%
df.describe()

# %%
# check for null or na values
df.isnull().sum()

# %%
# check for redundant rows
df.duplicated().sum()

# %%
df.sex.unique()

# %%
# convert categorical variables to numerical variables
df['sex'] = df['sex'].map({'M': 1, 'F': 2, 'I': 3})

# %%
# remove attribute that is not correlated to the outcome
correlation_matrix = df.corr()
correlation_matrix['rings'].sort_values(ascending=False)

# %%
# remove sex since ring does not correlate with sex
if "sex" in df: del df["sex"]

# %%
x = df.drop("rings", axis=1).values
y = df["rings"].values

# %%
x.shape

# %%
x.shape, y.shape

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# %%
x_train.shape, x_test.shape, y_train.shape, y_test.shape 

# %%
def gradient_of_cost_func(x_values, y_values, thetas):
    x_values_with_bias = np.c_[np.ones(len(x_values)), x_values]  # Add bias term
    predictions = np.dot(x_values_with_bias, thetas) 
    errors = predictions - y_values
    return np.dot(x_values_with_bias.T, errors) / len(y_values)

def cost_func(x_values, y_values, thetas):
    x_values_with_bias = np.c_[np.ones(len(x_values)), x_values]  # Add bias term
    predictions = np.dot(x_values_with_bias, thetas)
    return (1 / 2) * np.mean((predictions - y_values) ** 2)

def gradient_descent(x_values, y_values, learning_rate=0.01, threshold=1e-5, max_iterations=100000):
    thetas = np.random.rand(x_values.shape[1] + 1)  # +1 for bias term
    costs = []
    for _ in range(max_iterations):
        delta = -learning_rate * gradient_of_cost_func(x_values, y_values, thetas)
        costs.append(cost_func(x_values, y_values, thetas))
        if np.all(np.abs(delta) <= threshold):
            break
        thetas += delta
    return thetas, costs

def calculate_r_squared(y_actual, y_predict):
    # Residual Sum of Squares (RSS)
    rss = np.sum((y_actual - y_predict) ** 2)
    # Total Sum of Squares (TSS)
    tss = np.sum((y_actual - np.mean(y_actual)) ** 2)
    # R^2 calculation
    r_squared = 1 - (rss / tss)
    return r_squared

def grid_search(x_train, y_train, x_test, y_test, learning_rates, thresholds, max_iterations_list):
    best_r2 = -np.inf
    best_params = None
    results = []

    for lr, th, max_iter in product(learning_rates, thresholds, max_iterations_list):
        thetas, _ = gradient_descent(x_train, y_train, learning_rate=lr, threshold=th, max_iterations=max_iter)
        y_predict = np.dot(np.c_[np.ones(len(x_test)), x_test], thetas)
        r2 = calculate_r_squared(y_test, y_predict)
        
        # Print the R² score for each parameter combination
        logging.info(f"Parameters: Learning rate={lr}, Threshold={th}, Max iterations={max_iter} => R²={r2}")
        
        # Record the parameters and R² score
        results.append((lr, th, max_iter, r2))

        if r2 > best_r2:
            best_r2 = r2
            best_params = (lr, th, max_iter)

    return best_params, best_r2, results


# %% [markdown]
# 

# %%
# Define parameter grid
learning_rates = [0.001, 0.01, 0.1]
thresholds = [1e-6, 1e-5, 1e-4]
max_iterations_list = [5000, 10000, 50000]

# Perform grid search
best_params, best_r2, results = grid_search(x_train, y_train, x_test, y_test, learning_rates, thresholds, max_iterations_list)

# Print the best parameters and R²
print(f"Best parameters: Learning rate={best_params[0]}, Threshold={best_params[1]}, Max iterations={best_params[2]}")
print(f"Best R²: {best_r2}")

# Display all results
results_df = pd.DataFrame(results, columns=["Learning Rate", "Threshold", "Max Iterations", "R² Score"])


# Line plot of R² score vs Max Iterations for each Learning Rate
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x="Max Iterations", y="R² Score", hue="Learning Rate", marker="o")

plt.title("R² Score vs Max Iterations for Different Learning Rates")
plt.xlabel("Max Iterations")
plt.ylabel("R² Score")
plt.legend(title="Learning Rate")
plt.grid(True)
plt.show()


# Pivot table to format data for heatmap
pivot_table = results_df.pivot_table(index="Learning Rate", columns="Max Iterations", values="R² Score", aggfunc='mean')

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm")

plt.title("R² Score Heatmap for Learning Rate and Max Iterations")
plt.xlabel("Max Iterations")
plt.ylabel("Learning Rate")
plt.show()

# Threshold vs R2 score
plt.figure(figsize=(8, 6))
sns.lineplot(data=results_df, x="Threshold", y="R² Score", hue="Learning Rate", marker="o")
plt.title("R² Score vs Threshold for Different Learning Rates")
plt.xlabel("Threshold")
plt.ylabel("R² Score")
plt.legend(title="Learning Rate")
plt.grid(True)
plt.show()

