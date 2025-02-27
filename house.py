import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Generate synthetic data (no fixed seed for randomness)
np.random.seed()  # This will generate different data each time

# Features: Bedrooms, Square footage, Age, and Number of Bathrooms
bedrooms = np.random.randint(1, 6, 100)  # Bedrooms (1-5)
sqft = np.random.randint(500, 5000, 100)  # Square footage (500-5000)
age = np.random.randint(1, 50, 100)  # Age (1-50 years)
bathrooms = np.random.randint(1, 4, 100)  # Number of bathrooms (1-3)

# Target: Price of the house (including the new feature effect)
price = 50000 + (bedrooms * 20000) + (sqft * 50) - (age * 100) + (bathrooms * 15000) + np.random.randint(-10000, 10000, 100)

# Create a DataFrame
data = pd.DataFrame({
    'Bedrooms': bedrooms,
    'Square_Feet': sqft,
    'Age': age,
    'Bathrooms': bathrooms,
    'Price': price
})

# Features (X) and Target (y)
X = data[['Bedrooms', 'Square_Feet', 'Age', 'Bathrooms']]
y = data['Price']

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling: Standardize the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest Regressor model
model = RandomForestRegressor()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, None],  # Max depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best model after tuning hyperparameters
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()

# Cross-validation to evaluate model performance on multiple splits
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-cv_scores.mean()}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Generate synthetic data (no fixed seed for randomness)
np.random.seed()  # This will generate different data each time

# Features: Bedrooms, Square footage, Age, and Number of Bathrooms
bedrooms = np.random.randint(1, 6, 100)  # Bedrooms (1-5)
sqft = np.random.randint(500, 5000, 100)  # Square footage (500-5000)
age = np.random.randint(1, 50, 100)  # Age (1-50 years)
bathrooms = np.random.randint(1, 4, 100)  # Number of bathrooms (1-3)

# Target: Price of the house (including the new feature effect)
price = 50000 + (bedrooms * 20000) + (sqft * 50) - (age * 100) + (bathrooms * 15000) + np.random.randint(-10000, 10000, 100)

# Create a DataFrame
data = pd.DataFrame({
    'Bedrooms': bedrooms,
    'Square_Feet': sqft,
    'Age': age,
    'Bathrooms': bathrooms,
    'Price': price
})

# Features (X) and Target (y)
X = data[['Bedrooms', 'Square_Feet', 'Age', 'Bathrooms']]
y = data['Price']

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling: Standardize the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest Regressor model
model = RandomForestRegressor()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, None],  # Max depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best model after tuning hyperparameters
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()

# Cross-validation to evaluate model performance on multiple splits
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-cv_scores.mean()}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Generate synthetic data (no fixed seed for randomness)
np.random.seed()  # This will generate different data each time

# Features: Bedrooms, Square footage, Age, and Number of Bathrooms
bedrooms = np.random.randint(1, 6, 100)  # Bedrooms (1-5)
sqft = np.random.randint(500, 5000, 100)  # Square footage (500-5000)
age = np.random.randint(1, 50, 100)  # Age (1-50 years)
bathrooms = np.random.randint(1, 4, 100)  # Number of bathrooms (1-3)

# Target: Price of the house (including the new feature effect)
price = 50000 + (bedrooms * 20000) + (sqft * 50) - (age * 100) + (bathrooms * 15000) + np.random.randint(-10000, 10000, 100)

# Create a DataFrame
data = pd.DataFrame({
    'Bedrooms': bedrooms,
    'Square_Feet': sqft,
    'Age': age,
    'Bathrooms': bathrooms,
    'Price': price
})

# Features (X) and Target (y)
X = data[['Bedrooms', 'Square_Feet', 'Age', 'Bathrooms']]
y = data['Price']

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling: Standardize the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest Regressor model
model = RandomForestRegressor()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, None],  # Max depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best model after tuning hyperparameters
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()

# Cross-validation to evaluate model performance on multiple splits
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-cv_scores.mean()}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Generate synthetic data (no fixed seed for randomness)
np.random.seed()  # This will generate different data each time

# Features: Bedrooms, Square footage, Age, and Number of Bathrooms
bedrooms = np.random.randint(1, 6, 100)  # Bedrooms (1-5)
sqft = np.random.randint(500, 5000, 100)  # Square footage (500-5000)
age = np.random.randint(1, 50, 100)  # Age (1-50 years)
bathrooms = np.random.randint(1, 4, 100)  # Number of bathrooms (1-3)

# Target: Price of the house (including the new feature effect)
price = 50000 + (bedrooms * 20000) + (sqft * 50) - (age * 100) + (bathrooms * 15000) + np.random.randint(-10000, 10000, 100)

# Create a DataFrame
data = pd.DataFrame({
    'Bedrooms': bedrooms,
    'Square_Feet': sqft,
    'Age': age,
    'Bathrooms': bathrooms,
    'Price': price
})

# Features (X) and Target (y)
X = data[['Bedrooms', 'Square_Feet', 'Age', 'Bathrooms']]
y = data['Price']

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling: Standardize the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest Regressor model
model = RandomForestRegressor()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, None],  # Max depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best model after tuning hyperparameters
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()

# Cross-validation to evaluate model performance on multiple splits
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-cv_scores.mean()}")
