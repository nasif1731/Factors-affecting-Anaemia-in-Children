import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, binom, poisson, expon
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt




#file_path = '/Users/nasif/Downloads/archive/children anemia.csv'
data = pd.read_csv("children_anaemia.csv")

qualitative_attributes = [
    'Age in 5-year groups',
    'Type of place of residence',
    'Highest educational level',
    'Wealth index combined',
    'Current marital status',
    'Currently residing with husband/partner',
    'When child put to breast',
    'Had fever in last two weeks',
    'Anemia level',
    'Have mosquito bed net for sleeping (from household questionnaire)',
    'Smokes cigarettes',
    'Taking iron pills, sprinkles or syrup'
]

quantitative_attributes = [

    'Births in last five years',
    'Age of respondent at 1st birth',
    'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)',
    'Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'
]



# Define a mapping from numeric values to textual representations
value_mapping = {   'Immediately': 0,    'Hours: 1': 60, 'Days: 1':1440}
# Replace the values in the 'Values' column using the mapping
data['When child put to breast'] = data['When child put to breast'].replace(value_mapping)


# Convert the object column to an integer column
data['When child put to breast'] = data['When child put to breast'].astype(float)


# Calculate the average of the column
column_average = data['When child put to breast'].mean()

# Replace missing values (NaN) with the column average
data['When child put to breast'].fillna(column_average, inplace=True)



numeric_columns = data.select_dtypes(include=["float", "int"]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())


data.drop_duplicates()



# Encode the dependent variable
label_encoder = LabelEncoder()
data['Anemia level'] = label_encoder.fit_transform(data['Anemia level'])

# Define the dependent variable and its corresponding independent variables
dependent_variable = 'Anemia level'
independent_variables = [
    'Age in 5-year groups',
    'Type of place of residence',
    'Highest educational level',
    'Wealth index combined',
    'Have mosquito bed net for sleeping (from household questionnaire)',
    'Smokes cigarettes',
    'Current marital status',
    'Currently residing with husband/partner',
    'When child put to breast',
    'Had fever in last two weeks',
    'Taking iron pills, sprinkles or syrup'
]



# Iterate through each independent variable
for independent_variable in independent_variables:
    # Prepare data for the current combination of variables
    X = data[[independent_variable]].values  # Independent variable
    y = data[dependent_variable].values  # Dependent variable

    # One-hot encode categorical variables
    if isinstance(X[0][0], str):
        encoder = LabelEncoder()
        X = encoder.fit_transform(X)
    X = X.reshape(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot the data points and the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.title(f"{independent_variable} vs {dependent_variable}")
    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print results
    print(f"Independent Variable: {independent_variable}")
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared (R2): {r2}')
    print()




# Iterate through each independent variable
for independent_variable in independent_variables:
    # Prepare data for the current combination of variables
    X = data[[independent_variable]].values  # Independent variable
    y = data[dependent_variable].values  # Dependent variable

    # One-hot encode categorical variables
    if isinstance(X[0][0], str):
        encoder = LabelEncoder()
        X = encoder.fit_transform(X)
    X = X.reshape(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print results
    print(f"Independent Variable: {independent_variable}")

    # Get the coefficients
    coefficient = model.coef_[0]
    intercept = model.intercept_
    print(f"Coefficient: {coefficient}")
    print(f"Intercept: {intercept}")

    print()




# Choose an independent variable for prediction
independent_variable = 'Age in 5-year groups'

# Prepare data for the selected independent variable
X = data[[independent_variable]].values  # Independent variable
y = data[dependent_variable].values  # Dependent variable

# One-hot encode categorical variables if needed
if isinstance(X[0][0], str):
    encoder = LabelEncoder()
    X = encoder.fit_transform(X)
X = X.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')

# Plot regression line
x_values = np.linspace(X.min(), X.max(), 100)
y_values = model.predict(x_values.reshape(-1, 1))
plt.plot(x_values, y_values, color='red', label='Regression Line')

# Plot the point for prediction
new_data_point = 30  # Example data point for prediction
predicted_anemia_level = model.predict([[new_data_point]])
plt.scatter(new_data_point, predicted_anemia_level, color='purple', label='Predicted Point')

plt.title(f"{independent_variable} vs {dependent_variable}")
plt.xlabel(independent_variable)
plt.ylabel(dependent_variable)
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted Anemia Level for {independent_variable} = {predicted_anemia_level[0]}")
#Define thresholds for classification
threshold_mild = 0
threshold_moderate = 0.5  # Example threshold

# Classify the predicted anemia level
if predicted_anemia_level < threshold_mild:
    classification = "Not anemic"
elif predicted_anemia_level < threshold_moderate:
    classification = "Mild"
else:
    classification = "Moderate"

print("Predicted Anemia Level Classification:", classification)




# Choose an independent variable for prediction
independent_variable = 'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)'

# Prepare data for the selected independent variable
X = data[[independent_variable]].values  # Independent variable
y = data[dependent_variable].values  # Dependent variable

# One-hot encode categorical variables if needed
if isinstance(X[0][0], str):
    encoder = LabelEncoder()
    X = encoder.fit_transform(X)
X = X.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')

# Plot regression line
x_values = np.linspace(X.min(), X.max(), 100)
y_values = model.predict(x_values.reshape(-1, 1))
plt.plot(x_values, y_values, color='red', label='Regression Line')

# Plot the point for prediction
new_data_point = 30  # Example data point for prediction
predicted_anemia_level = model.predict([[new_data_point]])
plt.scatter(new_data_point, predicted_anemia_level, color='purple', label='Predicted Point')

plt.title(f"{independent_variable} vs {dependent_variable}")
plt.xlabel(independent_variable)
plt.ylabel(dependent_variable)
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted Anemia Level for {independent_variable} = {predicted_anemia_level[0]}")
#Define thresholds for classification
threshold_mild = 0
threshold_moderate = 0.5  # Example threshold

# Classify the predicted anemia level
if predicted_anemia_level < threshold_mild:
    classification = "Not anemic"
elif predicted_anemia_level < threshold_moderate:
    classification = "Mild"
else:
    classification = "Moderate"

print("Predicted Anemia Level Classification:", classification)




# Choose an independent variable for prediction
independent_variable = 'Type of place of residence'

# Prepare data for the selected independent variable
X = data[[independent_variable]].values  # Independent variable
y = data[dependent_variable].values  # Dependent variable

# One-hot encode categorical variables if needed
if isinstance(X[0][0], str):
    encoder = LabelEncoder()
    X = encoder.fit_transform(X)
X = X.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')

# Plot regression line
x_values = np.linspace(X.min(), X.max(), 100)
y_values = model.predict(x_values.reshape(-1, 1))
plt.plot(x_values, y_values, color='red', label='Regression Line')

# Plot the point for prediction
new_data_point = 30  # Example data point for prediction
predicted_anemia_level = model.predict([[new_data_point]])
plt.scatter(new_data_point, predicted_anemia_level, color='purple', label='Predicted Point')

plt.title(f"{independent_variable} vs {dependent_variable}")
plt.xlabel(independent_variable)
plt.ylabel(dependent_variable)
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted Anemia Level for {independent_variable} = {predicted_anemia_level[0]}")
##Define thresholds for classification
threshold_mild = 0
threshold_moderate = 0.5  # Example threshold

# Classify the predicted anemia level
if predicted_anemia_level < threshold_mild:
    classification = "Not anemic"
elif predicted_anemia_level < threshold_moderate:
    classification = "Mild"
else:
    classification = "Moderate"

print("Predicted Anemia Level Classification:", classification)




# Choose an independent variable for prediction
independent_variable = 'Highest educational level'

# Prepare data for the selected independent variable
X = data[[independent_variable]].values  # Independent variable
y = data[dependent_variable].values  # Dependent variable

# One-hot encode categorical variables if needed
if isinstance(X[0][0], str):
    encoder = LabelEncoder()
    X = encoder.fit_transform(X)
X = X.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')

# Plot regression line
x_values = np.linspace(X.min(), X.max(), 100)
y_values = model.predict(x_values.reshape(-1, 1))
plt.plot(x_values, y_values, color='red', label='Regression Line')

# Plot the point for prediction
new_data_point = 30  # Example data point for prediction
predicted_anemia_level = model.predict([[new_data_point]])
plt.scatter(new_data_point, predicted_anemia_level, color='purple', label='Predicted Point')

plt.title(f"{independent_variable} vs {dependent_variable}")
plt.xlabel(independent_variable)
plt.ylabel(dependent_variable)
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted Anemia Level for {independent_variable} = {predicted_anemia_level[0]}")
#Define thresholds for classification
threshold_mild = 0
threshold_moderate = 0.5  # Example threshold

# Classify the predicted anemia level
if predicted_anemia_level < threshold_mild:
    classification = "Not anemic"
elif predicted_anemia_level < threshold_moderate:
    classification = "Mild"
else:
    classification = "Moderate"

print("Predicted Anemia Level Classification:", classification)

