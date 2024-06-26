import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, binom, poisson, expon
import scipy
from bootstrapped import bootstrap as bs
from bootstrapped import stats_functions as bs_stats
import statsmodels.api as sm



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


# Calculate correlation matrix
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

# Visualize correlation matrix with heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Correlation Matrix')
plt.show()
#The heatmap color-codes the correlation coefficients, where warmer colors (e.g., red) indicate a strong positive correlation, cooler colors (e.g., blue) represent a strong negative correlation, and neutral colors (e.g., white) suggest little to no correlation.
#The sign of the correlation coefficient indicates the direction of the relationship between variables. A positive correlation coefficient suggests that as one variable increases, the other variable also tends to increase, while a negative correlation coefficient indicates that as one variable increases, the other tends to decrease.


numeric_data = data.select_dtypes(include=[np.number])


correlation_matrix = numeric_data.corr()

correlation_matrix

numeric_columns = data.select_dtypes(include=np.number).columns

print(numeric_columns)

for column in numeric_columns:
    # Bootstrap resampling for the column
    bootstrap_results = bs.bootstrap(data[column].values, stat_func=bs_stats.mean, num_iterations=1000, alpha=0.05)
    mean_estimate = bootstrap_results.value
    lower_ci, upper_ci = bootstrap_results.lower_bound, bootstrap_results.upper_bound
    print(f"Confidence interval for mean of {column}: Estimate={mean_estimate}, CI=({lower_ci}, {upper_ci})")
    
    
X = data[['Hemoglobin level adjusted for altitude (g/dl - 1 decimal)', 'Age of respondent at 1st birth']]  # Replace with your predictor variables
X = sm.add_constant(X)  # Add constant term for intercept
y = data['Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)']  # Replace with your target variable

# Fit ordinary least squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Compute confidence intervals for regression coefficients
coefficients = model.params
conf_int = model.conf_int(alpha=0.05)  # 95% confidence interval
conf_int.columns = ['Lower CI', 'Upper CI']
coefficients_with_ci = pd.concat([coefficients, conf_int], axis=1)
print("\nRegression coefficients with 95% confidence intervals:")
print(coefficients_with_ci)

numeric_columns = data.select_dtypes(include=np.number).columns

for column in numeric_columns:
    # Bootstrap resampling for the column
    bootstrap_results = bs.bootstrap(data[column].values, stat_func=bs_stats.mean, num_iterations=1000, alpha=0.05)
    mean_estimate = bootstrap_results.value
    lower_ci, upper_ci = bootstrap_results.lower_bound, bootstrap_results.upper_bound
    print(f"Confidence interval for mean of {column}: Estimate={mean_estimate}, CI=({lower_ci}, {upper_ci})")
    
    
X = data[['Births in last five years']]  # Replace with your predictor variables
X = sm.add_constant(X)  # Add constant term for intercept
y = data['When child put to breast']  # Replace with your target variable

# Fit ordinary least squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Compute confidence intervals for regression coefficients
coefficients = model.params
conf_int = model.conf_int(alpha=0.05)  # 95% confidence interval
conf_int.columns = ['Lower CI', 'Upper CI']
coefficients_with_ci = pd.concat([coefficients, conf_int], axis=1)
print("\nRegression coefficients with 95% confidence intervals:")
print(coefficients_with_ci)

numeric_columns = data.select_dtypes(include=np.number).columns

for column in numeric_columns:
    # Bootstrap resampling for the column
    bootstrap_results = bs.bootstrap(data[column].values, stat_func=bs_stats.mean, num_iterations=1000, alpha=0.05)
    mean_estimate = bootstrap_results.value
    lower_ci, upper_ci = bootstrap_results.lower_bound, bootstrap_results.upper_bound
    print(f"Confidence interval for mean of {column}: Estimate={mean_estimate}, CI=({lower_ci}, {upper_ci})")
    
    
X = data[['When child put to breast']]  # Replace with your predictor variables
X = sm.add_constant(X)  # Add constant term for intercept
y = data['Births in last five years']  # Replace with your target variable

# Fit ordinary least squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Compute confidence intervals for regression coefficients
coefficients = model.params
conf_int = model.conf_int(alpha=0.05)  # 95% confidence interval
conf_int.columns = ['Lower CI', 'Upper CI']
coefficients_with_ci = pd.concat([coefficients, conf_int], axis=1)
print("\nRegression coefficients with 95% confidence intervals:")
print(coefficients_with_ci)