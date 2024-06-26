import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, binom, poisson, expon




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


# Initialize lists to store discrete and continuous variables
discrete_variables = []
continuous_variables = []

# Iterate through each column
for col in data.columns:
    # Check if the data type of the column is 'object' (categorical)
    if data[col].dtype == 'object':
        discrete_variables.append(col)
        # Print variable name and column data for discrete variables
        print(f"Discrete Variable - Column '{col}':")
        print(data[col])
        print()
    else:
        # Calculate the ratio of unique values to the total number of data points
        unique_ratio = data[col].nunique() / len(data)
        # Check if the unique ratio is less than a threshold (indicating discreteness)
        if unique_ratio < 0.1:
            discrete_variables.append(col)
            # Print variable name and column data for discrete variables
            print(f"Discrete Variable - Column '{col}':")
            print(data[col])
            print()
        else:
            continuous_variables.append(col)
            # Print variable name and column data for continuous variables
            print(f"Continuous Variable - Column '{col}':")
            print(data[col])
            print()

# Print discrete variables
print("Discrete Variables:")
print(discrete_variables)
print()

# Print continuous variables
print("Continuous Variables:")
print(continuous_variables)
# Calculate probabilities for each attribute
probabilities = {}

# Iterate over each column in the dataset
for column in data.columns:
    # Count the occurrences of each unique value in the column
    value_counts = data[column].value_counts()
    # Calculate the probability of each unique value by dividing its count by the total number of rows
    probabilities[column] = value_counts / len(data)

# Print probabilities
for column, probs in probabilities.items():
    print("Probabilities for", column)
    # Print the probabilities for each unique value in the column
    print(probs)
    print()
# Calculate probabilities for each attribute
probabilities = {}

# Iterate over each column in the dataset
for column in data.columns:
    # Count the occurrences of each unique value in the column
    value_counts = data[column].value_counts()
    # Calculate the probability of each unique value by dividing its count by the total number of rows
    probabilities[column] = value_counts / len(data)

# Print probabilities
for column, probs in probabilities.items():
    print("Probabilities for", column)
    # Print the probabilities for each unique value in the column
    print(probs)
    print()



def distribution_analysis(data):
    """
    Perform distribution analysis for each quantitative attribute in the dataset.

    Parameters:
    data (DataFrame): The dataset containing quantitative attributes.
    """
    # Define quantitative attributes to analyze
    quantitative_attributes = [
        'Births in last five years',
        'Age of respondent at 1st birth',
        'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)',
        'Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'
    ]

    # Iterate over each quantitative attribute
    for attribute in quantitative_attributes:
        # Visualize distribution with a histogram
        plt.figure(figsize=(8, 6))  # Create a new figure
        sns.histplot(data=data[attribute], kde=True)  # Plot histogram
        plt.title(f'Distribution of {attribute}')  # Set title
        plt.xlabel(attribute)  # Set x-axis label
        plt.ylabel('Frequency')  # Set y-axis label
        plt.show()  # Show plot

        # Visualize distribution with a box plot
        plt.figure(figsize=(8, 6))  # Create a new figure
        sns.boxplot(x=data[attribute])  # Plot box plot
        plt.xlabel(attribute)  # Set x-axis label
        plt.title(f'Box Plot of {attribute}')  # Set title
        plt.show()  # Show plot

        # Calculate mean and variance
        mean = np.mean(data[attribute])  # Calculate mean
        variance = np.var(data[attribute])  # Calculate variance
        print(f"Mean of {attribute}: {mean}")  # Print mean
        print(f"Variance of {attribute}: {variance}\n")  # Print variance



# Perform distribution analysis
distribution_analysis(data)

# Create DataFrame
df = pd.DataFrame(data)
for col in df.columns:
    # Count the frequency of each unique value in the column
    col_freq = df[col].value_counts()  # Counting the frequency of each unique value in the column

    # Total number of observations
    total_observations = len(df)  # Calculating the total number of observations in the dataset

    # Calculate the probability distribution
    probability_distribution = col_freq / total_observations  # Calculating the probability distribution for each unique value

    # Print probability distribution for the current column
    print(f"Probability Distribution for column '{col}':")  # Printing the column name
    print(probability_distribution)  # Printing the probability distribution for the column
    print()  # Printing an empty line for better readability


# Iterate over each column
for col in df.columns:
    if df[col].dtype in [int, float]:
        # Calculate the mean (λ) for Poisson distribution
        mean_value = df[col].mean()

        # Generate a random variable X with a Poisson distribution
        X = poisson.rvs(mu=mean_value, size=1000)

        # Visualize the Poisson distribution
        plt.hist(X, bins=20, density=True, alpha=0.6, color='g')
        plt.title(f'Poisson Distribution of {col}')
        plt.xlabel(f'{col}')
        plt.ylabel('Probability')
        plt.show()

        # Calculate mean and variance of the Poisson distribution
        mean_poisson = poisson.mean(mu=mean_value)
        variance_poisson = poisson.var(mu=mean_value)

        print(f"Mean of the Poisson distribution for '{col}': {mean_poisson}")
        print(f"Variance of the Poisson distribution for '{col}': {variance_poisson}")
        print()
        # Iterate over each column in the DataFrame


for col in df.columns:
    # Check if the column's data type is numeric (int or float)
    if df[col].dtype in [int, float]:
        # Calculate the probability of success (p) for binomial distribution
        p_value = df[col].mean() / df[col].max()  # Normalizing mean value to range [0, 1]

        # Number of trials (n)
        n_trials = 100  # You can adjust this value as needed

        # Generate a random variable X with a binomial distribution
        X = binom.rvs(n=n_trials, p=p_value, size=1000)  # Generate random samples

        # Visualize the binomial distribution using a histogram
        plt.hist(X, bins=20, density=True, alpha=0.6, color='b')  # Plot histogram
        plt.title(f'Binomial Distribution of {col}')  # Set title
        plt.xlabel(f'{col}')  # Set x-axis label
        plt.ylabel('Probability')  # Set y-axis label
        plt.show()  # Show the plot

        # Calculate mean and variance of the binomial distribution
        mean_binomial = n_trials * p_value  # Mean of binomial distribution
        variance_binomial = n_trials * p_value * (1 - p_value)  # Variance of binomial distribution

        # Print mean and variance of the binomial distribution
        print(f"Mean of the Binomial distribution for '{col}': {mean_binomial}")
        print(f"Variance of the Binomial distribution for '{col}': {variance_binomial}")
        print()  # Print empty line for better readability
        # Define the normal PDF function


def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Apply normal_pdf function to each column
for col in data.columns:
    if col != 'ID':  # Skip 'ID' column if it exists
        # Check if the column contains numeric values
        if pd.api.types.is_numeric_dtype(data[col]):
            mu = data[col].mean()  # Calculate mean of the column
            sigma = data[col].std()  # Calculate standard deviation of the column
            x_values = np.linspace(data[col].min(), data[col].max(), 100)  # Generate values for x from min to max of the column
            pdf_values = normal_pdf(x_values, mu, sigma)  # Calculate PDF values for the generated x values
            # Plot PDF values
            plt.plot(x_values, pdf_values, label=col)
            plt.legend()
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.title('Normal Distribution PDFs for Each Column')
            plt.show()

# Show legend and plot



# Function to calculate the Bernoulli distribution for a given column
def bernoulli_pmf(data_column, p):
    """
    Calculate the Probability Mass Function (PMF) values of a Bernoulli distribution for a given column.

    Parameters:
        data_column (Series): The column data for which to calculate the PMF.
        p (float): The probability of success.

    Returns:
        list: The PMF values corresponding to the input data column.
    """
    pmf_values = [(p if val == 1 else 1 - p) for val in data_column]
    return pmf_values



# Iterate over each column (excluding the 'ID' column if it exists)
for col in data.columns:
    if col != 'ID':
        # Check if the column contains binary values
        if data[col].nunique() == 2:
            # Ensure that only numeric values are considered
            numeric_data = data[col].apply(pd.to_numeric, errors='coerce').dropna()
            if not numeric_data.empty:
                # Calculate the probability of success (p) as the proportion of ones in the column
                p = numeric_data.sum() / len(numeric_data)
                # Calculate the PMF values using the Bernoulli distribution formula
                pmf_values = bernoulli_pmf(numeric_data, p)
                # Print or use the pmf_values as needed
                print(f"Column '{col}':")
                print(pmf_values)
            else:
                print(f"Column '{col}' does not contain numeric data.")
        else:
            print(f"Column '{col}' is not binary.")


# Function to calculate the exponential distribution for a given column


def exponential_pdf(data_column, rate):
    """
    Calculate the probability density function (PDF) of the exponential distribution.

    Parameters:
        data_column (array-like): The input values at which to evaluate the PDF.
        rate (float): The rate parameter (λ) of the exponential distribution.

    Returns:
        array-like: The PDF values corresponding to the input values.
    """
    pdf_values = rate * np.exp(-rate * data_column)
    return pdf_values



try:
    
    # Check for non-numeric columns
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

    # Print non-numeric columns if exist
    if len(non_numeric_columns) > 0:
        print("\nNon-numeric columns:")
        for col in non_numeric_columns:
            print(col)
    else:
        print("\nAll columns are numeric.")

    # Iterate over each numeric column
    for col in data.select_dtypes(include=[np.number]).columns:
        # Calculate the rate parameter (λ) as the reciprocal of the mean of the column
        rate = 1 / data[col].mean()
        # Calculate the PDF values using the exponential distribution formula
        pdf_values = exponential_pdf(data[col], rate)
        # Print column name
        print(f"\nColumn '{col}':")
        # Print PDF values
        print(pdf_values)

except FileNotFoundError:
    print("File not found. Please provide the correct file path.")
except Exception as e:
    print("An error occurred:", str(e))
    # Function to automatically select and calculate distribution for each column
def calculate_distribution(data_column):
    """
    Automatically select and calculate the appropriate distribution for a given column.

    Parameters:
        data_column (array-like): The input values from the dataset column.

    Returns:
        array-like: The distribution values corresponding to the input values.
    """
    # Check if the column contains only zeros or ones (indicating a Bernoulli distribution)
    if set(data_column.unique()) == {0, 1}:
        # Calculate the probability of success (p) as the proportion of ones in the column
        p = data_column.mean()
        # Calculate the PMF values using the Bernoulli distribution formula
        pmf_values = [(p if val == 1 else 1 - p) for val in data_column]
        return pmf_values

    # Check if the column contains only positive values (indicating an exponential or lognormal distribution)
    elif data_column.min() > 0:
        # Calculate mean and standard deviation for the associated lognormal distribution
        mu = np.log(data_column.mean())
        sigma = np.log(1 + (data_column.std() / data_column.mean()))
        # Calculate the PDF values using the lognormal distribution formula
        pdf_values = (1 / (data_column * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(data_column) - mu)**2 / (2 * sigma**2))
        return pdf_values

    # Default to the normal distribution for other cases
    else:
        # Calculate mean and standard deviation
        mu = data_column.mean()
        sigma = data_column.std()
        # Calculate the PDF values using the normal distribution formula
        pdf_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data_column - mu) / sigma)**2)
        return pdf_values

try:
    # Assuming 'data' is a DataFrame loaded from a file
    # Iterate over each column
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            # Calculate the distribution values using the calculate_distribution function
            distribution_values = calculate_distribution(data[col])
            # Print column name
            print(f"Column '{col}':")
            # Print original data
            print("Original Data:")
            print(data[col])
            # Print distribution values
            print("Distribution Values:")
            print(distribution_values)
            print()

except FileNotFoundError:
    print("File not found. Please provide the correct file path.")
except Exception as e:
    print("An error occurred:", str(e))
    def lognormal_pdf(x, mu, sigma):
        x = np.array(x)
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for divide by zero and invalid values
            pdf = np.exp(-(np.log(x) - mu)*2 / (2 * sigma*2)) / (x * sigma * np.sqrt(2 * np.pi))
            pdf[~np.isfinite(pdf)] = 0  # Set non-finite values (inf, -inf, NaN) to 0

        return pdf

# Assuming you have imported the necessary libraries and defined 'data'

# Identify numerical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns

# Now you can iterate over the selected numerical columns
for col in numerical_columns:
    col_values = data[col]
    # Calculate mean and standard deviation for the associated normal distribution
    mu = np.log(np.mean(col_values))
    sigma = np.log(1 + (np.std(col_values) / np.mean(col_values)))
    try:
        # Calculate the PDF values using the lognormal distribution formula
        pdf_values = lognormal_pdf(col_values, mu, sigma)
        # Print the output
        print(f"Column '{col}' PDF:")
        print(pdf_values)
    except ValueError as e:
        print(f"Error processing column '{col}': {e}")
        
# Function to calculate the composition distribution for a given column
def calculate_composition_distribution(data_column):
    """
    Calculate the composition distribution for a given column based on its characteristics.

    Parameters:
        data_column (array-like): The input values from the dataset column.

    Returns:
        array-like: The distribution values corresponding to the input values.
    """
    # Check if the column contains only zeros or ones (indicating a Bernoulli distribution)
    if set(data_column) == {0, 1}:
        # Fit a Bernoulli distribution to the data
        p = np.mean(data_column)
        pmf_values = np.where(data_column == 1, p, 1 - p)
        return pmf_values
    
    # Check if the column contains only positive values (indicating an exponential or lognormal distribution)
    elif min(data_column) > 0:
        # Fit an Exponential distribution to the data
        loc, scale = np.mean(data_column), np.std(data_column)
        pdf_values = (1 / (scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data_column - loc) / scale) ** 2)
        return pdf_values
    
    # Default to the normal distribution for other cases
    else:
        # Fit a Normal distribution to the data
        loc, scale = np.mean(data_column), np.std(data_column)
        pdf_values = (1 / (scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data_column - loc) / scale) ** 2)
        return pdf_values

# Assuming 'data' is a dictionary loaded from a file
# Convert 'data' to a DataFrame (replace this with your actual data loading logic)
data = pd.DataFrame(data)

try:
    # Iterate over each column
    for col in data.columns:
        col_values = data[col]
        if np.issubdtype(col_values.dtype, np.number):
            # Calculate the composition distribution values using the calculate_composition_distribution function
            composition_distribution_values = calculate_composition_distribution(col_values)
            # Print column name
            print(f"Column '{col}':")
            # Print original data
            print("Original Data:")
            print(col_values)
            # Print composition distribution
            print("Composition Distribution:")
            print(composition_distribution_values)
            print()

except Exception as e:
    print("An error occurred:", str(e))


# Function to calculate the composition distribution for a given column
def calculate_composition_distribution(data_column):
    """
    Calculate the composition distribution for a given column based on its characteristics.

    Parameters:
        data_column (array-like): The input values from the dataset column.

    Returns:
        array-like: The distribution values corresponding to the input values.
    """
    # Check if the column contains only zeros or ones (indicating a Bernoulli distribution)
    if set(data_column.unique()) == {0, 1}:
        # Fit a Bernoulli distribution to the data
        p = data_column.mean()
        pmf_values = np.where(data_column == 1, p, 1 - p)
        return pmf_values
    
    # Check if the column contains only positive values (indicating an exponential or lognormal distribution)
    elif data_column.min() > 0:
        # Fit an Exponential distribution to the data
        loc, scale = np.mean(data_column), np.std(data_column)
        pdf_values = (1 / (scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data_column - loc) / scale) ** 2)
        return pdf_values
    
    # Default to the normal distribution for other cases
    else:
        # Fit a Normal distribution to the data
        loc, scale = np.mean(data_column), np.std(data_column)
        pdf_values = (1 / (scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data_column - loc) / scale) ** 2)
        return pdf_values

try:
    # Assuming 'data' is a DataFrame loaded from a file
    # Iterate over each column
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            # Calculate the composition distribution values using the calculate_composition_distribution function
            composition_distribution_values = calculate_composition_distribution(data[col])
            # Print column name
            print(f"Column '{col}':")
            # Print original data
            print("Original Data:")
            print(data[col])
            # Print composition distribution
            print("Composition Distribution:")
            print(composition_distribution_values)
            print()

except FileNotFoundError:
    print("File not found. Please provide the correct file path.")
except Exception as e:
    print("An error occurred:", str(e))
