
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import statsmodels.api as sm
from scipy.stats import poisson, bernoulli, norm
from PIL import Image

# Load the dataset
st.title("Children's Anemia Data Analysis with Advanced Statistics")
data = pd.read_csv("children_anaemia.csv")

image = Image.open("Logo.png")
st.sidebar.image(image, use_column_width=True)
# Function to clean data
def clean_data(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        df[col].fillna(df[col].mean(), inplace=True)


# Sidebar: Choose a category and specific analysis/plot
st.sidebar.title("CEREBRAL COLLECTIVE")
st.sidebar.header("Choose a Category")
category = st.sidebar.radio("Choose a category:", ["Data Overview", "Graphs", "Regression Analysis", "Distributions"])

if category == "Data Overview":
    st.header("Data Overview")
    st.write("Here are some statistics and information about the dataset:")
    st.write(data.describe())
    st.write("Columns with NaNs:")
    st.write(data.isnull().sum())

elif category == "Distributions":
    distribution_type = st.sidebar.selectbox("Select Distribution:", ["Poisson", "Bernoulli", "Normal"])

    if distribution_type == "Poisson":
        st.header("Poisson Distribution Analysis")
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                mean_value = data[col].mean()  # Calculate the mean for Poisson
                X = poisson.rvs(mu=mean_value, size=1000)  # Generate Poisson random variables
                plt.hist(X, bins=20, density=True, alpha=0.6, color='g')
                plt.title(f'Poisson Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Probability')
                st.pyplot(plt)

                mean_poisson = poisson.mean(mu=mean_value)
                variance_poisson = poisson.var(mu=mean_value)
                st.write(f"Mean of the Poisson distribution for '{col}': {mean_poisson}")
                st.write(f"Variance of the Poisson distribution for '{col}': {variance_poisson}")

    elif distribution_type == "Binomial":
        st.header("Binomial Distribution Analysis")
        bernoulli_col = st.sidebar.selectbox("Select a Column:", data.columns.tolist())
        
        if pd.api.types.is_numeric_dtype(data[bernoulli_col]):
            p = 0.5  # Probability of success
            pmf_values = [(p if val == 1 else 1 - p) for val in data[bernoulli_col]]
            st.write(f"Binomial PMF for '{bernoulli_col}': {pmf_values}")

            plt.hist(data[bernoulli_col], bins=2, alpha=0.6, color='b', rwidth=0.85)
            plt.title(f'Binomial Distribution for {bernoulli_col}')
            plt.xlabel(bernoulli_col)
            plt.ylabel('Count')
            st.pyplot(plt)

    elif distribution_type == "Normal":
        st.header("Normal Distribution Analysis")
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                mu = data[col].mean()  # Calculate mean of the column
                sigma = data[col].std()  # Calculate standard deviation
                x_values = np.linspace(data[col].min(), data[col].max(), 100)  # Generate values for x
                pdf_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x_values - mu)**2 / (2 * sigma**2))  # Calculate PDF
                plt.plot(x_values, pdf_values, label=col)
                plt.title(f'Normal Distribution PDFs for {col}')
                plt.xlabel('Value')
                plt.ylabel('Probability Density')
                plt.legend()
                st.pyplot(plt)


elif category == "Graphs":
    st.header("Graphs")
    
    plot_type = st.sidebar.selectbox("Select Graphs Type", [
        "Bar Charts",
        "Histograms",
        "Box Plots",
        "Correlation Matrix",
        "Pairplot",
        "Scatter Plot",
        "Bootstrap Confidence Intervals"
    ])
    
    if plot_type == "Bar Charts":
        plot_column = st.sidebar.selectbox("Choose a bar charts:", [
            "Age in 5-year groups",
            "Type of place of residence",
            "Highest educational level",
            "Wealth index combined",
            "Current marital status",
            "Currently residing with husband/partner",
            "When child put to breast",
            "Had fever in last two weeks",
            "Anemia level",
            "Have mosquito bed net for sleeping (from household questionnaire)",
            "Smokes cigarettes",
            "Taking iron pills, sprinkles or syrup"
        ])
        # Generate and display the selected bar plot
        data[plot_column].value_counts().plot(kind='bar', title=f'{plot_column}')
        plt.xlabel(plot_column)
        plt.ylabel("Count")
        st.pyplot(plt)

    elif plot_type == "Histograms":
        plot_column = st.sidebar.selectbox("Choose a histogram:", [
            "Births in last five years",
            "Age of respondent at 1st birth",
            "Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)",
            "Hemoglobin level adjusted for altitude (g/dl - 1 decimal)"
        ])
        # Generate and display the selected histogram
        data[plot_column].plot(kind='hist', title=f'{plot_column}')
        plt.xlabel(plot_column)
        plt.ylabel("Frequency")
        st.pyplot(plt)

    elif plot_type == "Box Plots":
        plot_column = st.sidebar.selectbox("Choose a box plot:", [
            "Births in last five years",
            "Age of respondent at 1st birth",
            "Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)",
            "Hemoglobin level adjusted for altitude (g/dl - 1 decimal)"
        ])
        # Generate and display the selected box plot
        sns.boxplot(x=data[plot_column])
        plt.xlabel(plot_column)
        plt.title(f'Box Plot of {plot_column}')
        st.pyplot(plt)

    elif plot_type == "Correlation Matrix":
        # Generate and display the correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        st.pyplot(plt)

    elif plot_type == "Pairplot":
        # Display a pairplot of the dataset
        sns.pairplot(data)
        st.pyplot(plt)

    elif plot_type == "Scatter Plot":
        
        data.columns = data.columns.str.strip()
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
        if len(numeric_columns) < 2:
            st.write("Error: Not enough numeric columns for scatter plot.")
        else:
            x_col = st.sidebar.selectbox("X-axis:", numeric_columns)
            y_col = st.sidebar.selectbox("Y-axis:", numeric_columns)
        
        # Validate selected columns
            if x_col not in data.columns or y_col not in data.columns:
                st.write("Error: Invalid column names selected for scatter plot.")
            else:
                sns.scatterplot(x=data[x_col], y=data[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'Scatter Plot: {x_col} vs {y_col}')
                st.pyplot(plt)

    elif plot_type == "Bootstrap Confidence Intervals":
        # Display bootstrap confidence intervals for all numeric columns
        st.header("Bootstrap Confidence Intervals")
        numeric_columns = data.select_dtypes(include=np.number).columns
        for column in numeric_columns:
            bootstrap_results = bs.bootstrap(data[column].values, stat_func=bs_stats.mean, num_iterations=1000, alpha=0.05)
            mean_estimate = bootstrap_results.value
            lower_ci, upper_ci = bootstrap_results.lower_bound, bootstrap_results.upper_bound
            st.write(f"Confidence interval for mean of {column}: Estimate={mean_estimate}, CI=({lower_ci}, {upper_ci})")


# Regression Analysis Category
elif category == "Regression Analysis":
    st.header("Regression Analysis")

    # Sub-category for OLS Regression
    analysis_type = st.sidebar.selectbox("Select Analysis:", [
        "OLS Regression",
        "Linear Regression"
    ])

    if analysis_type == "OLS Regression":
        # OLS regression function with NaN checks
        def ols_regression_analysis():
            # Clean data before defining predictors and target
            clean_data(data)

            # Define predictor variables and target variable for OLS regression analysis
            X = data[['Hemoglobin level adjusted for altitude (g/dl - 1 decimal)', 'Age of respondent at 1st birth']]
            X = sm.add_constant(X)

            # Define the target variable
            y = data['Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)']

            # Check for NaNs or infinite values
            if X.isnull().any().any() or X.isin([np.inf, -np.inf]).any().any():
                st.write("Error: Predictor data contains NaNs or infinite values after cleaning.")
                return
            
            if y.isnull().any() or y.isin([np.inf, -np.inf]).any():
                st.write("Error: Target data contains NaNs or infinite values after cleaning.")
                return

            model = sm.OLS(y, X).fit()

            # Display the model summary
            st.write(model.summary())
            coefficients_with_ci = model.conf_int(alpha=0.05)  # 95% confidence interval
            coefficients_with_ci.columns = ['Lower CI', 'Upper CI']
            coefficients_with_ci['Estimate'] = model.params
            st.write("Regression coefficients with 95% confidence intervals:")
            st.write(coefficients_with_ci)

        ols_regression_analysis()

    elif analysis_type == "Linear Regression":
        # Define the dependent variable and independent variables
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

        def linear_regression_analysis():
            label_encoder = LabelEncoder()
            data[dependent_variable] = label_encoder.fit_transform(data[dependent_variable])

            for independent_variable in independent_variables:
                # Prepare data for the current independent variable
                X = data[[independent_variable]].values
                y = data[dependent_variable].values
                
                if isinstance(X[0][0], str):
                    encoder = LabelEncoder()
                    X = encoder.fit_transform(X)
                X = X.reshape(-1, 1)

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create and train the model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions and evaluate the model
                y_pred = model.predict(X_test)
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
                st.pyplot(plt)

                # Display regression metrics
                st.write(f"Independent Variable: {independent_variable}")
                st.write(f"Mean Squared Error: {mse}")
                st.write(f"R-squared (R2): {r2}")
                st.write(f"Coefficient: {model.coef_[0]}")
                st.write(f"Intercept: {model.intercept_}")

        linear_regression_analysis()


