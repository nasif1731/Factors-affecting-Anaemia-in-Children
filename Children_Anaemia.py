import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
data=pd.read_csv("children_anaemia.csv")


# List of qualitative attributes
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

# List of quantitative attributes
quantitative_attributes = [
    'Births in last five years',
    'Age of respondent at 1st birth',
    'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)',
    'Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'
]

 #Define a mapping from numeric values to textual representations
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

# Checking for missing values percentage in each column
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

# Display data types of columns
print(data.dtypes)

# Check for non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# Print non-numeric columns if exist
if len(non_numeric_columns) > 0:
    print("\nNon-numeric columns:")
    for col in non_numeric_columns:
        print(col)
else:
    print("\nAll columns are numeric.")

# Visualize 'Age in 5-year groups' with a bar chart
data['Age in 5-year groups'].value_counts().plot(kind='bar', title='Age in 5-year groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart shows the distribution of children across different age groups, providing insight into the age composition of the dataset.
# That how many of the children are suffering with the disease within the age groups of five years.By graph we can see that the graph is positively skewed which means the anaemia patients are more in the first group which is 25-29 age group

# Visualize 'Type of place of residence' with a bar chart
data['Type of place of residence'].value_counts().plot(kind='bar', title='Type of place of residence')
plt.xlabel('Place of Residence')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart illustrates the distribution of children based on their place of residence, indicating whether they reside in urban or rural areas.
# That how many of the children are affected through their residence area.By graph we can see that mostly are from rural areas.

# Visualize 'Highest educational level' with a bar chart
data['Highest educational level'].value_counts().plot(kind='bar', title='Highest educational level')
plt.xlabel('Educational Level')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart displays the distribution of children's caregivers' highest educational attainment, providing insights into the educational background of caregivers in the dataset.
# which states that the patients are illiterate mostly.

# Visualize 'Wealth index combined' with a bar chart
data['Wealth index combined'].value_counts().plot(kind='bar', title='Wealth index combined')
plt.xlabel('Wealth Index')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart shows the distribution of households across different wealth indices, offering insights into the socioeconomic status of families in the dataset.
#It shows most of the children are from lower and lower middle and middle class

# Visualize 'Current marital status' with a bar chart
data['Current marital status'].value_counts().plot(kind='bar', title='Current marital status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart illustrates the distribution of caregivers' current marital status, providing information about the marital composition of caregivers in the dataset.
#This shows that mostly are married

# Visualize 'Currently residing with husband/partner' with a bar chart
data['Currently residing with husband/partner'].value_counts().plot(kind='bar', title='Currently residing with husband/partner')
plt.xlabel('Residing with Husband/Partner')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart displays whether caregivers are currently residing with their husband or partner, indicating living arrangements and relationships within households.
# the caregivers are mostly residing with partner
# Visualize 'When child put to breast' with a bar chart
data['When child put to breast'].value_counts().plot(kind='bar', title='When child put to breast')
plt.xlabel('Breastfeeding Initiation Time')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart shows the timing of breastfeeding initiation for children in the dataset, offering insights into breastfeeding practices among caregivers.
# The children were put into breast feeding immediately after birth.
# Visualize 'Had fever in last two weeks' with a bar chart
data['Had fever in last two weeks'].value_counts().plot(kind='bar', title='Had fever in last two weeks')
plt.xlabel('Fever Status')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart illustrates whether children had a fever in the last two weeks, providing information about recent health conditions among children in the dataset.
#This shows that children suffering from anaemia were not having any kind of fever in last two weeks.
# Visualize 'Anemia level' with a bar chart
data['Anemia level'].value_counts().plot(kind='bar', title='Anemia level')
plt.xlabel('Anemia Status')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart displays the distribution of children's anemia levels, indicating the prevalence of anemia among children in the dataset.
#Most children records don't show any anaemia status.

# Visualize 'Have mosquito bed net for sleeping (from household questionnaire)' with a bar chart
data['Have mosquito bed net for sleeping (from household questionnaire)'].value_counts().plot(kind='bar', title='Have mosquito bed net for sleeping')
plt.xlabel('Mosquito Bed Net Availability')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart illustrates whether households have mosquito bed nets for sleeping, providing insights into malaria prevention measures among households.
#Mostly households were having bed-nets a/c to graph
# Visualize 'Smokes cigarettes' with a bar chart
data['Smokes cigarettes'].value_counts().plot(kind='bar', title='Smokes cigarettes')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart displays the distribution of caregivers' smoking status, indicating the prevalence of smoking among caregivers in the dataset.
#the parents/caregivers mostly were not taking any ciggaretes.
# Visualize 'Taking iron pills, sprinkles or syrup' with a bar chart
data['Taking iron pills, sprinkles or syrup'].value_counts().plot(kind='bar', title='Taking iron pills, sprinkles or syrup')
plt.xlabel('Iron Supplementation Status')
plt.ylabel('Count')
plt.show()
# Conclusion: This bar chart illustrates whether children are taking iron supplements, providing insights into nutritional interventions for anemia prevention among children in the dataset.
#the children mostly were not taking any pills,sprinkles,syrup.

# Visualize distribution of 'Births in last five years'
data['Births in last five years'].plot(kind='hist', title='Distribution of Births in Last Five Years')
plt.xlabel('Births in Last Five Years')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This histogram illustrates the frequency distribution of the number of births in the last five years among respondents, showing the spread and central tendency of the data.
#Mostly births have been taken place in last two years
# Visualize distribution of 'Age of respondent at 1st birth'
data['Age of respondent at 1st birth'].plot(kind='hist', title='Distribution of Age of Respondent at 1st Birth')
plt.xlabel('Age of Respondent at 1st Birth')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This histogram displays the frequency distribution of the age of respondents at their first childbirth, providing insights into the age distribution of first-time mothers.
#Age of respondent is mostly between 15 to 20
# Visualize distribution of 'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)'
data['Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)'].plot(kind='hist', title='Distribution of Hemoglobin Level Adjusted for Altitude and Smoking')
plt.xlabel('Hemoglobin Level Adjusted for Altitude and Smoking (g/dl - 1 decimal)')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This histogram illustrates the frequency distribution of hemoglobin levels adjusted for altitude and smoking among children, providing insights into the distribution of nutritional and health indicators.
# Most haemogolobin level are not given but from given data the haemoglobin level peak is at approx 125
# Visualize distribution of 'Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'
data['Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'].plot(kind='hist', title='Distribution of Hemoglobin Level Adjusted for Altitude')
plt.xlabel('Hemoglobin Level Adjusted for Altitude (g/dl - 1 decimal)')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This histogram shows the frequency distribution of hemoglobin levels adjusted for altitude among children, offering insights into the distribution of nutritional and health indicators.
# Most haemoglobin level is from 100 to 125


# Visualize 'Births in last five years' with a box plot
sns.boxplot(x=data['Births in last five years'])
plt.xlabel('Births in last five years')
plt.title('Box Plot of Births in Last Five Years')
plt.show()
# Conclusion: This box plot visualizes the distribution of the number of births in the last five years, showing the median, quartiles, and outliers.
#The maximum data value is at the last 3 years but the lower and upper quartile values are 1 and 2 respectively.Their are three outliers which are residing in the last four,five and six years in dataset.
# Visualize 'Age of respondent at 1st birth' with a box plot
sns.boxplot(x=data['Age of respondent at 1st birth'])
plt.xlabel('Age of respondent at 1st birth')
plt.title('Box Plot of Age of Respondent at 1st Birth')
plt.show()
# Conclusion: This box plot displays the distribution of the age of respondents at their first childbirth, showing the median, quartiles, and outliers.
#The maximum datavalues is some where above 30 but between 30-35.The average value lies approx 19.The outliers range from approx 32 till above 45
# Visualize 'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)' with a box plot
sns.boxplot(x=data['Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)'])
plt.xlabel('Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)')
plt.title('Box Plot of Hemoglobin Level Adjusted for Altitude and Smoking')
plt.show()
# Conclusion: This box plot illustrates the distribution of hemoglobin levels adjusted for altitude and smoking among children, showing the median, quartiles, and outliers.
#The maximum value of haemoglobin lies above 200 and the lower quartile and upper quartile is 0 and 110 respectively.
# Visualize 'Hemoglobin level adjusted for altitude (g/dl - 1 decimal)' with a box plot
sns.boxplot(x=data['Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'])
plt.xlabel('Hemoglobin level adjusted for altitude (g/dl - 1 decimal)')
plt.title('Box Plot of Hemoglobin Level Adjusted for Altitude')
plt.show()
# Conclusion: This box plot visualizes the distribution of hemoglobin levels adjusted for altitude among children, showing the median, quartiles, and outliers.
# The maximum value of haemoglobin lies somewhere below 175 and the lower and upper quartile are 0 and  approx 87.5


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



# Visualize pairplot with labels
sns.pairplot(data)
plt.suptitle('Pairplot of All Features', y=2)
plt.tight_layout()
plt.show()
#The diagonal plots represent the distribution of each variable individually.
# They help in understanding the distribution of each feature and identifying potential outliers.

# Visualize scatter plot for Age vs Births
sns.scatterplot(x=data['Age in 5-year groups'], y=data['Births in last five years'])
plt.xlabel('Age in 5-year groups')
plt.ylabel('Births in last five years')
plt.title('Scatter Plot: Age vs Births')
plt.show()
#The scatter plot shows the relationship between the age of respondents grouped in 5-year intervals and the number of births they have had in the last five years.

# Visualize distribution of 'Births in last five years'
sns.histplot(data=data['Births in last five years'], kde=True)
plt.title('Distribution of Births in Last Five Years')
plt.xlabel('Births in Last Five Years')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This distribution chart illustrates the frequency distribution of the number of births in the last five years among respondents, showing the spread and central tendency of the data.

# Visualize distribution of 'Age of respondent at 1st birth'
sns.histplot(data=data['Age of respondent at 1st birth'], kde=True)
plt.title('Distribution of Age of Respondent at 1st Birth')
plt.xlabel('Age of Respondent at 1st Birth')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This distribution chart displays the frequency distribution of the age of respondents at their first childbirth, providing insights into the age distribution of first-time mothers.

# Visualize distribution of 'Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)'
sns.histplot(data=data['Hemoglobin level adjusted for altitude and smoking (g/dl - 1 decimal)'], kde=True)
plt.title('Distribution of Hemoglobin Level Adjusted for Altitude and Smoking')
plt.xlabel('Hemoglobin Level Adjusted for Altitude and Smoking (g/dl - 1 decimal)')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This distribution chart illustrates the frequency distribution of hemoglobin levels adjusted for altitude and smoking among children, providing insights into the distribution of nutritional and health indicators.
#The data is following the normal distribution of haemoglobin levels a/c to graph
# Visualize distribution of 'Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'
sns.histplot(data=data['Hemoglobin level adjusted for altitude (g/dl - 1 decimal)'], kde=True)
plt.title('Distribution of Hemoglobin Level Adjusted for Altitude')
plt.xlabel('Hemoglobin Level Adjusted for Altitude (g/dl - 1 decimal)')
plt.ylabel('Frequency')
plt.show()
# Conclusion: This distribution chart shows the frequency distribution of hemoglobin levels adjusted for altitude among children, offering insights into the distribution of nutritional and health indicators.
#The data is following the normal distribution of haemoglobin levels a/c to graph

# Print the dataset to ensure it's loaded correctly
print(data.to_string())

