Car Dataset Analysis and Modeling: Regression and Classification

This project aims to analyze a car dataset to evaluate fuel efficiency and build multiple predictive models. The project includes exploratory data analysis, Simple Linear Regression, Polynomial Regression, Logistic Regression, and methods to detect overfitting and outliers.

Dataset Information

The dataset consists of the following attributes:

	1.	mpg: Miles per gallon, a measure of fuel efficiency.
	2.	displacement: The cylinder volumes in cubic inches.
	3.	horsepower: Engine power.
	4.	weight: The weight of the car in pounds.
	5.	acceleration: The time in seconds to go from 0 to 60 mph.
	6.	origin: The region where the car was manufactured (USA, Japan, Europe).

Note: The dataset has already been cleaned and pre-processed for the purpose of this assignment.

Project Overview

Exercise 1: Exploratory Data Analysis

Exercise 1.1: Correlation Matrix



<img width="608" alt="test 2024-09-20 at 18 23 53" src="https://github.com/user-attachments/assets/3f8401a2-6689-4fc3-9e8c-f96717252995">


In this task, we compute and visualize the Pearson correlation matrix for the dataset, excluding the origin column. The correlation matrix helps us understand the relationships between variables.

The Pearson correlation measures the linear relationship between two variables. Correlation values range from -1 to 1, where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.

Result: The heatmap shows that acceleration has a very low correlation with mpg, making it an unlikely choice as the independent variable in Simple Linear Regression.

Exercise 1.2: Pairplot

We generate a pairplot to visualize the relationships between all features. The pairplot includes scatter plots for each pair of variables and shows the distributions of each feature based on the origin attribute.

Result: The relationship between mpg and horsepower is non-linear, so Polynomial Regression would be more suitable than Linear Regression for modeling this relationship.

Exercise 2: Linear and Polynomial Regression

Exercise 2.1: Splitting the Dataset

from sklearn.model_selection import train_test_split
X = df[['horsepower']]
y = df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15)

We split the dataset into training and testing sets using an 80:20 ratio. The horsepower feature is used as the independent variable, and mpg is the dependent variable.

Exercise 2.2: Simple Linear Regression

We implement Simple Linear Regression to predict mpg using the weight attribute. The Simple Linear Regression model follows the equation y = beta_0 + beta_1 * X, where beta_0 is the intercept (bias), beta_1 is the slope (coefficient), X is the input feature (weight), and y is the predicted output (mpg).

We evaluate the model using Mean Squared Error (MSE), which calculates the average squared difference between actual and predicted values.

Result: The testing MSE is reported, indicating the model’s performance.

Exercise 2.3: Polynomial Regression

We implement Polynomial Regression to predict mpg using the weight attribute. The model is fitted for degrees 2, 3, and 4. Polynomial Regression models the relationship between variables as a polynomial function, allowing for non-linear relationships.

We compute both training and testing MSE for each degree to evaluate the model’s performance and check for signs of overfitting.

Result: Overfitting is identified when the model performs well on the training set but poorly on the testing set for higher degrees.

Exercise 3: Logistic Regression

Exercise 3.1: Processing and Splitting the Dataset


<img width="393" alt="test 2024-09-20 at 18 24 46" src="https://github.com/user-attachments/assets/b8571db8-9f7b-4c30-815c-1d65ed8d278a">


We filter the dataset to include only cars from the USA and Japan. The dataset is then split into training and testing sets using an 80:20 ratio.

Exercise 3.2: Logistic Regression

We implement Logistic Regression to classify whether a car originated from the USA or Japan. Logistic Regression models the probability that a given input belongs to a particular class (USA or Japan). The model is evaluated using Precision, Recall, and the F1 Score.

Precision is the proportion of true positive predictions out of all positive predictions, and Recall is the proportion of true positives out of all actual positives.

Result: The testing precision and recall for cars from Japan and the USA are reported. If we were to classify cars between Japan and Europe, we expect lower performance because European cars share more features with Japanese cars than American cars do.

Exercise 4: Overfitting and Underfitting

Exercise 4.1: SSE and Variance Calculation

We calculate the Sum of Squared Errors (SSE) and variance for three sets of predictions. SSE measures the total deviation of predicted values from actual values, while variance measures how spread out the predictions are.

Exercise 4.2: Identifying Overfitting and Underfitting

Based on the SSE and variance values, we categorize the models as overfitting, underfitting, or a good fit.

Exercise 5: Outlier Detection

Exercise 5.1: Box Plot of Blood Pressure

We extract the BloodPressure attribute from the diabetes dataset and create a box plot to visualize outliers.

Exercise 5.2: Anomaly Detection Using One-Class SVM

We use the One-Class SVM algorithm for anomaly detection using the BMI and Insulin features from the diabetes dataset. The anomalies are visualized in a scatter plot.
