
# Wine Quality Dataset Analysis and Classification Model
## Introduction

This project involves analyzing the Wine Quality dataset and building a Random Forest Classifier model to predict the quality of wine. The dataset contains various chemical properties of red wine, and the objective is to use these features to classify the quality of the wine on a scale from 1 to 6.

## Dataset Description 

Loading the Data
We start by loading the dataset using pandas:

```python
import pandas as pd
df = pd.read_csv('/content/WineQT.csv')
```
## Dataset Overview

A quick overview of the dataset:

```python

df.describe()
```
## Feature Descriptions:
1. **Fixed Acidity**: Non-volatile acids that remain in the wine during fermentation.
2. **Volatile Acidity**: Represents acetic acid content, which can give wine an undesirable vinegar flavor.
3. **Citric Acid:** Provides freshness to wines and can contribute to flavor.
4. **Residual Sugar:** The amount of sugar remaining after fermentation.
5. **Chlorides:** The amount of salt in the wine.
6. **Free Sulfur Dioxide:** SO₂ protects wine from oxidation and microbial growth.
7. **Total Sulfur Dioxide:** Sum of free and bound SO₂.
8. **Density:** Density of the wine, closely related to alcohol and sugar content.
9. **pH:** Measures the acidity or alkalinity of wine.
10. **Sulphates:** A wine preservative and antioxidant.
11. **Alcohol:** Alcohol content of the wine.
12. **Quality:** Wine quality score, likely on a scale from 1 to 10.
13. **Id:** An identifier for each wine sample.
## Data Structure:
``` python
df.info()
```
The dataset contains no missing values, and the categorical quality feature needs to be encoded for model training.
## Exploratory Data Analysis (EDA)
### Correlation Heatmap
Compute and visualize the correlation matrix between features:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Features')
plt.show()
```
## Visualizations
### Count Plot of Wine Quality
```python
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='quality')
plt.title("Count plot of wine quality")
plt.show()
```
### Box Plots for Various Features
Box plot of alcohol content by wine quality:

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title('Alcohol Content by Wine Quality')
plt.show()
```
### Box plot of sulphates content by wine quality:

```python

plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='sulphates', data=df)
plt.title('Sulphates Content by Wine Quality')
plt.show()
```
### Box plot of citric acid content by wine quality:

```python

plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='citric acid', data=df)
plt.title('Citric Acid Content by Wine Quality')
plt.show()
```
### Box plot of fixed acidity content by wine quality:

```python

plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='fixed acidity', data=df)
plt.title('Fixed Acidity Content by Wine Quality')
plt.show()
```
### Scatter Plots
Scatter plot to show the relationship between alcohol content and quality:

```python

plt.figure(figsize=(10, 6))
sns.scatterplot(x='alcohol', y='quality', data=df, hue='quality', palette='coolwarm', s=100)
plt.title('Relationship Between Alcohol Content and Quality')
plt.show()
```
### Scatter plot to show the relationship between sulphates and quality:

```python

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sulphates', y='quality', data=df, hue='quality', palette='coolwarm', s=100)
plt.title('Relationship Between Sulphates Content and Quality')
plt.show()
```
### Scatter plot to show the relationship between citric acid and quality:

```python

plt.figure(figsize=(10, 6))
sns.scatterplot(x='citric acid', y='quality', data=df, hue='quality', palette='coolwarm', s=100)
plt.title('Relationship Between Citric Acid Content and Quality')
plt.show()
```
### Scatter plot to show the relationship between volatile acidity and quality:

```python

plt.figure(figsize=(10, 6))
sns.scatterplot(x='volatile acidity', y='quality', data=df, hue='quality', palette='coolwarm', s=100)
plt.title('Relationship Between Volatile Acidity Content and Quality')
plt.show()
```
## Data Preprocessing
### Encoding Quality
Map wine quality to a range from 1 to 6:

```python

map_dict = {3:1, 4:2, 5:3, 6:4, 7:5, 8:6}
df['quality'] = df['quality'].map(map_dict)
```
### Adding a Good Quality Column
```python

df['goodquality'] = [1 if x >= 3 else 0 for x in df['quality']]
```
### Renaming Columns and Dropping Irrelevant Features
```python

df.rename(columns={'free sulfur dioxide':'free SO2','total sulfur dioxide':'total SO2'}, inplace=True)
df = df.drop(['Id'], axis=1)
```
## Train-Test Split
We split the data into training and testing sets:

```python

from sklearn.model_selection import train_test_split

X = df.drop(columns=['quality'])
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```
## Model Training and Evaluation
### Random Forest Classifier
Initialize and fit the model:

```python

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
## Predictions and Accuracy
Make predictions and calculate accuracy:

```python

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```
## Classification Report
Get precision, recall, and F1-score for each class:

```python 

from sklearn.metrics import classification_report

print("Classification Report:\n", classification_report(y_test, y_pred))
```
## Confusion Matrix
Plot the confusion matrix:

```python

from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
## Results and Insights
Accuracy: The model achieved an accuracy of 68.80% on the test set.
Feature Importance: Alcohol, sulphates, citric acid, and fixed acidity are the most significant features affecting wine quality.
Quality Distribution: The dataset shows a distribution of wine quality with a higher frequency of average quality wines.
## Conclusion
This project demonstrates the application of basic machine learning techniques, including exploratory data analysis (EDA), data preprocessing, and model evaluation using the Random Forest Classifier. The model provides a robust prediction of wine quality based on chemical properties.

## Requirements
Python 3.x
Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn
```bash

pip install numpy pandas matplotlib seaborn scikit-learn
```
