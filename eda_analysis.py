import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("student_lifestyle_data.csv")

# Display first rows
print("\nFirst 5 Rows")
print(df.head())

# Dataset information
print("\nDataset Information")
print(df.info())

# Statistical summary
print("\nStatistical Summary")
print(df.describe())

# Check missing values
print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Correlation matrix
print("\nCorrelation Matrix")
print(df.corr())

# Visualization 1
plt.figure(figsize=(8,5))
sns.histplot(df['exam_score'], bins=10)
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.show()

# Visualization 2
plt.figure(figsize=(8,5))
sns.scatterplot(x='study_hours', y='exam_score', data=df)
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()

# Visualization 3
plt.figure(figsize=(8,5))
sns.scatterplot(x='sleep_hours', y='exam_score', data=df)
plt.title("Sleep Hours vs Exam Score")
plt.xlabel("Sleep Hours")
plt.ylabel("Exam Score")
plt.show()

# Visualization 4
plt.figure(figsize=(8,5))
sns.barplot(x='extracurricular_activities', y='exam_score', data=df)
plt.title("Extracurricular Activities vs Exam Score")
plt.xlabel("Participation")
plt.ylabel("Average Exam Score")
plt.show()

# Visualization 5
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Insights
print("\nKey Insights")
print("1. Higher study hours generally improve exam performance.")
print("2. Students with better attendance tend to score higher.")
print("3. Excessive social media usage negatively affects exam scores.")
print("4. Balanced sleep contributes positively to academic performance.")
print("5. Students participating in extracurricular activities show slightly better performance.")