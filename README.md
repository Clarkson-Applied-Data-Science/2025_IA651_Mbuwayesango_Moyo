# 2025_IA651_Mbuwayesango_Moyo

# Predicting Parkinson’s Disease Progression Using Machine Learning

## Project overview 
Parkinson’s Disease is a progressive neurological disorder that affects movement, speech, and various motor functions. Early and accurate tracking of its progression is critical for effective treatment and patient care. In this project, we leverage machine learning techniques to analyze a dataset containing voice measurements and clinical attributes from individuals with Parkinson’s Disease.

The primary objective is to build predictive models that estimate the progression of the disease using biomedical voice features and patient demographics. Through exploratory data analysis, feature engineering, and model development, we aim to identify patterns and markers that are indicative of disease advancement. This work not only showcases the power of machine learning in medical diagnostics but also contributes toward the development of non-invasive tools for disease monitoring.

## Dataset
The dataset used in this project is centered on analyzing the progression of Parkinson’s Disease based on various clinical, functional, and lifestyle-related features. It is sourced from Kaggle - Parkinson’s Disease Progression Dataset, which provides anonymized health records of 500 patients living with Parkinson’s Disease.

The dataset includes the following fields:

Patient_ID: Unique identifier for each patient

Age: Patient’s age

Gender: Biological sex of the patient (M/F)

Years_Since_Diagnosis: Time (in years) since Parkinson’s diagnosis

UPDRS_Score: Unified Parkinson's Disease Rating Scale score

Tremor_Severity: Severity of tremors (scale: 0–5)

Motor_Function: Motor function impairment rating (scale: 0–5)

Speech_Difficulty: Degree of speech difficulty (scale: 0–5)

Balance_Problems: Level of balance issues experienced (scale: 0–5)

Medications: Primary medication prescribed (e.g., Levodopa, Amantadine, Ropinirole, Pramipexole)

Exercise_Level: Self-reported physical activity level (Low, Moderate, High)

Disease_Progression: Disease progression severity (target variable, scale: 1–3)

This dataset supports the development of machine learning models that can predict disease progression based on physical, behavioral, and treatment-related attributes. Such models can aid in early intervention, personalized care, and informed decision-making for both clinicians and patients.


## Process Overview
This project was an iterative process of exploration and refinement. We started with data cleaning and EDA to understand key patterns and feature relationships. Early modeling attempts using basic classifiers revealed the need for better preprocessing, so we incorporated feature scaling and PCA to improve performance and interpretability.

Along the way, we adjusted our approach based on insights—giving more weight to clinically relevant features like UPDRS_Score and Exercise_Level. These pivots helped us build a more effective model and strengthened our understanding of the data and its real-world implications.

## Exploratory Data Analysis (EDA)

The dataset consists of 500 observations and 11 relevant features (excluding the Patient_ID). The X variables include demographic and clinical attributes such as Age, Gender, Tremor_Severity, Motor_Function, Speech_Difficulty, Exercise_Level, and Medications. The target variable (Y) is Disease_Progression, which classifies patients into three severity levels: mild, moderate, and severe. This defines our problem as a multi-class classification task. With a strong feature-to-observation ratio, the dataset is well-suited for building and evaluating predictive models.


## Features
The dataset includes the following features (columns):
![image](https://github.com/user-attachments/assets/f199d598-1a65-4ead-b71f-75200eb9f043)

## Target Variable 
The target variable in this dataset is Disease_Progression, which classifies patients into three categories based on the severity of Parkinson’s Disease: 1 for Mild, 2 for Moderate, and 3 for Severe. It represents the progression stage of the disease and is used for multi-class classification. Predicting this variable accurately is essential for identifying patient needs, tailoring treatment plans, and enabling early interventions in managing Parkinson’s Disease.

## Feature distribution 
![image](https://github.com/user-attachments/assets/2ff82ca5-367c-4e16-aeda-40b7ea2cda5b)



## Distribution of continuous variables 
![image](https://github.com/user-attachments/assets/d0a054fb-8944-4b15-9af2-1140b888006e)

## Correlation Analysis
The correlation heatmap shows generally weak relationships between most numeric features. Years_Since_Diagnosis has a slight positive correlation with Motor_Function and Balance_Problems, while Tremor_Severity is slightly negatively correlated with Age. Notably, UPDRS_Score shows minimal correlation with Disease_Progression, suggesting that linear relationships alone may not fully explain progression severity. Overall, no strong multicollinearity was observed, and all features were retained for modeling.

![image](https://github.com/user-attachments/assets/ab5110a8-e969-45a1-9fb2-6698c36b3a4e)

## Feature Engineering 

To enhance model performance and interpretability, we performed several feature engineering steps:

Dropped non-informative columns: Removed Patient_ID as it does not carry predictive value.

Categorical Encoding: Converted categorical features like Gender, Medications, and Exercise_Level into numerical format using label encoding or one-hot encoding depending on the model.

Feature Scaling: Applied standardization (e.g., StandardScaler) to continuous variables such as Age and UPDRS_Score to ensure uniform scale across features.

Dimensionality Reduction: Performed PCA (Principal Component Analysis) to reduce noise and better visualize feature structure during EDA.

Correlation Check: Examined feature correlation to identify redundant variables, though no features were removed due to weak correlations.

Target Encoding Insight: Ensured the target variable Disease_Progression was treated as a categorical class for classification models.

## Principal Component Analysis (PCA)
PCA was performed to reduce the dimensionality of the dataset and to visualize the variance explained by each component. The following chart shows the explained variance by each principal component:

![image](https://github.com/user-attachments/assets/554d7ce1-ccc2-4737-b251-2df19d81806d)

![image](https://github.com/user-attachments/assets/92448d00-b3de-427a-ac60-cef678ce8120)

![image](https://github.com/user-attachments/assets/d363a1b7-43b4-4041-a70f-fa4a83a694c0)

The PCA results indicated that approximately 90% of the variance in the dataset is explained by the first 9 principal components, suggesting that these components capture most of the meaningful information. This dimensionality reduction helped uncover the underlying structure of the data, reduced noise, and provided insights into which features contributed most to patient variability. These insights were valuable in guiding feature selection and improving the efficiency of our modeling process.

## Model Fitting
To evaluate the performance of various classification models, we split the dataset into training and testing sets using an 80/20 stratified split. Stratification ensured that each class in the target variable Disease_Progression was proportionally represented in both sets. This split ratio was chosen to maintain a sufficient number of samples for training while keeping a separate holdout set for unbiased evaluation.

We performed standard preprocessing by encoding categorical variables and scaling numeric features using StandardScaler. For model selection and tuning, we applied GridSearchCV to optimize hyperparameters for Logistic Regression and Support Vector Machine (SVC). We also trained Decision Tree and Random Forest classifiers using default or manually specified parameters. Each model was evaluated using standard classification metrics including accuracy, precision, recall, F1-score, and a confusion matrix.

Since this dataset is not time-based, there is no risk of data leakage due to temporal ordering. However, care was taken to ensure that no information from the test set was used during training or preprocessing to avoid any form of leakage.

The motivation behind testing multiple models and performing grid search was to compare both linear and non-linear approaches, and to identify the best-performing model for predicting Parkinson’s Disease progression based on clinical and lifestyle features.

## Confusion matrices for the models
![confusion matrix for logistic regression](https://github.com/user-attachments/assets/85c1c610-fc36-45f5-9209-b247760d7ae8)
![confusion matrix -descision tree](https://github.com/user-attachments/assets/e3ca896a-4e3e-41aa-8065-76736c29cd4c)
![confusion matrix -SVC](https://github.com/user-attachments/assets/729958ac-edd4-4c8c-9a12-a1afcb5f67da)
![random forrest - confusion matrix](https://github.com/user-attachments/assets/4a9d4c19-8c39-4d18-a4bb-b0a80f03cf36)
![Confusion matrix ensamble](https://github.com/user-attachments/assets/92a470d2-0a37-4bc9-bb88-4a2764eeaf64)
![ensamble confusion matrix](https://github.com/user-attachments/assets/46517333-19ca-40e8-9dcf-006a243ceb79)
![ensamble method](https://github.com/user-attachments/assets/53abbdc6-dc43-4195-a368-a665e9822223)









## ROC-AUC Curves
![image](https://github.com/user-attachments/assets/796ce197-0e2e-4d0b-b085-28c429f7c2d4)


## Consolidated Model Metrics

![image](https://github.com/user-attachments/assets/e621225c-0208-4856-b9cf-7660ab1effa7)




















