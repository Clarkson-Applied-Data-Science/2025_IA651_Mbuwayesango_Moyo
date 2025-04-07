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
c![image](https://github.com/user-attachments/assets/bad3b996-188f-48ff-a092-2fc060386101)


## Distribution of contineous variables 
![image](https://github.com/user-attachments/assets/d0a054fb-8944-4b15-9af2-1140b888006e)




