![](UTA-DataScience-Logo.png)

# Prediction of Accident Severity

* **One Sentence Summary** This repository holds an attempt to predict the severity of a car accident through machine learning models, using data from "US Accidents (2016 - 2023)" Kaggle data set https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents.

## Overview

The goal of this project is to predict the severity of car accidents on a scale of 1 to 4 using a large-scale dataset of real U.S. traffic incidents. The problem is framed as a multiclass classification task, where features such as weather conditions, time of day, road characteristics, and geographic location are used to determine how severe an accident will be.
The approach involved several stages of data preparation and modeling. Because the original dataset contains nearly 7.7 million records, it was first reduced to a 2-million-row random sample to make computation feasible. The data was then cleaned by removing irrelevant or high-cardinality features, handling missing values through median imputation for numerical columns and "Unknown" fill for categorical ones, and applying log transformations to skewed features like visibility and distance. Time-based features (hour, day of week, and month) were extracted from timestamps, and categorical variables with low cardinality were encoded using one-hot encoding. The dataset was split into training (70%), validation (15%), and test (15%) sets using stratified sampling to preserve the distribution of severity classes. Two models were trained and compared: Logistic Regression as a linear baseline and Random Forest as a nonlinear model. Class weights were applied to both models to address the significant imbalance in the dataset.
The Random Forest model slightly outperformed Logistic Regression, achieving an overall test accuracy of 47.1% compared to 46.3%, with improved recall across minority severity classes. While both models struggled with precision due to class imbalance, applying balanced class weights meaningfully improved detection of rare but important high-severity accidents.

## Summary of Workdone

### Data

* Data:
  * Type: CSV file of structured accident records.
    * Input: 46 features per accident (weather, location, road conditions, timestamps).
    * Output: Severity label (integer 1–4).
  * Size: Full dataset is 7,727,394 rows. A random 2,000,000-row sample was used for this project.
  * Instances:
    * Training: 1,400,000
    * Validation: 300,000
    * Test: 300,000

#### Preprocessing / Clean up
Several preprocessing steps were applied to prepare the data for modeling:
* Feature removal: Dropped non-predictive or excessively high-cardinality columns including ID, Description, Street, Zipcode, Country, End_Lat, End_Lng, and timestamp columns after feature extraction.
* Memory optimization: Downcast float64 to float32 and int64 to int32 to reduce memory usage and prevent kernel crashes.
* Missing values: Numerical columns were filled with their median; categorical columns were filled with "Unknown". Wind_Chill(F) had the highest missingness (~517K rows).
* Feature engineering: Extracted Hour, DayOfWeek, and Month from Start_Time.
* Log transformation: Applied log1p to Visibility(mi) and Distance(mi) to address right skew.
* Encoding: One-hot encoded low-cardinality categorical features (e.g., Source, State, Timezone, Sunrise_Sunset, Wind_Direction, Weather_Condition) with drop_first=True.
* Scaling: Applied StandardScaler to all features for Logistic Regression.
  
#### Data Visualization
<img width="892" height="569" alt="image" src="https://github.com/user-attachments/assets/5d1c9e25-e69a-4ba7-b4b3-424643327902" />
The bar chart shows accident counts broken down by severity for the ten most common weather conditions. "Fair" weather dominates by a wide margin, accounting for the vast majority of accidents across all severity levels, a reminder that most accidents happen in everyday driving conditions, not extreme weather. Conditions like Mostly Cloudy, Cloudy, Partly Cloudy, and Clear also show substantial counts, all with a similar pattern where Severity 2 makes up the bulk of incidents. Notably, Severity 4 (the most serious) is barely visible across all conditions, reflecting its rarity in the dataset.

<img width="722" height="558" alt="image" src="https://github.com/user-attachments/assets/66cda4f5-51af-43bd-b069-f95a780b9c88" />
The scatter plot maps accident locations across the contiguous United States, with each point colored by severity, blue representing lower-severity accidents (1–2) and red representing higher-severity ones (3–4). Accidents are concentrated most densely along the West Coast, the Northeast corridor, and the Southeast, which aligns with the higher population and traffic density in those regions. Higher-severity accidents (warmer-colored points) appear scattered throughout the country without a strong geographic cluster, suggesting that severity is not strongly tied to location alone but rather to situational factors at the time of the accident.

### Problem Formulation

* Define:
  * Input: 120 features derived from weather conditions (temperature, humidity, wind speed, visibility, precipitation), road characteristics (junction, traffic signal, crossing, etc.), time features (hour, day of week, month), and geographic/source metadata. 
  * Output: Accident severity class, one of 1, 2, 3, 4.
  * Models
    * Logistic Regression — Used as a linear baseline. Trained with class_weight="balanced" and max_iter=100. Note: the model hit the iteration limit and did not fully converge, suggesting the feature space may benefit from further scaling or dimensionality reduction.
    * Random Forest — Used as a nonlinear model to capture complex interactions. Configured with n_estimators=50, max_depth=10, class_weight="balanced", and n_jobs=-1 for parallel training.

### Training

* Describe the training:
  * Logistic Regression training was fast but did not converge within 100 iterations, indicating the problem is non-trivially linear.
  * Random Forest with 50 trees and max depth 10 completed in reasonable time; the depth cap helped manage memory and training time.
  * No explicit early stopping was used; model selection was based on validation accuracy after a single training run.
  * One notable difficulty was kernel crashes due to memory usage on the full dataset, which was resolved by downsampling and applying memory optimization (dtype downcasting).

### Performance Comparison
Key metric: Overall accuracy and per-class F1-score (weighted and macro averages), given the class imbalance.
<img width="799" height="159" alt="image" src="https://github.com/user-attachments/assets/69fb5e85-6100-4882-b6f6-ac3f846cd7a2" />
As you can see Random Forest had the best results overall

<img width="694" height="581" alt="image" src="https://github.com/user-attachments/assets/9a1755b5-d4e2-4286-96aa-d9dccbf16202" />
A confusion matrix heatmap was generated for the Random Forest model on the test set, providing a visual breakdown of correct and incorrect predictions per class.

### Conclusions
This project demonstrated that accident severity can be partially predicted from environmental, temporal, and road-related features, though the task is genuinely difficult due to the heavy class imbalance in the dataset. Nearly 80% of accidents are labeled Severity 2, which creates a strong prior that simpler models tend to exploit. Random Forest outperformed Logistic Regression across most metrics, confirming that nonlinear relationships between features matter in this problem. Applying balanced class weights was an important design choice, without it, both models would have collapsed to predicting the majority class nearly exclusively. Even with weighting, precision on the rare classes (1 and 4) remained very low, highlighting that the current feature set, while informative, does not fully separate these severity levels.

### Future Work

* Hyperparameter tuning on the Random Forest, in particular increasing n_estimators and exploring different max_depth values would likely yield additional gains.
* More powerful models such as Gradient Boosting (XGBoost, LightGBM) or stacked ensembles are natural next steps
* Addressing class imbalance more aggressively through oversampling techniques like SMOTE, rather than only class weighting, could improve recall-precision tradeoffs on the minority classes

## How to Reproduce Results

### Software Setup
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

### Data
Download the dataset from Kaggle: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
  
### Training
Open and run: https://github.com/maryamirfan747/ProjectTempate/blob/main/Final_Project-Version1.ipynb from top to bottom. The notebook handles all preprocessing, training, and evaluation in a single file. A machine with at least 8–16 GB of RAM is recommended due to the size of the dataset.
#### Performance Evaluation
Model performance metrics (accuracy, classification report, confusion matrix) are printed and visualized inline within the notebook at the end of the Machine Learning section.


## Citations

* Kaggle Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset." arXiv, 2019. https://arxiv.org/abs/1906.05409 







