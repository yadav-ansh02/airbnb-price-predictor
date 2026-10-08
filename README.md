# Airbnb Price Predictor

An end-to-end machine learning project that predicts the **price of Airbnb listings** based on property type, accommodates, cleaning fee, city, review score rating, and other relevant features.

The project covers data cleaning, exploratory data analysis, feature engineering, machine learning model development, model evaluation, and prediction.

---

## Table of Contents

* [Overview](#overview)
* [Business Problem](#business-problem)
* [Dataset](#dataset)
* [Tools & Technologies](#tools--technologies)
* [Project Structure](#project-structure)
* [Data Cleaning & Preparation](#data-cleaning--preparation)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Feature Engineering](#feature-engineering)
* [Model Training & Evaluation](#model-training--evaluation)
* [Model Performance](#model-performance)
* [How to Run](#how-to-run)
* [Future Improvements](#future-improvements)
* [Author](#author)

---

## Overview

Pricing an Airbnb property appropriately depends on multiple factors such as property type, accommodates, cleaning fee, city, review score rating,number of bedrooms,etc

This project uses machine learning to estimate the price of an Airbnb listing based on the information available about the property.

### Project Workflow

```text
Raw Airbnb Data
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Feature Selection
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Comparison
       ↓
Model Evaluation
       ↓
Price Prediction
```

---

## Business Problem

Airbnb hosts need to set competitive prices that attract customers while maximizing their potential revenue.

Setting the price too high may reduce bookings, while setting it too low may result in lost revenue.

The objective of this project is to develop a machine learning model that can estimate the price of an Airbnb listing based on its characteristics.

### Potential Business Applications

The model can help:

* Airbnb hosts estimate a competitive listing price
* Property managers make data-driven pricing decisions
* Identify factors that influence Airbnb prices
* Analyze pricing patterns across different locations
* Support dynamic or data-driven pricing strategies

---

## Dataset

The dataset contains information about Airbnb listings and their corresponding characteristics.

Depending on the dataset, relevant features may include:

* Listing location
* Room type
* Property type
* Number of bedrooms
* Number of beds
* Number of reviews
* Review scores
* Availability
* Minimum nights
* Host-related information
* Listing price

**Dataset:** `[Airbnb_price.csv]`

---

## Tools & Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **VS Code**
* **Git & GitHub**

### Machine Learning

* Regression
* Feature Engineering
* Model Comparison
* Cross-Validation
* Hyperparameter Tuning
* Model Evaluation

### Models

The following regression algorithms were evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

---

## Project Structure

```text
airbnb-price-predictor/
│
├── data/
│
├── images/
│
├── notebooks/
│
├── models/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```


---

## Data Cleaning & Preparation

The raw Airbnb dataset was cleaned and prepared before model training.

The preprocessing steps included:

* Removing unnecessary columns
* Handling missing values
* Removing duplicate records
* Identifying outliers
* Correcting inconsistent data types
* Cleaning numerical features
* Cleaning categorical features
* Checking for invalid values
* Preparing categorical variables for machine learning
* Preparing numerical features for modeling

The cleaned dataset was then used for exploratory analysis and model development.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the relationship between Airbnb listing characteristics and price.

### Analysis Performed

* Distribution of Airbnb prices
* Relationship between reviews and price
* Relationship between availability and price
* Correlation analysis between numerical variables
* Identification of potential outliers

### Key Insights

Some factors that can influence Airbnb listing prices include:

* Location
* Room type
* Property characteristics
* Number of bedrooms
* Number of bathrooms
* Review-related features
* number of people it accommodates
* Minimum stay requirements


---

## Feature Engineering

Feature engineering was performed to improve the quality of the input data and make it more suitable for machine learning.

The process included:

* Selecting relevant predictive features
* Encoding categorical variables
* Processing numerical variables
* Removing highly irrelevant features
* Creating additional features where appropriate
* Preparing the final feature matrix for model training

---

## Model Training & Evaluation

The cleaned and engineered dataset was divided into training and testing datasets.

Multiple regression algorithms were trained and compared to determine which model performed best on the price prediction task.

### Evaluation Metrics

The models were evaluated using:

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

These metrics provide different perspectives on the model's prediction performance.

---

## Model Performance

The performance of the evaluated models can be presented as follows:

| Model                   |    MAE |   RMSE | R² Score |
| ----------------------- | -----: | -----: | -------: |
| Linear Regression       |   0.39 |    0.5 |     0.43 |
| Decision Tree Regressor |   0.39 |   0.49 |     0.47 |
| Random Forest Regressor |**0.37** |**0.48** |**0.51** |

### Best Performing Model

**`Random Forest`** achieved the best overall performance based on **`[R²]`**.

The selected model was then used for predicting Airbnb listing prices.

---

## Prediction

The final trained model can be used to estimate the price of an Airbnb listing based on its input features.

Example workflow:

```text
Listing Information
       ↓
Data Preprocessing
       ↓
Feature Transformation
       ↓
Trained Regression Model
       ↓
Predicted Airbnb Price
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yadav-ansh02/airbnb-price-predictor.git
```

### 2. Navigate to the Project Directory

```bash
cd airbnb-price-predictor
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Future Improvements

Possible improvements include:

* Deploy the model as a web application
* Implement advanced hyperparameter tuning
* Experiment with Gradient Boosting models
* Experiment with XGBoost, LightGBM, or CatBoost
* Add location-based features
* Implement advanced outlier detection
* Add explainable AI techniques such as SHAP
* Develop a dynamic pricing recommendation system
* Integrate real-time Airbnb listing data where legally and technically appropriate

---

## Author

**Ansh Yadav**

* GitHub: [github.com/yadav-ansh02](https://github.com/yadav-ansh02)
* LinkedIn: [linkedin.com/in/ansh-yadav-4a7b04390](https://www.linkedin.com/in/ansh-yadav-4a7b04390)
* Email: [yadav.ansh.0224@gmail.com](mailto:yadav.ansh.0224@gmail.com)
    