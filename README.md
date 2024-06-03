# Hotel Booking Cancellation Prediction with R


## Table of Contents
1. [Project Overview](#project-overview)
2. [File Descriptions](#file-descriptions)
3. [How to Run](#how-to-run)
4. [Steps](#steps)
5. [Results](#results)
6. [Acknowledgements](#acknowledgements)

<a name="project-overview"></a>

## Overview
This project aims to understand the cancellation patterns for hotel bookings and predict the likelihood of cancellations using machine learning techniques. The dataset used in this project is from hotel booking records, and various steps such as data cleaning, exploratory data analysis (EDA), data transformation, clustering, classification, and prediction evaluation are performed.


<a name="file-descriptions"></a>

## File Descriptions
- `data/hotel_bookings.csv`: The invaluable dataset that fuels our analysis, containing a wealth of information on hotel bookings.
- `project-hotel-booking.R:`: This comprehensive R script encapsulates the entire journey of our analysis, from initial data exploration to model training and evaluation. File containing the code for data cleaning, EDA, data transformation, clustering, classification, and prediction evaluation.

<a name="how-to-run"></a>

## How to Run
1. Ensure R is installed on your system.
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hotel-booking-cancellation-prediction.git
   cd hotel-booking-cancellation-prediction
3. Run the `project-hotel-booking.R` script in RStudio or any R environment.

<a name="steps"></a>

## Steps
1. **Data Cleaning**: 
    - Identify and handle missing values, including NULL, NA, and undefined values.
    - Eliminate logical errors such as entries with no number of customers and negative average daily rates.
    
2. **Data Transformation**:
    - Create new columns for country names and continents based on ISO3 country codes.
    - Perform data transformation such as changing the value of the continent if the booking is domestic.
    
3. **EDA (Exploratory Data Analysis)**:
    - Visualize the distribution of cancellation status by hotel type, lead time, and month of arrival.
    - Analyze booking status by country and market segment.
    - Investigate the price charged by market segment.
    
4. **Data Selection**:
    - Analyze correlations between variables to select relevant features.
    - Implement feature selection using random forest to identify important features.
    
5. **Clustering**:
    - Determine the optimal number of clusters using hierarchical clustering and K-means.
    - Perform K-means clustering to group similar data points.
    
6. **Classification**:
    - Split the data into training and testing sets.
    - Train machine learning models including K-nearest neighbor, Decision Tree, XGBoost, and Random Forest.
    - Evaluate model performance using confusion matrix, ROC curve, and AUC.
    
7. **Prediction Evaluation**:
    - Evaluate the prediction accuracy, precision, and AUC of each model.
    - Compare the performance of different models.


<a name="results"></a>

## Results
- The project achieved a prediction accuracy of 99.3% in predicting cancellations.
- It was found that customers who reserved from the same country are more likely to cancel compared to those who reserved from outside the country.

<a name="acknowledgements"></a>

## Acknowledgement
- Douglas College for the comprehensive Data Analytics program.