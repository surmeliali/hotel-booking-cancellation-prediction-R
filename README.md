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
1. **Data Cleaning**: Handle missing values and eliminate logical errors.
2. **Data Transformation**: Create new columns for country names and continents; perform data transformation.
3. **EDA (Exploratory Data Analysis)**: Visualize distribution of cancellation status; analyze booking status by country and market segment.
4. **Data Selection**: Analyze correlations between variables; implement feature selection using random forest.
5. **Clustering**: Determine optimal number of clusters; perform K-means clustering.
6. **Classification**: Split data into training and testing sets; train machine learning models.
7. **Prediction Evaluation**: Evaluate model performance and compare different models.


<a name="results"></a>

## Results
- The project achieved a prediction accuracy of 99.3% in predicting cancellations.
- It was found that customers who reserved from the same country are more likely to cancel compared to those who reserved from outside the country.

<a name="acknowledgements"></a>

## Acknowledgement
- Douglas College for the comprehensive Data Analytics program.
