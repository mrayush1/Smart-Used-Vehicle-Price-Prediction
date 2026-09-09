# Smart-Used-Vehicle-Price-Prediction
A simple machine learning project that predicts the estimated resale price of used vehicles.

This is my Trimester 9's term project III for the BSc Data Science and AI program at IIT Guwahati.

The project is a simple machine learning based web application that predicts the resale price of a used vehicle.

The dataset contains different types of used vehicles, including cars, bikes and scooters. The user can enter the details of a vehicle and the system gives an estimated selling price in lakh.

## Project Features

- Predicts the estimated price of a used vehicle
- Supports different types of vehicles
- Simple and easy-to-use web interface
- Uses machine learning for prediction
- Shows the predicted price in Indian Rupees (₹ lakh)

## Dataset

The dataset contains information about used vehicles and their selling prices.

Some of the main columns are:

- Car_Name
- Year
- Selling_Price
- Present_Price
- Kms_Driven
- Fuel_Type
- Seller_Type
- Transmission
- Owner

The dataset was cleaned and prepared before training the machine learning models.

## Machine Learning

I tried different machine learning models for this project:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Gradient Boosting gave the best result among the models tested, so it was used for the final prediction system.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

## Project Structure

```text
Used Car Price Prediction/
│
├── app.py
├── cardata.csv
├── model.pkl
├── requirements.txt
└── README.md
