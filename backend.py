
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cardata.csv"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n" + "=" * 70)
print("USED CAR PRICE PREDICTION")
print("COMPLETE ML BACKEND")
print("=" * 70)

print("\nProject folder:")
print(BASE_DIR)

print("\nLooking for dataset:")
print(DATA_PATH)



print("\n[1] Checking dataset...")

if not os.path.exists(DATA_PATH):

    print("\nERROR: Dataset was not found.")
    print("\nPython is looking here:")
    print(DATA_PATH)

    print("\nPlease make sure your folder structure is exactly:")

    print("""
Used Car Price Prediction
│
├── backend.py
│
└── data
    └── cardata.csv
""")

    print("\nFiles currently inside the data folder:")

    data_folder = os.path.join(
        BASE_DIR,
        "data"
    )

    if os.path.exists(data_folder):

        for file in os.listdir(data_folder):
            print("   ", file)

    else:
        print("   data folder does not exist.")

    raise SystemExit


print("\nDataset found!")

try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    print("\nERROR while reading the CSV file:")
    print(e)

    raise SystemExit


print("Dataset loaded successfully.")


print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Records:", len(df))

print("Total Columns:", len(df.columns))


df.columns = df.columns.str.strip()

print("\nColumn Names:")

for column in df.columns:
    print(" -", column)


expected_columns = [
    "Car_Name",
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner"
]

print("\n[2] Checking required columns...")

missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR!")
    print("These columns are missing:")

    for column in missing_columns:
        print(" -", column)

    print("\nActual columns are:")
    print(list(df.columns))

    raise SystemExit

print("All required columns are present.")


print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

print(df.head())


print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)



print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print("\nTotal Missing Values:", total_missing)


print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate Records:", duplicate_count)

if duplicate_count > 0:

    print("\nRemoving duplicate records...")

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    print(
        "Records after removing duplicates:",
        len(df)
    )

else:

    print("No duplicate records found.")



numerical_columns = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Owner"
]

print("\n[3] Converting numerical columns...")

for column in numerical_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("Numerical columns processed.")


print("\n[4] Handling missing values...")

categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]

# Numerical missing values → median
for column in numerical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].median()
        )


# Categorical missing values → mode
for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


print("\nMissing values after handling:")

print(df.isnull().sum())



print("\n" + "=" * 70)
print("INVALID VALUE CHECK")
print("=" * 70)

print(
    "Year <= 0:",
    (df["Year"] <= 0).sum()
)

print(
    "Selling Price <= 0:",
    (df["Selling_Price"] <= 0).sum()
)

print(
    "Present Price <= 0:",
    (df["Present_Price"] <= 0).sum()
)

print(
    "Kms Driven < 0:",
    (df["Kms_Driven"] < 0).sum()
)

print(
    "Owner < 0:",
    (df["Owner"] < 0).sum()
)


before = len(df)

df = df[
    (df["Year"] > 0)
    & (df["Selling_Price"] > 0)
    & (df["Present_Price"] > 0)
    & (df["Kms_Driven"] >= 0)
    & (df["Owner"] >= 0)
].copy()

after = len(df)

print("\nInvalid records removed:", before - after)

print("Records remaining:", after)


print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe())


print("\n" + "=" * 70)
print("CATEGORICAL DATA")
print("=" * 70)

for column in categorical_columns:

    print("\n", column)

    print(
        df[column]
        .value_counts()
        .head(15)
    )



cleaned_file = os.path.join(
    OUTPUT_DIR,
    "cleaned_cardata.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)

print("\nCleaned dataset saved at:")

print(cleaned_file)


print("\n[5] Creating EDA visualizations...")


plt.figure(figsize=(9, 6))

plt.hist(
    df["Selling_Price"],
    bins=30
)

plt.title(
    "Distribution of Selling Price"
)

plt.xlabel(
    "Selling Price"
)

plt.ylabel(
    "Number of Cars"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "selling_price_distribution.png"
    )
)

plt.close()



plt.figure(figsize=(9, 6))

plt.scatter(
    df["Present_Price"],
    df["Selling_Price"],
    alpha=0.6
)

plt.title(
    "Present Price vs Selling Price"
)

plt.xlabel(
    "Present Price"
)

plt.ylabel(
    "Selling Price"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "present_vs_selling_price.png"
    )
)

plt.close()



plt.figure(figsize=(9, 6))

plt.scatter(
    df["Kms_Driven"],
    df["Selling_Price"],
    alpha=0.6
)

plt.title(
    "Kilometers Driven vs Selling Price"
)

plt.xlabel(
    "Kilometers Driven"
)

plt.ylabel(
    "Selling Price"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "kms_vs_selling_price.png"
    )
)

plt.close()


plt.figure(figsize=(9, 6))

plt.scatter(
    df["Year"],
    df["Selling_Price"],
    alpha=0.6
)

plt.title(
    "Car Year vs Selling Price"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Selling Price"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "year_vs_selling_price.png"
    )
)

plt.close()


correlation = df[
    numerical_columns
].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation)


plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title(
    "Correlation Matrix"
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_matrix.png"
    )
)

plt.close()


print("\n" + "=" * 70)
print("FEATURE AND TARGET PREPARATION")
print("=" * 70)

X = df.drop(
    "Selling_Price",
    axis=1
)

y = df["Selling_Price"]

print("\nTarget:")
print("Selling_Price")

print("\nFeatures:")

for column in X.columns:
    print(" -", column)



print("\n[6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    "\nTraining Records:",
    len(X_train)
)

print(
    "Testing Records:",
    len(X_test)
)



print("\n[7] Preparing preprocessing pipeline...")

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_columns
        ),

        (
            "numerical",

            "passthrough",

            [
                "Year",
                "Present_Price",
                "Kms_Driven",
                "Owner"
            ]
        )
    ]
)


models = {

    "Linear Regression":

        LinearRegression(),


    "Decision Tree":

        DecisionTreeRegressor(
            random_state=42,
            max_depth=12
        ),


    "Random Forest":

        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

results = []

trained_models = {}


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )


    # MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )


    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    # R2
    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2 Score": r2
    })


    trained_models[
        model_name
    ] = pipeline


    print(
        f"MAE      : {mae:.4f}"
    )

    print(
        f"RMSE     : {rmse:.4f}"
    )

    print(
        f"R2 Score : {r2:.4f}"
    )



print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print(
    results_df.to_string(
        index=False
    )
)


comparison_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)

print(
    "\nModel comparison saved:"
)

print(comparison_file)



best_model_name = (
    results_df.iloc[0]["Model"]
)

best_model = trained_models[
    best_model_name
]

best_r2 = (
    results_df.iloc[0]["R2 Score"]
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "\nBest Model:",
    best_model_name
)

print(
    "Best R2 Score:",
    round(best_r2, 4)
)



model_file = os.path.join(
    MODEL_DIR,
    "best_car_price_model.pkl"
)

joblib.dump(
    best_model,
    model_file
)

print(
    "\nBest model saved:"
)

print(model_file)



best_predictions = best_model.predict(
    X_test
)

prediction_results = pd.DataFrame({

    "Actual_Price":
        y_test.values,

    "Predicted_Price":
        best_predictions

})


print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED")
print("=" * 70)

print(
    prediction_results.head(15)
)



prediction_file = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

prediction_results.to_csv(
    prediction_file,
    index=False
)

print(
    "\nPrediction results saved:"
)

print(prediction_file)



plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.title(
    "Actual vs Predicted Selling Price"
)

plt.xlabel(
    "Actual Selling Price"
)

plt.ylabel(
    "Predicted Selling Price"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted.png"
    )
)

plt.close()


print("\n" + "=" * 70)
print("SAMPLE PREDICTION")
print("=" * 70)


sample_car = pd.DataFrame({

    "Car_Name": ["ritz"],

    "Year": [2014],

    "Present_Price": [5.59],

    "Kms_Driven": [27000],

    "Fuel_Type": ["Petrol"],

    "Seller_Type": ["Dealer"],

    "Transmission": ["Manual"],

    "Owner": [0]

})


sample_prediction = (
    best_model.predict(
        sample_car
    )
)


print("\nSample Car:")

print(
    sample_car.to_string(
        index=False
    )
)

print(
    "\nPredicted Selling Price:",
    round(
        sample_prediction[0],
        2
    )
)


print("\n" + "=" * 70)
print("BACKEND EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"""
Final Dataset Records : {len(df)}
Total Features        : {len(X.columns)}
Training Records      : {len(X_train)}
Testing Records       : {len(X_test)}

Models Trained        : {len(models)}
Best Model            : {best_model_name}
Best R2 Score         : {best_r2:.4f}

--------------------------------------------

Generated Files:

1. Cleaned Dataset
   {cleaned_file}

2. Best ML Model
   {model_file}

3. Model Comparison
   {comparison_file}

4. Test Predictions
   {prediction_file}

5. EDA Graphs
   {OUTPUT_DIR}

--------------------------------------------

NEXT STEP:
Create the Streamlit frontend app.py
and connect it with the saved model.
"""
)

print("=" * 70)
print("SUCCESS!")
print("=" * 70)

