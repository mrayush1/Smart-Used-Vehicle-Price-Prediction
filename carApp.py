import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import random



st.set_page_config(
    page_title="Smart Used Vehicle Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cardata.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_car_price_model.pkl"
)



st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #172554;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .small-text {
        color: #64748b;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.06);
    }

    .result-box {
        background: linear-gradient(
            135deg,
            #0f766e,
            #14b8a6
        );

        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;

        margin-top: 20px;
        margin-bottom: 20px;

        box-shadow:
            0 10px 30px rgba(20, 184, 166, 0.20);
    }

    .result-label {
        font-size: 16px;
        opacity: 0.9;
    }

    .result-price {
        font-size: 44px;
        font-weight: 800;
        margin: 8px 0;
    }

    .result-description {
        font-size: 13px;
        opacity: 0.85;
    }

    .info-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .warning-box {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    section[data-testid="stSidebar"] {
        background-color: #172554;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )



@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )



try:

    model = load_model()
    data = load_data()

except Exception as e:

    st.error(
        "Unable to load the model or dataset."
    )

    st.error(
        str(e)
    )

    st.stop()



bike_keywords = [
    "Activa",
    "Bajaj",
    "Hero ",
    "Honda CB",
    "Honda CBR",
    "Honda Dream",
    "Honda Karizma",
    "Hyosung",
    "KTM",
    "Mahindra Mojo",
    "Royal Enfield",
    "Suzuki Access",
    "TVS ",
    "Yamaha",
    "UM Renegade"
]


def identify_vehicle_type(vehicle_name):

    name = str(vehicle_name).strip()

    for keyword in bike_keywords:

        if name.lower().startswith(
            keyword.lower()
        ):

            return "Bike / Scooter"

    return "Car"


data["Vehicle_Type"] = data[
    "Car_Name"
].apply(
    identify_vehicle_type
)


st.sidebar.title(
    "🚗 AutoValue"
)

st.sidebar.write(
    "Smart Used Vehicle Price Prediction"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🧠 Smart Prediction",
        "✍️ Manual Prediction",
        "🚗🏍️ Vehicle Type"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    "### Dataset"
)

st.sidebar.write(
    f"📊 {len(data):,} vehicle records"
)

st.sidebar.write(
    f"🚘 {len(data[data['Vehicle_Type'] == 'Car']):,} cars"
)

st.sidebar.write(
    f"🏍️ {len(data[data['Vehicle_Type'] == 'Bike / Scooter']):,} bikes/scooters"
)

st.sidebar.divider()

st.sidebar.caption(
    "Prices are displayed in ₹ Lakh"
)


st.markdown(
    '<div class="title">'
    '🚗 Smart Used Vehicle Price Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based estimated resale price prediction '
    'for used cars, bikes and scooters'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


if page == "🏠 Home":

    st.markdown(
        '<div class="section-title">'
        'Welcome to Smart Used Vehicle Price Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application estimates the resale price of a used "
        "vehicle based on its characteristics and patterns "
        "learned from historical vehicle data."
    )

    st.markdown(
        """
        <div class="info-box">

        <b>💰 Price Unit</b><br>

        All prices in this application are expressed in
        <b>Indian Rupees (₹ Lakh)</b>.

        <br><br>

        For example:<br>

        <b>2.76 = ₹2.76 Lakh = ₹2,76,000</b><br>
        <b>4.80 = ₹4.80 Lakh = ₹4,80,000</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📊 Project Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📁 Total Vehicles",
            f"{len(data):,}"
        )

    with col2:

        st.metric(
            "🚗 Cars",
            f"{len(data[data['Vehicle_Type'] == 'Car']):,}"
        )

    with col3:

        st.metric(
            "🏍️ Bikes / Scooters",
            f"{len(data[data['Vehicle_Type'] == 'Bike / Scooter']):,}"
        )

    with col4:

        st.metric(
            "🏷️ Vehicle Names",
            f"{data['Car_Name'].nunique():,}"
        )

    st.divider()

    st.markdown(
        "### ⚡ Quick Start"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "🧠 **Smart Prediction**\n\n"
            "Select an existing vehicle record from the "
            "dataset. Every individual record is available, "
            "including repeated vehicle names."
        )

    with col2:

        st.info(
            "✍️ **Manual Prediction**\n\n"
            "Enter vehicle characteristics yourself and "
            "generate an estimated resale price."
        )

    st.divider()

    st.markdown(
        "### 🔄 How It Works"
    )

    st.write(
        "📊 Historical Vehicle Data"
    )

    st.write(
        "↓"
    )

    st.write(
        "🧹 Data Preparation"
    )

    st.write(
        "↓"
    )

    st.write(
        "🤖 Machine Learning Model"
    )

    st.write(
        "↓"
    )

    st.write(
        "💰 Estimated Resale Price"
    )

    st.divider()

    st.warning(
        "⚠️ The predicted value is an estimate based on "
        "historical data. It should not be interpreted as "
        "the exact current 2026 market price."
    )


elif page == "🧠 Smart Prediction":

    st.markdown(
        '<div class="section-title">'
        '🧠 Smart Vehicle Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select any individual vehicle record. "
        "Repeated names such as Verna or Ciaz are kept as separate "
        "records because their year, mileage, price and other "
        "characteristics may be different."
    )

    st.markdown(
        "### 🚘 Select Vehicle Record"
    )

    vehicle_options = []

    for index, row in data.iterrows():

        label = (
            f"Record {index + 1} | "
            f"{row['Car_Name']} | "
            f"{int(row['Year'])} | "
            f"{int(row['Kms_Driven']):,} km | "
            f"{row['Fuel_Type']} | "
            f"{row['Transmission']}"
        )

        vehicle_options.append(
            label
        )



    if "smart_record_index" not in st.session_state:

        st.session_state.smart_record_index = random.randint(
            0,
            len(data) - 1
        )

    current_index = st.session_state.smart_record_index



    selected_option = st.selectbox(
        "Choose a vehicle record",
        vehicle_options,
        index=current_index,
        key="smart_vehicle_selector"
    )

    selected_index = vehicle_options.index(
        selected_option
    )

    st.session_state.smart_record_index = selected_index



    col1, col2 = st.columns(
        [3, 1]
    )

    with col2:

        if st.button(
            "🔄 Random Vehicle",
            use_container_width=True
        ):

            st.session_state.smart_record_index = random.randint(
                0,
                len(data) - 1
            )

            st.rerun()


    selected_row = data.iloc[
        st.session_state.smart_record_index
    ]

    vehicle_name = str(
        selected_row["Car_Name"]
    )

    vehicle_type = str(
        selected_row["Vehicle_Type"]
    )

    vehicle_year = int(
        selected_row["Year"]
    )

    present_price = float(
        selected_row["Present_Price"]
    )

    kms_driven = int(
        selected_row["Kms_Driven"]
    )

    fuel_type = str(
        selected_row["Fuel_Type"]
    )

    seller_type = str(
        selected_row["Seller_Type"]
    )

    transmission = str(
        selected_row["Transmission"]
    )

    owner = int(
        selected_row["Owner"]
    )

    st.divider()

    st.markdown(
        "### 📋 Vehicle Details"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Vehicle",
            vehicle_name
        )

    with col2:

        st.metric(
            "Type",
            vehicle_type
        )

    with col3:

        st.metric(
            "Year",
            vehicle_year
        )

    with col4:

        st.metric(
            "Kilometres",
            f"{kms_driven:,}"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Present Price",
            f"₹{present_price:.2f} Lakh"
        )

    with col2:

        st.metric(
            "Fuel",
            fuel_type
        )

    with col3:

        st.metric(
            "Transmission",
            transmission
        )

    with col4:

        st.metric(
            "Previous Owners",
            owner
        )

    st.divider()


    if st.button(
        "🧠 Predict Estimated Resale Price",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "Car_Name": [vehicle_name],
                "Year": [vehicle_year],
                "Present_Price": [present_price],
                "Kms_Driven": [kms_driven],
                "Fuel_Type": [fuel_type],
                "Seller_Type": [seller_type],
                "Transmission": [transmission],
                "Owner": [owner]
            }
        )

        try:

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(
                0,
                float(prediction)
            )

            rupees = prediction * 100000

            st.markdown(
                f"""
                <div class="result-box">

                <div class="result-label">
                Estimated Resale Price
                </div>

                <div class="result-price">
                ₹{prediction:.2f} Lakh
                </div>

                <div class="result-description">
                Approximately ₹{rupees:,.0f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Estimated resale price generated successfully!"
            )

            st.info(
                "💡 This is an ML-based estimate using the "
                "selected historical vehicle record. It is "
                "not a guaranteed current market price."
            )

        except Exception as e:

            st.error(
                "Prediction could not be generated."
            )

            st.error(
                str(e)
            )



elif page == "✍️ Manual Prediction":

    st.markdown(
        '<div class="section-title">'
        '✍️ Manual Vehicle Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the vehicle characteristics manually to "
        "estimate its resale price."
    )



    vehicle_names = sorted(
        data["Car_Name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    fuel_types = sorted(
        data["Fuel_Type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    seller_types = sorted(
        data["Seller_Type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    transmissions = sorted(
        data["Transmission"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )



    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🚘 Vehicle Information"
        )

        vehicle_name = st.selectbox(
            "Vehicle Name",
            vehicle_names,
            key="manual_vehicle_name"
        )

        year = st.number_input(
            "Manufacturing Year",
            min_value=int(
                data["Year"].min()
            ),
            max_value=int(
                data["Year"].max()
            ),
            value=int(
                data["Year"].max()
            ),
            step=1
        )

        present_price = st.number_input(
            "Present Price (₹ Lakh)",
            min_value=0.0,
            max_value=float(
                data["Present_Price"].max()
            ),
            value=float(
                data["Present_Price"].median()
            ),
            step=0.05
        )

        kms_driven = st.number_input(
            "Kilometres Driven",
            min_value=0,
            max_value=int(
                data["Kms_Driven"].max()
            ),
            value=int(
                data["Kms_Driven"].median()
            ),
            step=500
        )

    with col2:

        st.subheader(
            "⚙️ Vehicle Specifications"
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_types
        )

        seller_type = st.selectbox(
            "Seller Type",
            seller_types
        )

        transmission = st.selectbox(
            "Transmission",
            transmissions
        )

        owner = st.number_input(
            "Previous Owners",
            min_value=0,
            max_value=int(
                data["Owner"].max()
            ),
            value=0,
            step=1
        )

    st.divider()

    st.info(
        "💰 All prices are entered and displayed in ₹ Lakh."
    )

    if st.button(
        "🔮 Predict Estimated Resale Price",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "Car_Name": [vehicle_name],
                "Year": [year],
                "Present_Price": [present_price],
                "Kms_Driven": [kms_driven],
                "Fuel_Type": [fuel_type],
                "Seller_Type": [seller_type],
                "Transmission": [transmission],
                "Owner": [owner]
            }
        )

        try:

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(
                0,
                float(prediction)
            )

            rupees = prediction * 100000

            st.markdown(
                f"""
                <div class="result-box">

                <div class="result-label">
                Estimated Resale Price
                </div>

                <div class="result-price">
                ₹{prediction:.2f} Lakh
                </div>

                <div class="result-description">
                Approximately ₹{rupees:,.0f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Prediction generated successfully!"
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.error(
                str(e)
            )



elif page == "🚗🏍️ Vehicle Type":

    st.markdown(
        '<div class="section-title">'
        '🚗🏍️ Vehicle Type Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Choose whether you want to work with cars or "
        "bikes/scooters."
    )



    vehicle_type_choice = st.radio(
        "Select Vehicle Category",
        [
            "🚗 Cars",
            "🏍️ Bikes / Scooters"
        ],
        horizontal=True
    )

    if vehicle_type_choice == "🚗 Cars":

        selected_type = "Car"

        st.subheader(
            "🚗 Car Price Prediction"
        )

    else:

        selected_type = "Bike / Scooter"

        st.subheader(
            "🏍️ Bike / Scooter Price Prediction"
        )



    type_data = data[
        data["Vehicle_Type"]
        == selected_type
    ].copy()

    type_vehicle_names = sorted(
        type_data["Car_Name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )



    st.info(
        f"This category contains "
        f"{len(type_data):,} records and "
        f"{len(type_vehicle_names):,} unique vehicle names."
    )



    selected_vehicle = st.selectbox(
        "Select Vehicle",
        type_vehicle_names
    )



    vehicle_records = type_data[
        type_data["Car_Name"].astype(str)
        == selected_vehicle
    ]

    record_options = []

    for index, row in vehicle_records.iterrows():

        label = (
            f"Record {index + 1} | "
            f"{selected_vehicle} | "
            f"{int(row['Year'])} | "
            f"{int(row['Kms_Driven']):,} km | "
            f"{row['Fuel_Type']}"
        )

        record_options.append(
            (index, label)
        )

    selected_record_label = st.selectbox(
        "Select Specific Vehicle Record",
        [x[1] for x in record_options]
    )

    selected_record_index = next(
        x[0]
        for x in record_options
        if x[1] == selected_record_label
    )

    selected_record = data.loc[
        selected_record_index
    ]



    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Vehicle",
            selected_record["Car_Name"]
        )

    with col2:

        st.metric(
            "Year",
            int(selected_record["Year"])
        )

    with col3:

        st.metric(
            "Kilometres",
            f"{int(selected_record['Kms_Driven']):,}"
        )

    with col4:

        st.metric(
            "Present Price",
            f"₹{float(selected_record['Present_Price']):.2f} Lakh"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Fuel",
            selected_record["Fuel_Type"]
        )

    with col2:

        st.metric(
            "Seller",
            selected_record["Seller_Type"]
        )

    with col3:

        st.metric(
            "Transmission",
            selected_record["Transmission"]
        )

    with col4:

        st.metric(
            "Owners",
            int(selected_record["Owner"])
        )

    st.divider()



    if st.button(
        "🔮 Predict Price",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "Car_Name": [
                    selected_record["Car_Name"]
                ],

                "Year": [
                    selected_record["Year"]
                ],

                "Present_Price": [
                    selected_record["Present_Price"]
                ],

                "Kms_Driven": [
                    selected_record["Kms_Driven"]
                ],

                "Fuel_Type": [
                    selected_record["Fuel_Type"]
                ],

                "Seller_Type": [
                    selected_record["Seller_Type"]
                ],

                "Transmission": [
                    selected_record["Transmission"]
                ],

                "Owner": [
                    selected_record["Owner"]
                ]
            }
        )

        try:

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(
                0,
                float(prediction)
            )

            rupees = prediction * 100000

            st.markdown(
                f"""
                <div class="result-box">

                <div class="result-label">
                Estimated {selected_type} Resale Price
                </div>

                <div class="result-price">
                ₹{prediction:.2f} Lakh
                </div>

                <div class="result-description">
                Approximately ₹{rupees:,.0f}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Vehicle price prediction generated successfully!"
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.error(
                str(e)
            )


st.divider()

st.caption(
    "🚗 Smart Used Vehicle Price Prediction System | "
    "Machine Learning Project "
)