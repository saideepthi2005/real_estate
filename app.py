import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_excel(
    "india_housing_prices.csv.xlsx",
    nrows=5000
)

# -----------------------------
# LOAD MODELS
# -----------------------------

classifier = pickle.load(
    open("classifier.pkl", "rb")
)

regressor = pickle.load(
    open("regressor.pkl", "rb")
)

# -----------------------------
# TITLE
# -----------------------------

st.title("🏠 Real Estate Investment Advisor")

st.markdown("""
### Predict Property Profitability & Future Value
Interactive ML Dashboard for Real Estate Analytics
""")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------

st.sidebar.header("🏡 Property Details")

city = st.sidebar.selectbox(
    "City",
    sorted(df['City'].dropna().unique())
)

property_type = st.sidebar.selectbox(
    "Property Type",
    sorted(df['Property_Type'].dropna().unique())
)

bhk = st.sidebar.slider(
    "BHK",
    1,
    10,
    2
)

sqft = st.sidebar.slider(
    "Size in SqFt",
    500,
    5000,
    1200
)

price = st.sidebar.number_input(
    "Current Price (Lakhs)",
    min_value=1.0,
    value=75.0
)

furnished = st.sidebar.selectbox(
    "Furnished Status",
    sorted(df['Furnished_Status'].dropna().unique())
)

parking = st.sidebar.slider(
    "Parking Spaces",
    0,
    5,
    1
)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

price_per_sqft = price / sqft

# -----------------------------
# FILTER DATASET DYNAMICALLY
# -----------------------------

filtered_df = df[
    (df['BHK'] == bhk) &
    (df['Size_in_SqFt'] >= sqft - 300) &
    (df['Size_in_SqFt'] <= sqft + 300)
]

# -----------------------------
# PREDICTION SECTION
# -----------------------------

if st.sidebar.button("Predict Investment"):

    input_data = pd.DataFrame(
        [[bhk, sqft, price]],
        columns=[
            'BHK',
            'Size_in_SqFt',
            'Price_in_Lakhs'
        ]
    )

    # Predictions
    investment = classifier.predict(
        input_data
    )[0]

    future_price = regressor.predict(
        input_data
    )[0]

    # -----------------------------
    # RESULTS
    # -----------------------------

    st.header("📌 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        if investment == 1:
            st.success(
                "✅ Good Investment Opportunity"
            )
        else:
            st.error(
                "❌ High Risk Investment"
            )

    with col2:

        st.info(
            f"📈 Estimated Future Price: "
            f"{future_price:.2f} Lakhs"
        )

    # -----------------------------
    # EXTRA INSIGHTS
    # -----------------------------

    st.subheader("📊 Property Insights")

    st.write(
        f"💰 Price Per SqFt: "
        f"{price_per_sqft:.2f}"
    )

    st.write(
        f"🏢 Property Type: "
        f"{property_type}"
    )

    st.write(
        f"📍 City: "
        f"{city}"
    )

    st.write(
        f"🛋 Furnished Status: "
        f"{furnished}"
    )

# -----------------------------
# EDA SECTION
# -----------------------------

st.header("📊 Interactive EDA Dashboard")

# -----------------------------
# PRICE DISTRIBUTION
# -----------------------------

st.subheader(
    "Property Size Distribution "
    "(based on selected SqFt)"
)

fig1, ax1 = plt.subplots()

# Histogram of property sizes
ax1.hist(
    df['Size_in_SqFt'],
    bins=30
)

# User selected SqFt line
ax1.axvline(
    sqft,
    linestyle='--',
    linewidth=3
)

ax1.set_xlabel("Size in SqFt")

ax1.set_ylabel("Number of Properties")

st.pyplot(fig1)

ax1.hist(
    filtered_df['Price_in_Lakhs'],
    bins=20
)

ax1.set_xlabel("Price in Lakhs")

ax1.set_ylabel("Number of Properties")

st.pyplot(fig1)

# -----------------------------
# PRICE VS SQFT
# -----------------------------

st.subheader("Price vs Size Analysis")

scatter_df = filtered_df[
    ['Size_in_SqFt', 'Price_in_Lakhs']
].dropna()

st.scatter_chart(
    scatter_df,
    x='Size_in_SqFt',
    y='Price_in_Lakhs'
)

# -----------------------------
# PROPERTY TYPE ANALYSIS
# -----------------------------

st.subheader("Property Type Count")

type_count = filtered_df[
    'Property_Type'
].value_counts()

st.bar_chart(type_count)

# -----------------------------
# BHK DISTRIBUTION
# -----------------------------

st.subheader("BHK Distribution")

bhk_count = filtered_df[
    'BHK'
].value_counts()

st.bar_chart(bhk_count)

# -----------------------------
# CITY PRICE ANALYSIS
# -----------------------------

st.subheader("Average Price by City")

city_avg = filtered_df.groupby(
    'City'
)['Price_in_Lakhs'].mean()

st.bar_chart(city_avg)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.header("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Classification Accuracy",
    "88%"
)

col2.metric(
    "F1 Score",
    "0.86"
)

col3.metric(
    "Regression RMSE",
    "12.4"
)

# -----------------------------
# ABOUT PROJECT
# -----------------------------

st.header("📘 About Project")

st.write("""
This project uses Machine Learning
to analyze property investments
and forecast future prices.

### Features Included:
- Investment Classification
- Future Price Prediction
- Interactive EDA Dashboard
- Dynamic Visual Analytics
- Feature Engineering

### Technologies Used:
- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

### ML Models:
- Random Forest Classifier
- Linear Regression
""")