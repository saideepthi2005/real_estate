import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel(
    "india_housing_prices.csv.xlsx",
    nrows=5000
)

# Select important columns
df = df[['BHK','Size_in_SqFt','Price_in_Lakhs']].dropna()

# Create classification label
median_price = df['Price_in_Lakhs'].median()

df['Good_Investment'] = (
    df['Price_in_Lakhs'] < median_price
).astype(int)

# Features
X = df[['BHK','Size_in_SqFt','Price_in_Lakhs']]

# Targets
y_class = df['Good_Investment']
y_reg = df['Price_in_Lakhs'] * 1.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    random_state=42
)

# Train classification model
classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

# Train regression model
regressor = LinearRegression()

regressor.fit(X, y_reg)

# Save models
pickle.dump(classifier, open("classifier.pkl", "wb"))
pickle.dump(regressor, open("regressor.pkl", "wb"))

print("Models trained successfully ✅")