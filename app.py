import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Create Streamlit app
st.title("FraudShield")
st.markdown('<div class="info-text"> Spotting Scams Swiftly, Securely, and Seamlessly!</div>', unsafe_allow_html=True)
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Add custom styled text


# Create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # Get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # Make prediction
    prediction = model.predict(features.reshape(1, -1))
    # Display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(45deg, orange, red);
        height: 100vh;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
    }
    h1 {
        font-family: 'Times New Roman', serif;
        color: #333333;
        text-align: center;
    }
    
    .info-text {
        font-family: 'Arial', sans-serif;
        font-size: 20px;
        color: #333333;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    .header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #333333;
        text-align: center;
    }
    .footer {
        font-family: 'Courier New', Courier, monospace;
        color: #333333;
        font-size: 20px;
        text-align: center;
        padding-top: 20px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="footer">Team Innovates</div>', unsafe_allow_html=True)
