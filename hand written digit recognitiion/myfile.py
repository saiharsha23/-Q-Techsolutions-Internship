import pickle
import streamlit as st

# Load the pre-trained model
model1 = pickle.load(open("logistic_regression_model.pkl", "rb"))

# Streamlit app function
def myf1():
    st.title("Digit Prediction with Logistic Regression")
    st.write("This app predicts digits based on your input using a Logistic Regression model.")

    # Input for prediction
    rows = st.number_input(
        "Enter a value to predict (scaled between 0 and 1):", 
        min_value=0.0, 
        max_value=1.0, 
        step=0.1
    )

    # Prediction button
    pred = st.button("Predict")
    if pred:
        try:
            # Ensure the input is correctly shaped for the model
            op = model1.predict([[rows]])
            st.success(f"The predicted digit is: {op[0]}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Run the app function
myf1()
