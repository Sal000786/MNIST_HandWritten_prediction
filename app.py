import streamlit as st
import joblib as jb 
import tensorflow as tf

model = tf.keras.models.load_model("mnist_model.h5")
st.title('MNIST Hand Written Digit Classification')


# v1 = st.number_input("GRE Scores ( out of 340 )")
# v2 = st.number_input("TOEFL Scores ( out of 120 )")
# v3 = st.number_input("University Rating ( out of 5 )")
# # v4 = st.number_input("Statement of Purpose (out of 5 in float(like 4.0) )")
# v5 = st.number_input("Letter of Recommendation Strength ( out of 5 in float(like 4.1) )")
# v6 = st.number_input("Undergraduate GPA ( out of 10 float(like 6.1))")
# v7 = st.number_input("Research Experience ( either 0 or 1 )")




# Define a function to make predictions
def make_prediction():
    input_data = [[v1, v2, v3, v5, v6, v7]]
    scaled_data=scaler.transform(input_data)
    print(scaled_data)
    prediction = model.predict(scaled_data)
    return prediction

# Use the st.button with on_click parameter
if st.button('Submit'):
    prediction_result = make_prediction()
    # st.write("Prediction:", prediction_result)
    # print(prediction_result)
    if prediction_result>0.75:
        print(prediction_result)
        st.write(" Hurray!! The student is eligible.")
    else:
        print(prediction_result)
        st.write("Please try next time !!")
