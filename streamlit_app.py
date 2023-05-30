import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

def load_pickles(model_pickle_path, label_encoder_pickle_path):
    with open(model_pickle_path, "rb") as model_pickle_opener:
        model = pickle.load(model_pickle_opener)
    with open(label_encoder_pickle_path, "rb") as label_encoder_opener:
        label_encoder_dict = pickle.load(label_encoder_opener)
        
    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df_out = df.copy()
    df_out.replace(" ", 0, inplace=True)
    df_out.loc[:, 'TotalCharges'] = pd.to_numeric(df_out.loc[:, 'TotalCharges'])
    if 'customerID' in df_out.columns:
        df_out.drop('customerID', axis=1, inplace=True)
    for column, le in label_encoder_dict.items():
        if column in df_out.columns:
            df_out.loc[:, column] = le.transform(df_out.loc[:, column])          
    
    return df_out

def make_predictions(test_data):
    model_pickle_path = "./models/churn_prediction_model.pkl"
    label_encoder_pickle_path = "./models/churn_prediction_label_encoder.pkl"
    
    model, label_encoder_dict = load_pickles(model_pickle_path, label_encoder_pickle_path)
    
    data_processed = pre_process_data(test_data, label_encoder_dict)
    if "Churn" in data_processed.columns:
        data_processed = data_processed.drop('Churn', axis=1)
    prediction = model.predict(data_processed)
    return prediction

if __name__ == "__main__":
    st.title("Customer churn prediction")
    data = pd.read_csv("./data/holdout_data.csv")
    
    #visualize customer's data
    gender = st.selectbox("Select customer's gender", ["Female", "Male"])

    senior_citizen_input = st.selectbox('Is the customer a senior citizen?', ['No','Yes'])
    senior_citizen = 1 if senior_citizen_input == "Yes" else 0
    
    partner = st.selectbox("Does the customer has a partner?", ["Yes", "No"])

    dependents = st.selectbox("Does the customer has dependents", ["Yes", "No"])
    
    tenure = st.slider("How long of tenure?", 0, 72, 24)

    phone_service = st.selectbox("Does the customer has access to phone service?", ["Yes", "No"])

    multiple_lines = st.selectbox("Does the customer has multiple lines?", ["Yes", "No", "No phone service"])

    internet_service = st.selectbox("Does the customer has internet service?", ["Fiber optic", "DSL", "No"])

    online_security = st.selectbox("Does the customer has online security?", ["Yes", "No", "No internet service"])
    
    online_backup = st.selectbox("Does the customer has online backup?", ["Yes", "No", "No internet service"])
   
    device_protection = st.selectbox("Does the customer has device protection?", ["Yes", "No", "No internet service"])

    tech_support = st.selectbox("Does the customer has tech support?", ["Yes", "No", "No internet service"])

    streaming_tv = st.selectbox("Does the customer has streaming TV?", ["Yes", "No", "No internet service"])

    streaming_movies = st.selectbox("Does the customer has streaming movies?", ["Yes", "No", "No internet service"])

    contract = st.selectbox("What is the customer's contract?", ["Month-to-month", "One year", "Two year "])
    
    paperless_billing = st.selectbox("Is the billing paperless?", ["Yes", "No"])

    payment_method = st.selectbox("Payment method?", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    monthly_charges = st.slider("Monthly charges:", 0, 120, 50)

    total_charges = st.slider("Total charges:", 0, 8600, 2000)

    
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    customer_data = pd.DataFrame([input_dict])
    st.table(customer_data)
    
    if st.button("Predict Churn"):
        prediction = make_predictions(customer_data)[0]
        prediction_string = "Will churn" if prediction == 1 else "Won't churn"
        st.markdown(f"Customer prediction: {prediction_string}")        
        