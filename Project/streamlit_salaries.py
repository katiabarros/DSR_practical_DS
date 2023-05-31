import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def load_pickles(model_pickle_path, label_encoder_pickle_path):
    with open(model_pickle_path, "rb") as model_pickle_opener:
        model = pickle.load(model_pickle_opener)
    with open(label_encoder_pickle_path, "rb") as label_encoder_opener:
        label_encoder_dict = pickle.load(label_encoder_opener)
        
    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df_out = df.copy()
    df_out['work_year'] = (df_out['work_year']).astype(int) / 2023
    df_out['remote_ratio'] = (df_out['remote_ratio']).astype(int) / 100
    #df_out['salary_in_usd'] = df_out['salary_in_usd'] / 300_000

    for column, le in label_encoder_dict.items():
        df_out.loc[:, column] = le.transform(df_out.loc[:, column])          
    
    return df_out


def make_predictions(test_data):
    model_pickle_path = "./models/salaries_prediction_model.pkl"
    label_encoder_pickle_path = "./models/salaries_prediction_label_encoder.pkl"
    
    model, label_encoder_dict = load_pickles(model_pickle_path, label_encoder_pickle_path)
    
    data_processed = pre_process_data(test_data, label_encoder_dict)
    prediction = model.predict(data_processed)
    return prediction

if __name__ == "__main__":
    st.title("Data Scientists salary prediction")
    #data = pd.read_csv("./data/holdout_data.csv")
    
    #visualize customer's data
    work_year = st.selectbox("Select work year", ["2020", "2021", "2022", "2023"])

    experience_level = st.selectbox('What is the experience level? (EN = Entry Level, MI = Mid Level, SE = Senior Level, EX = Executive Level)', ['EN', 'MI', 'SE', 'EX'])
    
    employment_type = st.selectbox("What type of employment? (FT = full time, PT = part time, CT = contractor, FL = freelancer)", ["FT", "PT", "CT", "FL"])

    job_title = st.selectbox("What is the job title", 
                             ["Data Engineer", "Data Scientist", "Data Analyst",
                              "Machine Learning Engineer", "Analytics Engineer",
                              "Data Architect", "Research Scientist", "Data Science Manager",
                              "Applied Scientist", "Research Engineer", "Other"])
    
    employee_residence = st.selectbox("Where do you live", ["US", "GB", "CA", "ES", "IN", 
                                                            "DE", "Other"])

    remote_ratio = st.selectbox("How much would you like to do home office [in %]?", ["0", "50", "100"])

    company_location = st.selectbox("Where is the company located at?", ["US", "GB", "CA", "ES", "IN", 
                                                            "DE", "Other"])

    company_size = st.selectbox("Company size", ["S", "M", "L"])


    
    input_dict = {
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'employee_residence': employee_residence,
        'remote_ratio': remote_ratio,
        'company_location': company_location,
        'company_size': company_size,
    }
    
    customer_data = pd.DataFrame([input_dict])
    st.table(customer_data)
    
    if st.button("Predict Salary"):
        prediction = make_predictions(customer_data)[0]
        st.markdown(f"Salary prediction: {round(prediction*300_000, 2)} USD")      