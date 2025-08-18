import os
import random
import datetime
#from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import openai
import json
import time
import requests
#import kumoai.experimental.rfm as rfm, os

# Set page config
st.set_page_config(
    page_title="Hospital Portal",
    page_icon="🏥",
    layout="wide"
)
with st.sidebar:
    st.image("assets/hrra_logo.png", use_column_width=True)

st.markdown("<h2 style='color:#0077B6;'>Hospital Readmission Risk Assessment</h2>", unsafe_allow_html=True)

st.markdown("""
**Using this tool is simple — Complete the 5 sections below in order from top to bottom:**
            
""")

@st.cache_data
def load_tables(csv_folder="/Users/kevinsoderholm/Desktop/Readmit/"):
    patients = pd.read_csv(os.path.join(csv_folder, "patients.csv"))
    admissions = pd.read_csv(os.path.join(csv_folder, "admissions.csv"))
    labs = pd.read_csv(os.path.join(csv_folder, "labs.csv"))
    meds = pd.read_csv(os.path.join(csv_folder, "medications.csv"))
    diagnosis = pd.read_csv(os.path.join(csv_folder, "diagnosis.csv"))
    return patients, admissions, labs, meds, diagnosis

patients_df, admissions_df, labs_df, meds_df, diagnosis_df = load_tables()

@st.cache_resource
def load_model():
    KUMO_API_KEY = st.secrets["kumo"]["api_key"]
    os.environ["KUMO_API_KEY"] = KUMO_API_KEY
    rfm.init()
    patients = rfm.LocalTable(
        patients_df,
        name="patients",
        primary_key="patient_id")
    patients['chronic_conditions'].stype = "multicategorical"
    labs = rfm.LocalTable(
        labs_df,
        name="labs",
        primary_key="lab_id",
        time_column="lab_timestamp")   
    medications = rfm.LocalTable(
        meds_df,
        name="medications",
        primary_key="med_id",
        time_column="start_date")
    admissions = rfm.LocalTable(
        admissions_df,
        name="admissions",
        primary_key="admission_id",
        time_column="discharge_date")
    admissions['los'].stype = "numerical"
    diagnosis = rfm.LocalTable(
        diagnosis_df,
        name="diagnosis",
        primary_key="diagnosis_id")
    graph = rfm.LocalGraph(tables=[
        patients,
        labs,
        medications,
        admissions,
        diagnosis
        ])
    graph.link(src_table="admissions", fkey="patient_id", dst_table="patients")
    graph.link(src_table="medications", fkey="admission_id", dst_table="admissions")
    graph.link(src_table="labs", fkey="admission_id", dst_table="admissions")
    graph.link(src_table="admissions", fkey="diagnosis_id", dst_table="diagnosis")
    return rfm.KumoRFM(graph)

def admission_factors(patient_row, admission_row, labs_df, meds_df):
    """
    Calculate factors for an admission that contribute to readmission risk.
    Returns a list of (factor_name, relative_importance_percent) tuples.
    Max importance normalized to 100%.
    """
    factors = []

    # --- Age ---
    age = patient_row.get("age", 0)
    if age >= 80:
        factors.append(("Advanced age (>=80)", 2.0))
    elif age >= 70:
        factors.append(("Age 70-79", 1.2))
    elif age >= 65:
        factors.append(("Age 65-69", 0.6))

    # --- Chronic conditions ---
    chronic_val = patient_row.get("chronic_conditions", "")
    chronic_list = str(chronic_val).split(", ") if pd.notna(chronic_val) else []
    for cond in chronic_list:
        if cond == "CHF": factors.append(("Chronic CHF", 1.2))
        if cond == "CKD": factors.append(("Chronic CKD", 1.0))
        if cond == "COPD": factors.append(("Chronic COPD", 0.8))
        if cond == "Diabetes": factors.append(("Chronic Diabetes", 0.5))
        if cond == "Hypertension": factors.append(("Chronic Hypertension", 0.3))
        if cond == "Cancer": factors.append(("Chronic Cancer", 0.7))
        if cond == "Stroke": factors.append(("Chronic Stroke", 0.6))
        if cond == "Asthma": factors.append(("Chronic Asthma", 0.3))

    # --- Diagnosis ---
    dx = admission_row.get("primary_diagnosis", "")
    if dx in ["CHF","Sepsis","Pneumonia","MI","CKD","Stroke"]:
        factors.append((f"Primary diagnosis: {dx}", 1.4))
    elif dx in ["GI Bleed","COPD","UTI"]:
        factors.append((f"Primary diagnosis: {dx}", 0.9))
    # Elective Surgery / Minor Trauma add nothing

    # --- Admission type ---
    adm_type = admission_row.get("admission_type", "")
    if adm_type == "Emergency": factors.append(("Emergency admission", 1.0))
    elif adm_type == "Urgent": factors.append(("Urgent admission", 0.5))

    # --- Labs ---
    adm_id = admission_row.get("admission_id")
    labs = labs_df[labs_df["admission_id"] == adm_id] if labs_df is not None else None
    if labs is not None and not labs.empty:
        for lab_name, threshold, imp in [
            ("BNP", 400, 1.0),
            ("Creatinine", 2.0, 0.9),
            ("WBC", 12.0, 0.7),
            ("CRP", 15.0, 0.5),
            ("Hemoglobin", 10.0, 0.5)
        ]:
            vals = labs[labs["lab_test"]==lab_name]["lab_value"]
            if not vals.empty:
                val = vals.max()
                if (lab_name != "Hemoglobin" and val > threshold) or (lab_name == "Hemoglobin" and val < threshold):
                    factors.append((f"Abnormal {lab_name} ({val:.1f})", imp))

    # --- Medications ---
    meds = meds_df[meds_df["admission_id"]==adm_id]["drug"].tolist() if meds_df is not None else []
    for drug in meds:
        if drug in ["Vancomycin","Piperacillin/Tazobactam","Meropenem"]: factors.append((f"High-risk med: {drug}", 1.1))
        if drug == "Furosemide": factors.append((f"CHF med: {drug}", 0.6))
        if drug == "Insulin": factors.append((f"Insulin prescribed", 0.3))

    # --- Normalize importance ---
    if factors:
        max_val = max([imp for _, imp in factors])
        factors_normalized = [(name, round(imp/max_val*100,1)) for name, imp in factors]
    else:
        factors_normalized = []
    # Sort descending
    factors_normalized = sorted(factors_normalized, key=lambda x: x[1], reverse=True)

    return factors_normalized

# Function to generate video using HeyGen API
def generate_video(script_text):
    url = 'https://api.heygen.com/v2/video/generate'
    headers = {
        'X-Api-Key': HEYGEN_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": "Stacy_in_Doctor_Front",  # Replace with your avatar ID
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": script_text,
                    "voice_id": "6fe41db3d6ee40e4800718270ba22670"  # Replace with your voice ID
                }
            }
        ]
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["data"]["video_id"]

def check_video_status(video_id):
    url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    headers = {"X-Api-Key": HEYGEN_API_KEY}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["data"]


 #Predicted probability of readmission: {context['prediction']:.2%}
# --- Call LLM ---
def generate_patient_script(context):
    prompt = f"""
    You are a compassionate hospital discharge nurse explaining the risk of readmission to the patient as they await discharge.

    Patient details:
    {json.dumps(context['patient_info'].to_dict(), indent=2, default=str)}

    Admission details:
    {json.dumps(context['admission_info'].to_dict(), indent=2, default=str)}           

    Lab results from this admission:
    {json.dumps(context['latest_labs'].to_dict(), indent=2, default=str)}

    Prescribed medications from this admission:
    {json.dumps(context['latest_medications'].to_dict(), indent=2, default=str)}

   
    Readmission Prediction Risk level: {context['risk_level']}

    Contributing factors: {', '.join(f'{factor} (importance: {importance*100:.0f}%)' for factor, importance in context['top_contributing_factors'])}

    Please:
    1. Explain the prediction risk level in plain, reassuring language for a non-medical audience.
    2. Explain the contributing factors and add more color to how they relate to the given diagnosis and risk of readmission.
    3. Suggest next steps associated with the risk level.
    4. Start each script with 'Hi there, I hope you're feeling better.'  End each script with 'Take care.'

    Keep the tone empathetic, medically professional, and keep it under 250 words and 1,000 characters.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content

if "get_pred" not in st.session_state:
    st.session_state.get_pred = False
if "factors" not in st.session_state:
    st.session_state.factors_normalized = None
if "prob" not in st.session_state:
    st.session_state.prob = None
if "risk_desc" not in st.session_state:
    st.session_state.risk_desc = None
if "risk" not in st.session_state:
    st.session_state.risk = None
if "script" not in st.session_state:
    st.session_state.script = None
if "script_generated" not in st.session_state:
    st.session_state.script_generated = False

with st.container(border=True):
    verify=True
    st.subheader('Verify Patient')
    # Left column: patient selector & recent history
    col1, col2 = st.columns(2)
    with col1:
        pid = st.selectbox("Select patient ID", patients_df["patient_id"].tolist(), index=None)
    with col2:
        dob = st.date_input("Date of Birth", datetime.date.today(),min_value=datetime.date(1920, 1, 1))
    if st.button("Verify"):
        dob_entered=pd.to_datetime(dob).date()
        patient_dob=pd.to_datetime(patients_df.loc[patients_df['patient_id'] == pid, 'dob'].iloc[0]).date()
        if pid and dob_entered == patient_dob:
            st.success('Patient Verified', icon="✅")
            #verify=True
        else:
            st.error('Patient Not Verified')

if verify==True:
    with st.container(border=True):
        st.subheader('Admission Overview')
        if pid: 
            pid_admissions = admissions_df[admissions_df["patient_id"] == pid]
            admissions_df["discharge_date"] = pd.to_datetime(admissions_df["discharge_date"])
            curr_admission_id = (pid_admissions.sort_values("admission_date", ascending=False).iloc[0]["admission_id"])
            pid_info = patients_df[patients_df["patient_id"] == pid].iloc[0]
            curr_admission = admissions_df[admissions_df["admission_id"] == curr_admission_id].iloc[0]
            curr_discharge = curr_admission['discharge_date']
            curr_meds = meds_df[meds_df["admission_id"] == curr_admission_id]
            curr_labs = labs_df[labs_df["admission_id"] == curr_admission_id]
            curr_admission_display = admissions_df[admissions_df["admission_id"] == curr_admission_id].iloc[[0]].copy()
            curr_admission_display['admission_date'] = curr_admission_display['admission_date'].map(lambda x: x[:10] if isinstance(x, str) else '')
            curr_admission_display['discharge_date'] = curr_admission_display['discharge_date'].dt.strftime('%Y-%m-%d')
            med_display = curr_meds[['drug', 'dose', 'start_date', 'end_date']].copy()
            med_display['start_date'] = med_display['start_date'].apply(lambda x: x[:10] if pd.notnull(x) else '')
            med_display['end_date'] = med_display['end_date'].apply(lambda x: x[:10] if pd.notnull(x) else '')
            med_display = med_display.rename(columns={
                'drug': 'Drug Name',
                'dose': 'Dosage',
                'start_date': 'Start Date',
                'end_date': 'End Date'
            })
            lab_display = curr_labs[['lab_test', 'lab_value', 'lab_timestamp']].copy()
            lab_display['lab_timestamp'] = lab_display['lab_timestamp'].apply(lambda x: x[:10] if pd.notnull(x) else '')
            lab_display = lab_display.rename(columns={
                'lab_test': 'Test Name',
                'lab_value': 'Result',
                'lab_timestamp': 'Test Date'
            })

            col1,col2,col3,col4=st.columns(4)
            with col1:
                st.markdown(f"**Patient ID**")
                st.markdown(f"_{pid_info['patient_id']}_")
            with col2:
                st.markdown(f"**Age**")
                st.markdown(f"_{pid_info['age']}_")
            with col3:
                st.markdown(f"**Sex**")
                st.markdown(f"_{pid_info['sex']}_")
            with col4:
                st.markdown(f"**Chronic Conditions**")
                if pd.isna(pid_info['chronic_conditions']):
                    st.markdown("_None_")
                else:
                    st.markdown(f"_{pid_info['chronic_conditions']}_")
            st.divider()
            col1,col2,col3,col4=st.columns(4)
            with col1:
                st.markdown(f"**Admission Type**")
                st.markdown(f"_{curr_admission_display['admission_type'].iloc[0]}_")
            with col2:
                st.markdown(f"**Primary Diagnosis**")
                st.markdown(f"_{curr_admission_display['primary_diagnosis'].iloc[0]}_")
            with col3:               
                st.markdown(f"**Admission Date**")
                st.markdown(f"_{curr_admission_display['admission_date'].iloc[0]}_")
            with col4:
                st.markdown(f"**Discharge Date (Estimated)**")
                st.markdown(f"_{curr_admission_display['discharge_date'].iloc[0]}_")
            st.divider()
            st.markdown(f"**Lab Results**")
            st.dataframe(lab_display, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown(f"**Medications Prescribed**")
            st.dataframe(med_display, use_container_width=True, hide_index=True)
            st.divider()

            with st.expander("How we use your data"):
                st.write('The information above and from past admissions at this hospital are used for predicting your likelihood of hospital readmission, so it is important for you to review it for accuracy.')
                
    with st.container(border=True):
        st.subheader('Get Prediction')
        if pid:
            perf_window = st.slider(
            "Predict Readmission within how many days?",
            min_value=1,
            max_value=30,
            step=1,
            value=30
            )
            if st.button("Predict Readmission"):
                st.session_state.get_pred = True
                st.session_state.factors = admission_factors(pid_info, curr_admission, labs_df, meds_df)
                model=load_model()
                query = f"PREDICT COUNT(admissions.*, 0, 30, days)>0 FOR patients.patient_id='{pid}'"
                value = model.predict(query, anchor_time=curr_discharge)
                prob=float(value['True_PROB'].iloc[0])
                #insert KUMO prediction here
                st.session_state.prob = prob
                if st.session_state.prob > 0.12:
                    risk='High'
                    risk_desc="High readmission risk — Your doctor will need to review your case before discharge and may advise a longer stay, additional testing, and/or changes to prescribed medications."
                    dial_value=0.83
                elif st.session_state.prob > 0.08:
                    risk='Moderate'
                    risk_desc="Moderate risk — You will continue through the standard discharge process, but we will need to arrange a follow-up appointment within 7 days to monitor your situation."
                    dial_value=0.5
                else:
                    risk='Low'
                    risk_desc="Low risk — you will continue through the standard discharge process."
                    dial_value=0.17

                st.session_state.risk_desc = risk_desc
                st.session_state.risk = risk
                fig = go.Figure(go.Indicator(
                    mode = "gauge",
                    value = dial_value,
                    number={'suffix': "", 'valueformat': ''},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Readmission risk within {perf_window} days"},
                    gauge = {'axis': {'range': [0, 1], 'tickvals':[0.17,0.5,0.83],'ticktext':['Low','Moderate','High']},
                             'bar': {'color': "dimgray"},
                                'steps' : [
                                {'range': [0, 0.33], 'color': "lightgreen"},
                                {'range': [0.33, 0.66], 'color': "khaki"},
                                {'range': [0.66, 1], 'color': "lightcoral"}]}))
        
                st.plotly_chart(fig)
                col1,col2,col3=st.columns(3)
                with col1:
                    st.subheader("Top contributing factors")
                    if not st.session_state.factors:
                        st.write("- No strong contributors found.")
                    else:
                        for f, w in st.session_state.factors:
                            st.write(f"- **{f}** — relative importance {int(w)}%")
                with col2:
                    st.subheader("Risk Level:")
                    if st.session_state.risk =='High':
                        st.error('High')
                    elif st.session_state.risk =='Moderate':
                        st.warning('Moderate')
                    elif st.session_state.risk =='Low':
                        st.success('Low')
                with col3:
                    if st.session_state.risk == 'High':
                        st.subheader("Most Likely Readmission Diagnosis")
                        query2 = f"PREDICT LIST_DISTINCT(admissions.diagnosis_id, 0, 30, days) RANK TOP 3 FOR patients.patient_id='{pid}'"
                        df_pred = model.predict(query, anchor_time=curr_discharge)
                        next_dx = diagnosis_df[diagnosis_df['diagnosis_id']==df_pred['CLASS'].iloc[0]]['primary_diagnosis'].iloc[0]                  
                        st.write(f"{next_dx}")
            with st.expander("How we make predictions"):
                st.write('Predictions are powered by KumoRFM - a state of the art relational foundation model from Kumo AI.' \
                ' This advanced technology uses a pre-trained model informed with your data as in-context learning to make accurate predictions. Learn more at https://kumo.ai/')
                #st.code("query = f"PREDICT COUNT(admissions.*, 0, 30, days)>0 FOR patients.patient_id='{pid}'"" )
                st.image("assets/kumo_logo.jpg", width=100)

    with st.container(border=True):
        st.subheader('Watch Video')
        # --- CONFIG ---
        if pid:
            if st.session_state.get_pred:
                admission_context = {
                        "patient_info": pid_info,
                        "prediction": st.session_state.prob,
                        "top_contributing_factors": st.session_state.factors,
                        "risk_level": st.session_state.risk_desc,
                        "latest_labs": curr_labs,
                        "latest_medications": curr_meds,
                        "admission_info": curr_admission,
                    }

                # --- Streamlit UI ---
                if st.button("Generate Video") or st.session_state.script_generated:
                    if not st.session_state.script_generated:
                        openai.api_key = st.secrets["OPENAI_API_KEY"]  # Store securely in .streamlit/secrets.toml
                        st.session_state.script = generate_patient_script(admission_context)
                        st.session_state.script_generated = True
                    if st.session_state.script_generated:
                        # Get HeyGen API key from secrets.toml
                        HEYGEN_API_KEY = st.secrets["heygen"]["api_key"]
                        with st.spinner("Requesting video generation..."):
                            video_id = generate_video(st.session_state.script)
                            status = ""
                            last_status = None  # track the previous status
                            while True:
                                status_data = check_video_status(video_id)
                                status = status_data["status"]

                                if status != last_status:  # only print if status changed
                                    st.info(f"Status: {status}")
                                    last_status = status

                                if status == "completed":
                                    st.success("Video generation completed!")
                                    st.video(status_data["video_url"])
                                    break
                                elif status == "failed":
                                    st.error("Video generation failed.")
                                    st.write("Full failure details:", status_data)
                                    break
                                time.sleep(5)

    with st.container(border=True):
        st.subheader('Next Steps')
        if st.session_state.script_generated:
            if st.session_state.risk=='Low':
                st.write('Your readmission risk level is low, so you will continue with the regular discharge process.')
            elif st.session_state.risk=='Moderate':
                st.write('Your readmission risk level is moderate, so you will need to schedule a follow up appointment within the next 7 days. Select from available time and date options below:') 
                date = st.date_input("Appointment Date", datetime.date.today())
                time = st.time_input("Appointment Time", datetime.time(9, 0))
                if st.button("Book Appointment"):
                    st.success(f"Appointment booked on {date} at {time}. You will receive an email confirmation at the email address on file.")
            elif st.session_state.risk=='High':
                st.write('Your readmission risk level is high, so your doctor will need to review your case before discharge, and may advise a longer stay, additional testing, and/or changes to prescribed medications.')


