from flask import Flask, render_template, request,jsonify
from pymongo import MongoClient
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import requests
import math
import string
import random
import io
import requests
import base64
import datetime
from flask import Flask, render_template_string, request, send_file,render_template,redirect
from pymongo import MongoClient
from xhtml2pdf import pisa
from PyPDF2 import PdfWriter, PdfReader
import qrcode
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.backends import default_backend
from urllib.parse import quote, unquote
import uuid
import boto3
import os
from botocore.exceptions import ClientError
from PIL import Image
from pytesseract import pytesseract
import re
import requests
#Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from flask import Flask, render_template, request,jsonify
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pymongo
import requests
from flask import Flask, render_template, request, redirect, url_for,jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
import warnings
from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import requests
import math
import string
import random
import io
import base64
import datetime
from flask import Flask, render_template_string, request, send_file,render_template,redirect
from xhtml2pdf import pisa
from PyPDF2 import PdfWriter, PdfReader
import qrcode
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.backends import default_backend
from urllib.parse import quote, unquote
import uuid
from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image 
import numpy as np
import os
import time
from datetime import date
from flask import Flask, render_template, request

app = Flask(__name__)

#Chethine(Start)
## Connect to MongoDB
connection_string = "mongodb+srv://kasun312:kasun123@shmps.vpmeq0h.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)
db = client['shmps']
regformdata =  db['regfrom_data']
formdata = db['patient_data']
ocr = db['form_data']


# MongoDB Setup
MONGO_URI = "mongodb+srv://iresh678:iresh123@cluster0.vsmqddt.mongodb.net/?retryWrites=true&w=majority"
client4 = MongoClient(MONGO_URI)
db = client4.mydatabase
hospitals_collection = db.hospitals
pharmacy_collection = db.pharmacy


# AWS credentials and configuration
aws_access_key_id = "AKIA4D6ILQ56PY34B36R"
aws_secret_access_key = "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"
aws_region = "ap-south-1"
bucket_name = "imageshmps"

# Initialize S3 client
s3_client = boto3.client("s3", aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=aws_region)

rekognition_client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=aws_region)
#Chethine(End)

# Connect to MongoDB
connection_string = "mongodb+srv://kasun312:kasun123@shmps.vpmeq0h.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)
db = client['shmps']
collection = db['patient_data']
collection1 = db['pdf_info']
doctors_col = db['doctors']
keys_col = db['keys']
requests_col = db['requests']
patients_col = db['patient_data']

# notify.lk API configuration
NOTIFYLK_USER_ID = '25513'
NOTIFYLK_API_KEY = 'DpubzXHTqfjV6W5wUngb'

def generate_random_password(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

with open("private_key.pem", "rb") as key_file:
    private_key_pem = key_file.read()
private_key_instance = serialization.load_pem_private_key(private_key_pem, password=None, backend=default_backend())

def sign_data(data):
    signature = private_key_instance.sign(data, asym_padding.PSS(mgf=asym_padding.MGF1(hashes.SHA256()), salt_length=asym_padding.PSS.MAX_LENGTH), hashes.SHA256())
    return signature

def combine_and_encode(reference_number, signature):
    signature_b64 = base64.b64encode(signature).decode('utf-8')
    signature_b64 = quote(signature_b64)  # URL-encode the Base64 signature
    combined_str = f"{reference_number}||{signature_b64}"
    verification_url = f"http://verify.myshmps.com/verify_signature?data={combined_str}"
    return verification_url

def generate_qr_data_uri(data):
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    buffered = io.BytesIO()
    qr_image.save(buffered, format="PNG")
    qr_data_uri = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
    return qr_data_uri

def encrypt_pdf(input_stream, password):
    output = io.BytesIO()
    pdf_writer = PdfWriter()
    pdf_reader = PdfReader(input_stream)

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_writer.add_page(page)

    pdf_writer.encrypt(password)
    pdf_writer.write(output)
    output.seek(0)
    return output

    # Step 1: Data Preprocessing (Assuming you have your diabetes dataset in a CSV file)
diabetes_data = pd.read_csv("diabetes.pro.csv")

# Assuming the target column is named "diabetes" and all other columns are features
X = diabetes_data.drop(columns='Outcome')
y = diabetes_data['Outcome']

# Split the dataset into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Decision Tree Model
# Train a Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Fine-tune Decision Tree hyperparameters using GridSearchCV
param_grid = {
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10]
}

grid_search_tree = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)

# Best hyperparameters for Decision Tree
best_tree_params = grid_search_tree.best_params_

# Create the Decision Tree model with the best hyperparameters
best_decision_tree = DecisionTreeClassifier(
    max_depth=best_tree_params["max_depth"],
    min_samples_split=best_tree_params["min_samples_split"],
    random_state=42
)
best_decision_tree.fit(X_train, y_train)

# Step 3: Random Forest Model
# Train a Random Forest classifier
random_forest = RandomForestClassifier(random_state=42)

# Fine-tune Random Forest hyperparameters using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search_forest = GridSearchCV(random_forest, param_grid, cv=5, scoring='accuracy')
grid_search_forest.fit(X_train, y_train)

# Best hyperparameters for Random Forest
best_forest_params = grid_search_forest.best_params_

# Create the Random Forest model with the best hyperparameters
best_random_forest = RandomForestClassifier(
    n_estimators=best_forest_params["n_estimators"],
    max_depth=best_forest_params["max_depth"],
    min_samples_split=best_forest_params["min_samples_split"],
    random_state=42
)
best_random_forest.fit(X_train, y_train)

# Step 4: Feature Importance Analysis (Optional)
# You can analyze feature importances from both models if you want to understand important features

# Step 5: Hybrid Model Combination
# The hybrid model will combine predictions from Decision Tree and Random Forest using weighted averaging
def hybrid_predict(X):
    tree_prediction = best_decision_tree.predict_proba(X)
    forest_prediction = best_random_forest.predict_proba(X)
    
    # Adjust weights for the Decision Tree and Random Forest predictions (you can fine-tune these weights)
    weight_tree = 0.4
    weight_forest = 0.6
    
    hybrid_prediction = weight_tree * tree_prediction + weight_forest * forest_prediction
    return np.argmax(hybrid_prediction, axis=1)

# Step 6: Model Evaluation
y_pred_hybrid = hybrid_predict(X_test)
accuracy_hybrid = accuracy_score(y_test, y_pred_hybrid)
print("Hybrid Model Accuracy:", accuracy_hybrid)


@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input_id = request.form['patient_id']

        # Fetch the patient's data from MongoDB based on the entered ID
        print("Searching for patient ID:", user_input_id)
        patient_data = collection.find_one({'patient_id': int(user_input_id)})
# Print the retrieved patient_data
        print("Retrieved patient_data:", patient_data)
        
        if patient_data:
            # Extract the necessary features from the patient_data dictionary
            necessary_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            features = [patient_data[feature] for feature in necessary_features]

            # Create a DataFrame with the fetched data
            user_input_df = pd.DataFrame([features], columns=necessary_features)

            # Make a prediction using the hybrid model
            prediction = hybrid_predict(user_input_df)

            # Convert prediction to human-readable result
            result = "You have diabetic." if prediction[0] == 1 else "You have not diabetic."
            collection.update_one({'patient_id': int(user_input_id)}, {
        '$set': {
            'result': result,
            
        }
    })
            return render_template('render.html', result=result)
            
        else:
            error_msg = "Patient ID not found in the database."
            return render_template('render.html', error_msg=error_msg)

    return render_template('render.html')



# Risk factors and their weights
RISK_FACTORS = {
    'age': 0.2,
    'bmi': 0.1,
    'family_history': 1.5,
    'sedentary_lifestyle': 1.2,
    'high_blood_pressure': 1.3,
    'high_cholesterol': 1.3,
    'gestational_diabetes': 1.6,
    'glucose': 0.1,
    'blood_pressure': 0.1,
    'diabetes_pedigree_function': 1.3,
}

def calculate_risk(age, bmi, family_history, sedentary_lifestyle, high_blood_pressure,
                   high_cholesterol, gestational_diabetes, glucose, blood_pressure,
                   diabetes_pedigree_function):
    risk = 1  # base risk

# Age risk
    if age > 40:
        risk *= (1 + RISK_FACTORS['age'] * ((age - 40) // 10))

    # BMI risk
    if bmi > 25:
        risk *= (1 + RISK_FACTORS['bmi'] * (bmi - 25))

    # Family history risk
    if family_history:
        risk *= RISK_FACTORS['family_history']

    # Sedentary lifestyle risk
    if sedentary_lifestyle:
        risk *= RISK_FACTORS['sedentary_lifestyle']

    # High blood pressure risk
    if high_blood_pressure:
        risk *= RISK_FACTORS['high_blood_pressure']

    # High cholesterol risk
    if high_cholesterol:
        risk *= RISK_FACTORS['high_cholesterol']

    # Gestational diabetes risk
    if gestational_diabetes:
        risk *= RISK_FACTORS['gestational_diabetes']

    # Glucose risk
    if glucose > 7:
        risk *= (1 + RISK_FACTORS['glucose'] * (glucose - 7))

    # Blood pressure risk
    if blood_pressure > 120:
        risk *= (1 + RISK_FACTORS['blood_pressure'] * ((blood_pressure - 120) // 10))

    # Diabetes pedigree function risk
    if diabetes_pedigree_function > 0.5:
        risk *= RISK_FACTORS['diabetes_pedigree_function']

    # Normalize risk to a scale of 0 to 100
    risk = min(risk, 100)

    return risk

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabets')
def diabetes():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/ocr')
def ocr():
    return render_template('ocr.html')

@app.route('/contactus')
def contactuss():
    return render_template('contact.html')

@app.route('/regfrom')
def regform():
    return render_template('form.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/policy')
def privacy():
    return render_template('privacy.html')

@app.route('/input_data')
def input_data():
    return render_template('input.html')

@app.route('/inputs')
def inputs():
    return render_template('pred.html')

@app.route('/calculate_risk', methods=['GET'])
def calculate_diabetes_risk():
    patient_id = request.args.get('patient_id')

    if not patient_id:
        return "Patient ID not provided", 400

    patient_data = collection.find_one({'patient_id': int(patient_id)})

    if patient_data is None:
        return "Patient not found", 404

    age = int(patient_data['Age'])
    bmi = float(patient_data['BMI'])
    family_history = bool(patient_data['family_history'])
    sedentary_lifestyle = bool(patient_data['sedentary_lifestyle'])
    high_blood_pressure = bool(patient_data['high_blood_pressure'])
    high_cholesterol = bool(patient_data['high_cholesterol'])
    gestational_diabetes = bool(patient_data['gestational_diabetes'])
    glucose = float(patient_data['Glucose'])
    blood_pressure = int(patient_data['BloodPressure'])
    diabetes_pedigree_function = float(patient_data['diabetes_pedigree_function'])

    # Calculate risk using the first method
    risk_1 = calculate_risk(age, bmi, family_history, sedentary_lifestyle, high_blood_pressure,
                          high_cholesterol, gestational_diabetes, glucose, blood_pressure,
                          diabetes_pedigree_function)

    # Calculate risk percentage using the second method
    risk_2_percentage = diabetes_risk_calculator(age, bmi, family_history, sedentary_lifestyle,
                                               high_blood_pressure, high_cholesterol, gestational_diabetes,
                                               glucose, blood_pressure, diabetes_pedigree_function)
    
    if risk_2_percentage <= 30:
        risk_category = "Low Risk"
        recommendation = "Continue maintaining a healthy lifestyle and regular exercise to minimize risk. Stay hydrated and prioritize whole, unprocessed foods."
    elif risk_2_percentage <= 60:
        risk_category = "Moderate Risk"
        recommendation = "Prioritize regular exercise, weight management, and a balanced diet. Limit sugary foods and beverages and practice stress management techniques."
    else:
        risk_category = "High Risk"
        recommendation = "Consult a healthcare professional for a comprehensive assessment and guidance. Make significant changes to your diet and exercise routine. Monitor blood sugar levels if advised."

    collection.update_one({'patient_id': int(patient_id)}, {
        '$set': {
            'risk_1': risk_1,
            'Risk_precen': risk_2_percentage,
            'recommendation': recommendation
        }
    })



    return render_template('final.html', risk_1=risk_1, risk_2_percentage=risk_2_percentage,risk_category=risk_category, recommendation=recommendation)


def diabetes_risk_calculator(age, bmi, family_history, sedentary_lifestyle, high_blood_pressure,
                             high_cholesterol, gestational_diabetes, glucose, blood_pressure, 
                             diabetes_pedigree_function):
    # Define weights for each risk factor
    age_weight = 2
    bmi_weight = 3
    family_history_weight = 2
    sedentary_lifestyle_weight = 2
    high_blood_pressure_weight = 2
    high_cholesterol_weight = 1
    gestational_diabetes_weight = 2
    glucose_weight = 2
    blood_pressure_weight = 1
    diabetes_pedigree_function_weight = 1

    # Calculate the risk score for each risk factor
    age_score = age * age_weight
    bmi_score = bmi * bmi_weight
    family_history_score = family_history_weight if family_history else 0
    sedentary_lifestyle_score = sedentary_lifestyle_weight if sedentary_lifestyle else 0
    high_blood_pressure_score = high_blood_pressure_weight if high_blood_pressure else 0
    high_cholesterol_score = high_cholesterol_weight if high_cholesterol else 0
    gestational_diabetes_score = gestational_diabetes_weight if gestational_diabetes else 0
    glucose_score = glucose * glucose_weight
    blood_pressure_score = blood_pressure * blood_pressure_weight
    diabetes_pedigree_function_score = diabetes_pedigree_function * diabetes_pedigree_function_weight

    # Calculate the total risk score
    total_score = age_score + bmi_score + family_history_score + sedentary_lifestyle_score + \
                  high_blood_pressure_score + high_cholesterol_score + gestational_diabetes_score + \
                  glucose_score + blood_pressure_score + diabetes_pedigree_function_score

    # Define the maximum possible score based on all risk factors having their highest values
    max_score = (100 * age_weight) + (100 * bmi_weight) + (family_history_weight) + \
                (sedentary_lifestyle_weight) + (high_blood_pressure_weight) + (high_cholesterol_weight) + \
                (gestational_diabetes_weight) + (200 * glucose_weight) + \
                (200 * blood_pressure_weight) + (100 * diabetes_pedigree_function_weight)

    # Calculate the risk score as a percentage
    risk_percentage = (total_score / max_score) * 100

    return risk_percentage
def get_symptoms_recommendation(symptoms):
    recommendations = []
    
    if symptoms['increased_thirst_and_urination']:
        recommendations.append("Increased thirst and urination could be signs of high blood sugar. Consult a healthcare professional for assessment.")
    if symptoms['increased_hunger']:
        recommendations.append("Increased hunger might indicate changes in blood sugar levels. Monitoring is advised.")
    if symptoms['weight_loss']:
        recommendations.append("Unintended weight loss may be linked to various health issues, including diabetes. Seek medical attention if this continues.")
    if symptoms['fatigue']:
        recommendations.append("Experiencing fatigue? Check with a healthcare professional to rule out underlying causes, including diabetes.")
    if symptoms['blurred_vision']:
        recommendations.append("Blurred vision can be related to high blood sugar levels. It's recommended to consult a healthcare provider.")
    if symptoms['slow_healing_sores_or_infections']:
        recommendations.append("Slow healing of sores or infections might be related to diabetes. Get medical advice if this persists.")
    if symptoms['areas_of_darkened_skin']:
        recommendations.append("Darkened skin patches might be a sign of insulin resistance. Consult a healthcare professional for assessment.")
    
    return recommendations


# Define Recommendations for Health Parameters (BMI, Blood Pressure, Glucose)
def get_health_recommendation(bmi, blood_pressure, glucose):
    recommendations = []
    
    if bmi > 25:
        recommendations.append("Consider maintaining a healthy weight to improve overall health.")
    
    if blood_pressure > 120:
        recommendations.append("Keep blood pressure within a healthy range through diet and exercise.")
    
    if glucose > 100:
        recommendations.append("Monitor blood sugar levels and consult a healthcare professional for high glucose.")
    
    return recommendations    

def calculate_diabetes_risk(symptoms, weights):
    risk_score = sum(symptoms[symptom] * weights[symptom] for symptom in symptoms)
    return risk_score

def calculate_diabetes_risk_percen(symptoms, weights):
    risk_score = calculate_diabetes_risk(symptoms, weights)
    total_possible_score = sum(weights.values())
    risk_percen = (risk_score / total_possible_score) * 100
    return risk_percen

def determine_risk_level(risk_score):
    if risk_score <= 4:
        return "Low risk"
    elif risk_score <= 7:
        return "Medium risk"
    else:
        return "High risk"

@app.route('/symptoms_risk', methods=['POST'])
def symptoms_risk():
    patient_id = request.form.get('patient_id')

    # Fetch patient data from MongoDB
    patient_data = collection.find_one({'patient_id': int(patient_id)})
    if patient_data is None:
        return "Patient not found", 404

    # Create a symptoms dictionary and map boolean values from patient data
    symptoms = {
        "increased_thirst_and_urination": bool(patient_data.get("increased_thirst_and_urination", False)),
        "increased_hunger": bool(patient_data.get("increased_hunger", False)),
        "weight_loss": bool(patient_data.get("weight_loss", False)),
        "fatigue": bool(patient_data.get("fatigue", False)),
        "blurred_vision": bool(patient_data.get("blurred_vision", False)),
        "slow_healing_sores_or_infections": bool(patient_data.get("slow_healing_sores_or_infections", False)),
        "areas_of_darkened_skin": bool(patient_data.get("areas_of_darkened_skin", False))
    }

    weights = {
        "increased_thirst_and_urination": 1.5,
        "increased_hunger": 1.0,
        "weight_loss": 1.0,
        "fatigue": 1.0,
        "blurred_vision": 1.5,
        "slow_healing_sores_or_infections": 1.0,
        "areas_of_darkened_skin": 1.5
    }

    risk_score = calculate_diabetes_risk(symptoms, weights)
    risk_percen = calculate_diabetes_risk_percen(symptoms, weights)
    risk_level = determine_risk_level(risk_score)
   
    bmi = float(patient_data.get("bmi", 0))
    blood_pressure = int(patient_data.get("blood_pressure", 0))
    glucose = float(patient_data.get("glucose", 0))
    
    symptoms_recommendations = get_symptoms_recommendation(symptoms)
    health_recommendations = get_health_recommendation(bmi, blood_pressure, glucose)
    
    combined_recommendations = symptoms_recommendations + health_recommendations
    
    
    # ... (existing risk calculations and database updates)  
     # Update the patient's document with the newly calculated risk values
    collection.update_one({'patient_id': int(patient_id)}, {
        '$set': {
            'risk_score': risk_score,
            'sym_precentage': risk_percen,
            'risk_level': risk_level,
            'recommendations': combined_recommendations
        }
    })
    
    return render_template('newrisk.html', risk_score=risk_score, risk_percentage=risk_percen,
                           risk_level=risk_level, patient_data=patient_data,
                           symptoms_recommendations=symptoms_recommendations,
                           health_recommendations=health_recommendations,
                           combined_recommendations=combined_recommendations)
    
    
    # OpenCage Geocoding API endpoint
endpoint = 'https://api.opencagedata.com/geocode/v1/json'
api_key = '33365c941d474dc086c98c6cb87823a3'

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_location_coordinates(location):
    params = {
        'q': location,
        'key': api_key,
        'limit': 1
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lon = data['results'][0]['geometry']['lng']
        return lat, lon
    return None, None

def get_nearby_person(location, radius):
    lat1, lon1 = get_location_coordinates(location)
    if not lat1 or not lon1:
        return lat1, lon1, []

   
    persons = {
        'DR SACHITH ABHAYARATNA  MEDICAL CENTER : ASIRI MEDICAL HOSPITAL - KIRULA ROAD - COLOMBO 05': (6.8912, 79.8657),  # Example coordinates (latitude, longitude)
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : CDEM HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10 ': (6.9409, 79.8651),
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03 ': (6.9167, 79.8500),
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : GOLDEN KEY HOSPITALS LIMITED - RAJAGIRIYA ': (6.909504,79.896218),
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : HEMAS HOSPITAL - THALAWATHUGODA': (6.876349,79.935376),
            
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : HEMAS HOSPITAL - WATTALA': (6.989871,79.892709),
            'DR UDITHA BULUGAHAPITIYA MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03': (6.9167, 79.8500),
            'DR UDITHA BULUGAHAPITIYA MEDICAL CENTER : KINGS HOSPITAL (FORMER OASIS) - COLOMBO 05': (6.8912, 79.8657),
            
            'DR SAMANTHI COORAY MEDICAL CENTER :HEMAS HOSPITAL - WATTALA': (6.989871,79.892709),
            'DR SAMANTHI COORAY MEDICAL CENTER :KINGS HOSPITAL (FORMER OASIS) - COLOMBO 05': (6.8912, 79.8657),
            'DR SAMANTHI COORAY MEDICAL CENTER :MEDIHELP HOSPITAL - HORANA': (6.716628,80.061949),
           
            
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : ASIRI CENTRAL HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10': (6.9409, 79.8651),
            'DR CHAMINDA GARUSINGHE': (7.5368871,80.9129994),
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : DR. NEVILLE FERNANDO TEACHING HOSPITAL - MALABE': (6.904072,79.954619),
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : NAWALOKA MEDICAL CENTER - KIRIBATHGODA': (6.97806,79.927391),
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : NAWALOKA HOSPITAL - NEGOMBO': (7.26542,80.5839),
            
            'DR NALIN D GUNARATNE MEDICAL CENTER :ASIRI MEDICAL HOSPITAL - KIRULA ROAD - COLOMBO 05': (6.8912, 79.8657),
            'DR NALIN D GUNARATNE MEDICAL CENTER :DR. NEVILLE FERNANDO TEACHING HOSPITAL - MALABE': (6.904072,79.954619),
            
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03 ': (6.9167, 79.8500),
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : HEMAS HOSPITAL - WATTALA ': (6.989871,79.892709),
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : MIRACLE CHANNELING CENTRE (PVT) LTD - KURUNEGALA': (7.487046,80.364908),
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : NAWALOKA MEDICARE - KURUNEGALA': (7.487046,80.364908),

            
            'Prof. PRASAD KATULANDA MEDICAL CENTER : ASIRI CENTRAL HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10': (6.9409, 79.8651),
            'Prof. PRASAD KATULANDA MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03': (6.9167, 79.8500),
            'Prof. PRASAD KATULANDA MEDICAL CENTER : WESTERN HOSPITAL - COLOMBO 08': (6.9344, 79.8737),
            
            'DR(MRS) DIMUTHU T. MUTHUKUDA MEDICAL CENTER : HEMAS HOSPITAL - THALAWATHUGODA': (6.88297,79.90708),
            
            'DR SAJITH SIYAMBALAPITIYA MEDICAL CENTER :DURDANS HOSPITAL - COLOMBO 03': (6.9167, 79.8500),
            
            'DR NOEL SOMASUNDARAM MEDICAL CENTER : ASIRI CENTRAL HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10 ': (6.9409, 79.8651),
            
            'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER : ASIRI MEDICAL HOSPITAL - KIRULA ROAD - COLOMBO 05  ': (6.8912, 79.8657),
            'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER : ESESS HOSPITAL - KADAWATHA  ': (7.001966,79.951267),
            'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER : MEDIHELP HOSPITAL - MOUNT LAVINIA  ': (6.83125,79.862858),
           
           'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER :  KINGS HOSPITAL (FORMER OASIS) - COLOMBO 05  ': (6.8912, 79.8657),
           
           'DR(MRS) DR VASANTHA S HETTIARACHCHI   MEDICAL CENTER : ASIRI SURGICAL HOSPITAL - KIRIMANDALA MW - COLOMBO 05  ': (6.8912, 79.8657),
           'DR(MRS) DR VASANTHA S HETTIARACHCHI   MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03  ': (6.9167, 79.8500),
           'DR(MRS) DR VASANTHA S HETTIARACHCHI   MEDICAL CENTER : HEALTHY LIFE CLINIC - 139, DHARAMAPALA MAWATHA, COLOMBO 07  ': (6.9161, 79.8640),
           
           'DR(MRS) DR Z JAMALDEEN  MEDICAL CENTER :  NAWALOKA MEDICARE - KURUNEGALA  ': (7.487046,80.364908),
           
           'DR(MRS) DR MOHAN JAYATILAKE  MEDICAL CENTER : ASIRI SURGICAL HOSPITAL - KIRIMANDALA MW - COLOMBO 05 ': (6.8912, 79.8657),
           'DR(MRS) DR MOHAN JAYATILAKE  MEDICAL CENTER : DR. NEVILLE FERNANDO TEACHING HOSPITAL - MALABE ': (6.904072,79.954619),
          
           'DR(MRS) DR JAYANTHIMALA JAYAWARDENA  MEDICAL CENTER : ASIRI SURGICAL HOSPITAL - KIRIMANDALA MW - COLOMBO 05 ': (6.8912, 79.8657),
           'DR(MRS) DR JAYANTHIMALA JAYAWARDENA  MEDICAL CENTER : WESTERN HOSPITAL NEW WING - COLOMBO 08 ': (6.9344, 79.8737),
           
           'DR(MRS) DR(MRS) SUBHASHINI JAYAWICKREME  MEDICAL CENTER : CHANNELED CONSULTATION CENTER, KANDY (CCC KANDY) ': (7.29236948711015, 80.63215939533397),
           
           'DR(MRS) DR PRIYANTHA KANNANGARA  MEDICAL CENTER : CHANNELED CONSULTATION CENTER, KANDY (CCC KANDY) ': (7.29236948711015, 80.63215939533397),
           'DR(MRS) DR PRIYANTHA KANNANGARA  MEDICAL CENTER : SUWASEVANA HOSPITAL - KANDY) ': (7.281547112062025, 80.62047316649738),
           
           
           'DR WASANTHA KAPUWATTA  MEDICAL CENTER :MELSTA HOSPITALS(FORMER BROWNS HOSPITALS) - RAGAMA) ': (7.028106, 79.920623),
           
           'DR AJITH KULARATHNA  MEDICAL CENTER : CHANNELED CONSULTATION CENTER, KANDY (CCC KANDY) ': (7.29236948711015, 80.63215939533397),
           
           'DR A.S.L LIYANARACHCHI  MEDICAL CENTER : MEDIHELP HOSPITAL - BANDARAGAMA ': (6.734513449318712, 79.99364903367851),
           'DR A.S.L LIYANARACHCHI  MEDICAL CENTER : MEDIHELP HOSPITAL - MATHUGAMA ': (6.522718534013628, 80.1142648844469),
           'DR A.S.L LIYANARACHCHI  MEDICAL CENTER : NEW PHILIP HOSPITALS - KALUTARA ': (6.579488049707041, 79.96290026649329),
           
           'DR A.B.M.FAIZAL NAWALAPITIYA   ': (7.0558938,80.5350145 ),
           
           'DR M.Z.F.ZEEN  WATHTHEGAMA  ': (6.7946957,81.5083199 ),
           
           'DR M.M.ZIYARD  WATHTHEGAMA  ': (6.7946957,81.5083199 ),
           
           'DR C.SRI DEVA  MEDICAL CENTER : Medical & Dental Clinic': (9.663187975161327, 80.02533649131068),
           
           'DR C.SRI DEVA  MEDICAL CENTER : Mediquick,No.514/2.Hospital road, Jaffna': (9.667091073232164, 80.01087879535338),
           
           'DR U.RAJAPAKSE KULIYAPITIYA  ': (7.32910834123194, 80.02452745712763 ),
           
           'DR B.KARUNARATNE KADUGANNAWA ': (7.2553264190055975, 80.51585909381734 ),
           
           'DR A.J.JAMEEL MEDICAL CENTER :Aloka Private Hospital, Good Shed Road, Ratnapura ': (6.682163461186225, 80.40340748183985 ),
           
           'DR W.D.ARIYADASA MATHARA ': (5.97186245436438, 80.69574941647325 ),
           
           'DR K.L.E.AMERASEKERA MEDICAL CENTER :LIFELINE FAMILY PRACTICE. Bindunuwewa,Bandarawela': (6.829809262202222, 80.99223554355243 ),
           
           'DR J.C.A.WIJERATNE VEYANGODA ': (7.155780191564204, 80.05662128870165 ),
           
           'DR R.I.K.KARUNATHILLAKE GAMPOLA ': (7.166686733039956, 80.57173359945679),
           
           'DR J.S.ANURASIRI PALMADULLA ': (6.633317072289015, 80.52334672942273),
           
           'DR K.L.E.AMERASEKERA MEDICAL CENTER : Suwasevana Dispensary': (6.952898016229476, 80.78180408279609 ),
           
           'DR R.D.DE.S.GINIGE GALLE': (6.032984955309647, 80.21518731551234),
           
           'DR RADIKA SUGATHAPALA AMBALANGODA': (6.240724510979678, 80.05398221569772),
           
           'DR RADIKA SUGATHAPALA MAWANELLA': (7.2523113,80.4468038 ),
           
           'DR TITUS FERNANDO CHILLAW': (7.563805398402483, 79.80240928725684 ),
           
           'DR S.J.V.COORAY THISSAHMAHARAMA': (6.278507088638997, 81.28535266699866 ),
           
           'DR D.M.ABEYSEKERA DIVLAPITIYA': (7.229835782402614, 80.01744571362437),
           
           'Dr. CHATHURI JAYAWARDHANA  MEDICAL CENTER : NAWALOKA MEDICARE - GAMPAHA ': (7.092085320173964, 80.00030885115021),
           
           'DR (Ms.) DULANI KOTTAHACHCHI  MEDICAL CENTER : NAWALOKA MEDICARE - GAMPAHA ': (7.092085320173964, 80.00030885115021),
           
           'DR MAHESHI AMARAWARDENA  MEDICAL CENTER :  DURDANS MEDICAL CENTER - ANURADHAPURA ': (8.325429986658794, 80.41387083806947),
            
           'DR CHATHURI JAYAWARDHANA MEDICAL CENTER :  OSRO HOSPITALS - KEGALLE ': (7.247732996359117, 80.34812817813214),
           
           'DR VAJIRA JAYASINGHE MEDICAL CENTER : KUMUDU HOSPITAL- MATALE ': (7.464534816328399, 80.62536456464306),
           
           'DR ACHINI WIJESINGHE MEDICAL CENTER : PERLYN HOSPITAL - BADULLA ': (6.990945716187946, 81.0530705088226),
    }

    nearby_persons = []
    for person, (lat2, lon2) in persons.items():
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        if distance <= radius:
            nearby_persons.append({"name": person, "lat": lat2, "lon": lon2, "distance": distance})

    return lat1, lon1, nearby_persons

DEFAULT_RADIUS = 3  # Default radius in kilometers
RADIUS_INCREMENT = 5  # Increment in kilometers


@app.route('/manual', methods=['GET', 'POST'])
def manual():
    if request.method == 'POST':
        location = request.form['location']
        radius = DEFAULT_RADIUS
        while True:
            lat, lon, nearby_persons = get_nearby_person(location, radius)
            if nearby_persons:
                break  # Stop increasing radius if at least one person is found
            radius += RADIUS_INCREMENT
        return render_template('lom.html', nearby_persons=nearby_persons, lat=lat, lon=lon, radius=radius)
    return render_template('man.html')


# OpenCage Geocoding API endpoint
endpoint = 'https://api.opencagedata.com/geocode/v1/json'
api_key = '33365c941d474dc086c98c6cb87823a3'

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_location_coordinates(location):
    params = {
        'q': location,
        'key': api_key,
        'limit': 1
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lon = data['results'][0]['geometry']['lng']
        return lat, lon
    return None, None

def get_nearby_persons(location, initial_radius, radius_increment):
    lat1, lon1 = location
    radius = initial_radius

    while True:
        persons = {
             'DR SACHITH ABHAYARATNA  MEDICAL CENTER : ASIRI MEDICAL HOSPITAL - KIRULA ROAD - COLOMBO 05': (6.8912, 79.8657),  # Example coordinates (latitude, longitude)
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : CDEM HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10 ': (6.9409, 79.8651),
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03 ': (6.9167, 79.8500),
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : GOLDEN KEY HOSPITALS LIMITED - RAJAGIRIYA ': (6.909504,79.896218),
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : HEMAS HOSPITAL - THALAWATHUGODA': (6.876349,79.935376),
            
            'DR SACHITH ABHAYARATNA  MEDICAL CENTER : HEMAS HOSPITAL - WATTALA': (6.989871,79.892709),
            'DR UDITHA BULUGAHAPITIYA MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03': (6.9167, 79.8500),
            'DR UDITHA BULUGAHAPITIYA MEDICAL CENTER : KINGS HOSPITAL (FORMER OASIS) - COLOMBO 05': (6.8912, 79.8657),
            
            'DR SAMANTHI COORAY MEDICAL CENTER :HEMAS HOSPITAL - WATTALA': (6.989871,79.892709),
            'DR SAMANTHI COORAY MEDICAL CENTER :KINGS HOSPITAL (FORMER OASIS) - COLOMBO 05': (6.8912, 79.8657),
            'DR SAMANTHI COORAY MEDICAL CENTER :MEDIHELP HOSPITAL - HORANA': (6.716628,80.061949),
           
            
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : ASIRI CENTRAL HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10': (6.9409, 79.8651),
            'DR CHAMINDA GARUSINGHE': (7.5368871,80.9129994),
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : DR. NEVILLE FERNANDO TEACHING HOSPITAL - MALABE': (6.904072,79.954619),
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : NAWALOKA MEDICAL CENTER - KIRIBATHGODA': (6.97806,79.927391),
            'DR CHAMINDA GARUSINGHE MEDICAL CENTER : NAWALOKA HOSPITAL - NEGOMBO': (7.26542,80.5839),
            
            'DR NALIN D GUNARATNE MEDICAL CENTER :ASIRI MEDICAL HOSPITAL - KIRULA ROAD - COLOMBO 05': (6.8912, 79.8657),
            'DR NALIN D GUNARATNE MEDICAL CENTER :DR. NEVILLE FERNANDO TEACHING HOSPITAL - MALABE': (6.904072,79.954619),
            
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03 ': (6.9167, 79.8500),
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : HEMAS HOSPITAL - WATTALA ': (6.989871,79.892709),
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : MIRACLE CHANNELING CENTRE (PVT) LTD - KURUNEGALA': (7.487046,80.364908),
            'DR SHYAMINDA KAHANDAWA MEDICAL CENTER : NAWALOKA MEDICARE - KURUNEGALA': (7.487046,80.364908),

            
            'Prof. PRASAD KATULANDA MEDICAL CENTER : ASIRI CENTRAL HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10': (6.9409, 79.8651),
            'Prof. PRASAD KATULANDA MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03': (6.9167, 79.8500),
            'Prof. PRASAD KATULANDA MEDICAL CENTER : WESTERN HOSPITAL - COLOMBO 08': (6.9344, 79.8737),
            
            'DR(MRS) DIMUTHU T. MUTHUKUDA MEDICAL CENTER : HEMAS HOSPITAL - THALAWATHUGODA': (6.88297,79.90708),
            
            'DR SAJITH SIYAMBALAPITIYA MEDICAL CENTER :DURDANS HOSPITAL - COLOMBO 03': (6.9167, 79.8500),
            
            'DR NOEL SOMASUNDARAM MEDICAL CENTER : ASIRI CENTRAL HOSPITAL - NORRIS CANAL ROAD-COLOMBO 10 ': (6.9409, 79.8651),
            
            'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER : ASIRI MEDICAL HOSPITAL - KIRULA ROAD - COLOMBO 05  ': (6.8912, 79.8657),
            'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER : ESESS HOSPITAL - KADAWATHA  ': (7.001966,79.951267),
            'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER : MEDIHELP HOSPITAL - MOUNT LAVINIA  ': (6.83125,79.862858),
           
           'DR NIRANJALA MEEGODA WIDANEGE   MEDICAL CENTER :  KINGS HOSPITAL (FORMER OASIS) - COLOMBO 05  ': (6.8912, 79.8657),
           
           'DR(MRS) DR VASANTHA S HETTIARACHCHI   MEDICAL CENTER : ASIRI SURGICAL HOSPITAL - KIRIMANDALA MW - COLOMBO 05  ': (6.8912, 79.8657),
           'DR(MRS) DR VASANTHA S HETTIARACHCHI   MEDICAL CENTER : DURDANS HOSPITAL - COLOMBO 03  ': (6.9167, 79.8500),
           'DR(MRS) DR VASANTHA S HETTIARACHCHI   MEDICAL CENTER : HEALTHY LIFE CLINIC - 139, DHARAMAPALA MAWATHA, COLOMBO 07  ': (6.9161, 79.8640),
           
           'DR(MRS) DR Z JAMALDEEN  MEDICAL CENTER :  NAWALOKA MEDICARE - KURUNEGALA  ': (7.487046,80.364908),
           
           'DR(MRS) DR MOHAN JAYATILAKE  MEDICAL CENTER : ASIRI SURGICAL HOSPITAL - KIRIMANDALA MW - COLOMBO 05 ': (6.8912, 79.8657),
           'DR(MRS) DR MOHAN JAYATILAKE  MEDICAL CENTER : DR. NEVILLE FERNANDO TEACHING HOSPITAL - MALABE ': (6.904072,79.954619),
          
           'DR(MRS) DR JAYANTHIMALA JAYAWARDENA  MEDICAL CENTER : ASIRI SURGICAL HOSPITAL - KIRIMANDALA MW - COLOMBO 05 ': (6.8912, 79.8657),
           'DR(MRS) DR JAYANTHIMALA JAYAWARDENA  MEDICAL CENTER : WESTERN HOSPITAL NEW WING - COLOMBO 08 ': (6.9344, 79.8737),
           
           'DR(MRS) DR(MRS) SUBHASHINI JAYAWICKREME  MEDICAL CENTER : CHANNELED CONSULTATION CENTER, KANDY (CCC KANDY) ': (7.29236948711015, 80.63215939533397),
           
           'DR(MRS) DR PRIYANTHA KANNANGARA  MEDICAL CENTER : CHANNELED CONSULTATION CENTER, KANDY (CCC KANDY) ': (7.29236948711015, 80.63215939533397),
           'DR(MRS) DR PRIYANTHA KANNANGARA  MEDICAL CENTER : SUWASEVANA HOSPITAL - KANDY) ': (7.281547112062025, 80.62047316649738),
           
           
           'DR WASANTHA KAPUWATTA  MEDICAL CENTER :MELSTA HOSPITALS(FORMER BROWNS HOSPITALS) - RAGAMA) ': (7.028106, 79.920623),
           
           'DR AJITH KULARATHNA  MEDICAL CENTER : CHANNELED CONSULTATION CENTER, KANDY (CCC KANDY) ': (7.29236948711015, 80.63215939533397),
           
           'DR A.S.L LIYANARACHCHI  MEDICAL CENTER : MEDIHELP HOSPITAL - BANDARAGAMA ': (6.734513449318712, 79.99364903367851),
           'DR A.S.L LIYANARACHCHI  MEDICAL CENTER : MEDIHELP HOSPITAL - MATHUGAMA ': (6.522718534013628, 80.1142648844469),
           'DR A.S.L LIYANARACHCHI  MEDICAL CENTER : NEW PHILIP HOSPITALS - KALUTARA ': (6.579488049707041, 79.96290026649329),
           
           'DR A.B.M.FAIZAL NAWALAPITIYA   ': (7.0558938,80.5350145 ),
           
           'DR M.Z.F.ZEEN  WATHTHEGAMA  ': (6.7946957,81.5083199 ),
           
           'DR M.M.ZIYARD  WATHTHEGAMA  ': (6.7946957,81.5083199 ),
           
           'DR C.SRI DEVA  MEDICAL CENTER : Medical & Dental Clinic': (9.663187975161327, 80.02533649131068),
           
           'DR C.SRI DEVA  MEDICAL CENTER : Mediquick,No.514/2.Hospital road, Jaffna': (9.667091073232164, 80.01087879535338),
           
           'DR U.RAJAPAKSE KULIYAPITIYA  ': (7.32910834123194, 80.02452745712763 ),
           
           'DR B.KARUNARATNE KADUGANNAWA ': (7.2553264190055975, 80.51585909381734 ),
           
           'DR A.J.JAMEEL MEDICAL CENTER :Aloka Private Hospital, Good Shed Road, Ratnapura ': (6.682163461186225, 80.40340748183985 ),
           
           'DR W.D.ARIYADASA MATHARA ': (5.97186245436438, 80.69574941647325 ),
           
           'DR K.L.E.AMERASEKERA MEDICAL CENTER :LIFELINE FAMILY PRACTICE. Bindunuwewa,Bandarawela': (6.829809262202222, 80.99223554355243 ),
           
           'DR J.C.A.WIJERATNE VEYANGODA ': (7.155780191564204, 80.05662128870165 ),
           
           'DR R.I.K.KARUNATHILLAKE GAMPOLA ': (7.166686733039956, 80.57173359945679),
           
           'DR J.S.ANURASIRI PALMADULLA ': (6.633317072289015, 80.52334672942273),
           
           'DR K.L.E.AMERASEKERA MEDICAL CENTER : Suwasevana Dispensary': (6.952898016229476, 80.78180408279609 ),
           
           'DR R.D.DE.S.GINIGE GALLE': (6.032984955309647, 80.21518731551234),
           
           'DR RADIKA SUGATHAPALA AMBALANGODA': (6.240724510979678, 80.05398221569772),
           
           'DR RADIKA SUGATHAPALA MAWANELLA': (7.2523113,80.4468038 ),
           
           'DR TITUS FERNANDO CHILLAW': (7.563805398402483, 79.80240928725684 ),
           
           'DR S.J.V.COORAY THISSAHMAHARAMA': (6.278507088638997, 81.28535266699866 ),
           
           'DR D.M.ABEYSEKERA DIVLAPITIYA': (7.229835782402614, 80.01744571362437),
           
           'Dr. CHATHURI JAYAWARDHANA  MEDICAL CENTER : NAWALOKA MEDICARE - GAMPAHA ': (7.092085320173964, 80.00030885115021),
           
           'DR (Ms.) DULANI KOTTAHACHCHI  MEDICAL CENTER : NAWALOKA MEDICARE - GAMPAHA ': (7.092085320173964, 80.00030885115021),
           
           'DR MAHESHI AMARAWARDENA  MEDICAL CENTER :  DURDANS MEDICAL CENTER - ANURADHAPURA ': (8.325429986658794, 80.41387083806947),
            
           'DR CHATHURI JAYAWARDHANA MEDICAL CENTER :  OSRO HOSPITALS - KEGALLE ': (7.247732996359117, 80.34812817813214),
           
           'DR VAJIRA JAYASINGHE MEDICAL CENTER : KUMUDU HOSPITAL- MATALE ': (7.464534816328399, 80.62536456464306),
           
           'DR ACHINI WIJESINGHE MEDICAL CENTER : PERLYN HOSPITAL - BADULLA ': (6.990945716187946, 81.0530705088226),
        }

        nearby_persons = []
        for person, (lat2, lon2) in persons.items():
            distance = calculate_distance(lat1, lon1, lat2, lon2)
            if distance <= radius:
                nearby_persons.append({"name": person, "lat": lat2, "lon": lon2, "distance": distance})

        if not nearby_persons:
            radius += radius_increment
        else:
            break

    return lat1, lon1, nearby_persons, radius

@app.route('/gps', methods=['GET', 'POST'])
def gps():
    if request.method == 'POST':
        location = request.form['location']
        lat, lon = map(float, location.split(","))
        initial_radius = 2.0
        radius_increment = 5.0
        lat, lon, nearby_persons, new_radius = get_nearby_persons((lat, lon), initial_radius, radius_increment)
        return render_template('results.html', nearby_persons=nearby_persons, lat=lat, lon=lon, radius=new_radius)
    return render_template('loc.html')


@app.route('/pdf')
def pdf():
    return render_template('pdf.html')

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    patient_id = request.form.get('patient_id')

    if patient_id:
        patient_data = collection.find_one({'patient_id': int(patient_id)})
        if not patient_data:
            return "Patient not found."

        # Generate QR code data URI
        reference_number = ''.join(random.choices(string.digits, k=8))
        
        # Render the HTML template with patient data
        with open('templates/patient_report_template.html', 'r') as template_file:
            template = template_file.read()
        rendered_html = render_template_string(template, 
                                               patient_id=patient_data['patient_id'], 
                                               name=patient_data['name'], 
                                               glucose=patient_data['Glucose'], 
                                               age=patient_data['Age'], 
                                               blood_pressure=patient_data['BloodPressure'],
                                               recommendation=patient_data['recommendations'],
                                               risk_percentage=patient_data['Risk_precen'],
                                               status=patient_data['result'],
                                               risk_5=patient_data['risk_1'],
                                               risk_sym=patient_data['sym_precentage'],
                                               level=patient_data['risk_level'],
                                               date=datetime.datetime.now().strftime('%Y-%m-%d'))

        # Convert rendered HTML to PDF
        pdf_buffer = io.BytesIO()
        pisa.CreatePDF(rendered_html, dest=pdf_buffer)

        # Compute hash of the PDF
        pdf_buffer.seek(0)
        pdf_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
        pdf_hash.update(pdf_buffer.getvalue())
        pdf_hash = pdf_hash.finalize()

        # Digitally sign the hash
        signature = sign_data(pdf_hash)

        # Store reference number and signed hash in the database
        pdf_info = {'reference_number': reference_number, 'hash': pdf_hash}
        collection1.insert_one(pdf_info)

        # Generate verification URL for QR code
        verification_url = combine_and_encode(reference_number, signature)
        qr_data_uri = generate_qr_data_uri(verification_url)

        # Embed the QR code in the PDF
        rendered_html_with_qr = render_template_string(template, 
                                                       patient_id=patient_data['patient_id'], 
                                                       name=patient_data['name'], 
                                                       glucose=patient_data['Glucose'],
                                                       age=patient_data['Age'], 
                                                       blood_pressure=patient_data['BloodPressure'],
                                                       recommendation=patient_data['recommendations'], 
                                                       risk_percentage=patient_data['Risk_precen'],
                                                       status=patient_data['result'],
                                                       risk_5=patient_data['risk_1'],
                                                       risk_sym=patient_data['sym_precentage'],
                                                       level=patient_data['risk_level'],
                                                       date=datetime.datetime.now().strftime('%Y-%m-%d'),
                                                       qr_data_uri=qr_data_uri)
        pdf_buffer_with_qr = io.BytesIO()
        pisa.CreatePDF(rendered_html_with_qr, dest=pdf_buffer_with_qr)

        # Encrypt the PDF
        password = generate_random_password()
        encrypted_pdf_buffer = encrypt_pdf(pdf_buffer_with_qr, password)
        
        phone_number = patient_data.get('phone_number', None) # Assuming 'phone_number' is the field name in the database
        if phone_number:
            send_password_via_sms(phone_number, password)
        else:
            print(f"No phone number found for patient ID: {patient_id}")

        return send_file(
            encrypted_pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'{patient_id}_{reference_number}report.pdf'
        )
    else:
        return "Please enter a patient ID."

@app.route('/verify_signature')
def verify_signature():
    data = unquote(request.args.get('data'))  # URL-decode the data
    reference_number, signature_b64 = data.split('||')
    padding_needed = len(signature_b64) % 4  # Correcting padding if needed
    if padding_needed:
        signature_b64 += "=" * (4 - padding_needed)
    signature = base64.b64decode(signature_b64)
    pdf_info = collection1.find_one({'reference_number': reference_number})
    stored_hash = pdf_info['hash']
    public_key_instance = private_key_instance.public_key()
    try:
        public_key_instance.verify(signature, stored_hash, asym_padding.PSS(mgf=asym_padding.MGF1(hashes.SHA256()), salt_length=asym_padding.PSS.MAX_LENGTH), hashes.SHA256())
        return redirect(f'/result_page?status=valid&signature={signature_b64}&hash={stored_hash}&ref={reference_number}')
        
    except Exception as e:
        print("Verification Exception:", str(e))
        return redirect('/result_page?status=invalid')
    
@app.route('/result_page')
def result_page():
    status = request.args.get('status')
    return render_template('re.html', status=status)

def send_password_via_sms(phone_number, password):
    url = "https://app.notify.lk/api/v1/send"
    params = {
        "user_id": NOTIFYLK_USER_ID,
        "api_key": NOTIFYLK_API_KEY,
        "sender_id": "NotifyDEMO",
        "to": phone_number,
        "message": f"Use this password to unlock your report : {password}"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("SMS sent successfully")
    else:
        print("Failed to send SMS")
        print("Response content:", response.content)

# notify.lk API endpoint and your API Key
NOTIFY_ENDPOINT = "https://app.notify.lk/api/v1/send"
API_KEY = "DpubzXHTqfjV6W5wUngb"  # Corrected API_KEY format

def send_smss2(phone_number, message):
    data = {
        "user_id": "25513",
        "api_key": API_KEY,
        "sender_id": "NotifyDEMO",
        "to": phone_number,
        "message": message
    }
    response = requests.post(NOTIFY_ENDPOINT, data=data)
    
    # Check if the response indicates success or failure
    if response.status_code != 200:
        print(f"Error sending SMS: {response.text}")
        return None
    
    response_data = response.json()
    if 'status' in response_data and response_data['status'] != "success":
        print(f"Error from notify.lk: {response_data['message']}")
        
    return response_data

@app.route('/doctor')
def doc():
    return render_template('doctor.html')

@app.route('/request_patient_details', methods=['GET', 'POST'])
def request_patient_details_page():
    if request.method == 'POST':
        data = request.form
        patient = patients_col.find_one({"patient_id": int(data["patient_id"])})
        doctor = doctors_col.find_one({"doctor_id": int(data["doctor_id"])})

        if not patient or not doctor:
            return jsonify({"message": "Patient or Doctor not found!"}), 400
        
        request_details = {
            "patient_id": int(data["patient_id"]),
            "doctor_id": int(data["doctor_id"]),
            "doctor_name": doctor['name']
        }
        requests_col.insert_one(request_details)
        
        accept_reject_url = f"http://verify.myshmps.com/accept_request/{data['patient_id']}/{data['doctor_id']}"
        sms_content = (f"Doctor {doctor['name']} ({doctor['phone_number']}) is requesting your details. "
                       f"Click here to accept or reject: {accept_reject_url}")
        send_smss2(patient['phone_number'], sms_content)

        return redirect("/show_popup1")
    return render_template('home.html')

@app.route('/accept_request/<patient_id>/<doctor_id>', methods=['GET', 'POST'])
def accept_request_portal(patient_id, doctor_id):
    if request.method == 'POST':
        unique_key = str(uuid.uuid4())
          # Fetch the patient's name from patients_col
        patient = patients_col.find_one({"patient_id": int(patient_id)})
        patient_name = "Unknown"
        if patient and "name" in patient:
            patient_name = patient["name"]
            
        keys_col.insert_one({"patient_id": int(patient_id), "key": unique_key,"doctor_id":int(doctor_id),"name":patient_name})
        
        # Delete the request record from requests_col
        requests_col.delete_one({"patient_id": int(patient_id), "doctor_id": int(doctor_id)})
        # Create a URL with the unique key
        unique_key_url = f"http://verify.myshmps.com/doctor_dashboard/{doctor_id}"
        
        doctor_data = doctors_col.find_one({"doctor_id": int(doctor_id)})
        sms_content = f"{patients_col.find_one({'patient_id': int(patient_id)})['name']} has accepted your request. Here is the link you can use this link one time.You can retrieve the patient's details by visiting: {unique_key_url}"
        send_smss2(doctor_data['phone_number'], sms_content)

        return redirect("/show_popup")
    
    return render_template('accept.html', patient_id=patient_id, doctor_id=doctor_id)


@app.route('/show_popup')
def show_popup():
    return '''
    <html>
        <head>
            <script type="text/javascript">
                window.onload = function() {
                    alert("Request accepted!");
                    // Set the reload flag in localStorage
                    localStorage.setItem("reload", "true");
                    setTimeout(function() {
                        window.history.go(-1);
                    }, 1000);  // 1000 milliseconds = 1 second
                }
            </script>
        </head>
        <body>
            <!-- Your HTML content here -->
        </body>
    </html>
    '''

@app.route('/show_popup1')
def show_popup1():
    return '''
    <html>
        <head>
            <script type="text/javascript">
                window.onload = function() {
                    alert("Request has sent sucessfully!");
                    // Set the reload flag in localStorage
                    localStorage.setItem("reload", "true");
                    setTimeout(function() {
                        window.history.go(-1);
                    }, 1000);  // 1000 milliseconds = 1 second
                }
            </script>
        </head>
        <body>
            <!-- Your HTML content here -->
        </body>
    </html>
    '''
    
@app.route('/reject_request/<patient_id>/<doctor_id>', methods=['POST'])
def reject_request(patient_id, doctor_id):
    # Delete the request record from requests_col
    requests_col.delete_one({"patient_id": int(patient_id), "doctor_id": int(doctor_id)})

    return jsonify({"message": "Request rejected!"})

@app.route('/request_dashboard/<patient_id>', methods=['GET'])
def request_dashboard(patient_id):
    requests = requests_col.find({"patient_id": int(patient_id)})  # Retrieve requests for the specific patient
    return render_template('request_dashboard.html', requests=requests, patient_id=patient_id)

@app.route('/doctor_dashboard/<doctor_id>', methods=['GET'])
def doctor_dashboard(doctor_id):
    assigned_requests = list(keys_col.find({"doctor_id": int(doctor_id)}))
    return render_template('doctor_dashboard.html', assigned_requests=assigned_requests)

@app.route('/retrieve_details/<unique_key>', methods=['GET', 'POST'])
def retrieve_patient_details(unique_key):
    patient = None
    key_entry = keys_col.find_one({"key": unique_key})

    if key_entry:
        patient = patients_col.find_one({"patient_id": int(key_entry["patient_id"])})
        
        # Delete the unique key from the DB to ensure one-time use
        keys_col.delete_one({"key": unique_key})
    else:
        return "Invalid or expired url", 400

    return render_template('rat.html', patient=patient)



#chethine component

# Load CSV data into a Pandas DataFrame
dataset = pd.read_csv('Dataset.csv')  # Provide the path to your CSV file


# Remove the  rows with null values
data = dataset.dropna()


#List of simple to collect
features = ["age",
            "weight",
            "height",
            "gender"
            "bmi"]

#Header in the hardcoded order
dataset_header = features + ["alcoholIntakeType", "tobaccoUsage", "physicalActiveType",
                             "physicalActiveness", "genericSuspectability", "results"]

#Replace null values with empty strings
null_values = dataset.isna().sum() #Get the total of null values in each column
dataset = dataset.replace(np.nan, '', regex=True ) #Replace with empty strings

index_names = dataset[dataset['age'] == '#NAME?'].index
dataset.drop(index_names, inplace = True)

#Convert binary to int
dataset["physicalActiveness"] = dataset["physicalActiveness"].astype(int) #physicalActiveness represeants number of hours a user spent on doing exercise on a week. Then we assign
                                                                          #physical activeness category to the patient based on the no of hours.

X = dataset.drop('results', axis=1)
y = dataset['results']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2)

#Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Create an instance of the QDA classifier
qda = QuadraticDiscriminantAnalysis()

# Fit the classifier to the training data
qda.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = qda.predict(X_test)

from sklearn.metrics import accuracy_score

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Use the SVM model to make predictions
svm_predictions = svm_model.predict(X_test)

# Use the QDA model to make predictions
qda_predictions = qda.predict(X_test)

# Use the random forest model to make predictions
rf_predictions = clf.predict(X_test)


class OCR:
    def __init__(self):
        self.path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
    def extract(self, filename):
        try:
            pytesseract.tesseract_cmd = self.path
            text = pytesseract.image_to_string(filename)
            return text
        except Exception as e:
            print(e)
            return "Error"

def extract_numeric_values(text):
    pattern = r"(\d+)\s*g/dl"
    matches = re.findall(pattern, text)
    extracted_values = [int(match) for match in matches]
    return extracted_values

@app.route("/")
def index6():
    return render_template("pp.html")  # Render the HTML template

@app.route("/kyc")
def index1():
    return render_template("kyc.html")  # Render the HTML template

@app.route("/idfront")
def index2():
    return render_template("idfront.html")  # Render the HTML template

@app.route("/idback")
def index3():
    return render_template("idback.html")  # Render the HTML template

@app.route("/kycform")
def kycform():
    return render_template("pp.html")  # Render the HTML template

@app.route("/dis")
def index4():
    return render_template("disease.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/thankyou")
def thankyou():
    return render_template("thankyou.html")

@app.route("/thankyoukyc")
def thankyoukyc():
    return render_template("thankyoukyc.html")

@app.route("/thankyouu")
def thankyouu():
    return render_template("thankyouu.html")

@app.route("/ocr")
def index5():
    return render_template("ocr.html")

@app.route("/insurance")
def insurance():
    return render_template("chu.html")

@app.route("/predictedresults")
def predictedresults():
    return render_template("predictedresults.html")

@app.route("/predictdiabetics")
def predictdiabetics():
    return render_template("home.html")

@app.route("/predictheart")
def predictheart():
    return render_template("predictedresults.html")

@app.route("/twitter")
def twitter():
    return redirect("https://www.twitter.com/")
                           
@app.route("/facebook")
def facebook():
    return redirect("https://www.facebook.com/")
                           
@app.route("/instergram")
def instergram():
    return redirect("https://www.instergram.com/")

@app.route("/google")
def google():
    return render_template("index.html")

@app.route("/linkedin")
def linkedin():
    return redirect("https://www.linkedin.com/")

@app.route("/uploadnic", methods=["POST"])
def nic_image():
    global nicnumber
    nicnumber = request.form["nic"]
    return render_template("kyc.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No image received", 400
    
    image_file = request.files["image"]
    image_data = image_file.read()
    
    s3_keynic = nicnumber + "nic.jpg"
    try:
        # Upload the image to S3
        s3_client.put_object(Bucket=bucket_name, Key=s3_keynic, Body=image_data)
        return "Image uploaded successfully"
    except Exception as e:
        print("Error uploading image", e)
        return "Error uploading image", 500

@app.route("/uploadidf", methods=["POST"])
def upload_imageidf():
    if "image" not in request.files:
        return "No image received", 400

    image_file = request.files["image"]
    image_data = image_file.read()

    s3_keyidf = nicnumber + "idf.jpg"
    try:
        # Upload the image to S3
        s3_client.put_object(Bucket=bucket_name, Key=s3_keyidf, Body=image_data)
        return "Image uploaded successfully"
    except Exception as e:
        print("Error uploading image", e)
        return "Error uploading image", 500\

@app.route("/uploadidb", methods=["POST"])
def upload_imageidb():
    if "image" not in request.files:
        return "No image received", 400

    image_file = request.files["image"]
    image_data = image_file.read()

    s3_keyidb = nicnumber + "idb.jpg"
    try:
        # Upload the image to S3
        s3_client.put_object(Bucket=bucket_name, Key=s3_keyidb, Body=image_data)
        return "Image uploaded successfully"
    except Exception as e:
        print("Error uploading image", e)
        return "Error uploading image", 500

@app.route('/compare_images', methods=['POST'])
def compare_images():
    user_verified = 0
    image_url_1 = nicnumber + "nic.jpg"
    image_url_2 = nicnumber + "idf.jpg"

    try:
        # Call Rekognition's CompareFaces API
        response = rekognition_client.compare_faces(
            SourceImage={'S3Object': {'Bucket': 'imageshmps', 'Name': image_url_1}},
            TargetImage={'S3Object': {'Bucket': 'imageshmps', 'Name': image_url_2}}
        )

        # Parse and return the comparison result
        if response['FaceMatches']:
            similarity = response['FaceMatches'][0]['Similarity']

        # Parse and return the comparison result
            if similarity > 85:
                user_verified = 1
            else:
                user_verified = 0

            # Return the results as needed
            return render_template('thankyoukyc.html')
                
        else:
            return "No matching faces found"
    
    except ClientError as e:
        return f"Error: {e}"

# notify.lk API configuration
NOTIFYLK_USER_ID = '25513'
NOTIFYLK_API_KEY = 'DpubzXHTqfjV6W5wUngb'

def send_sms(phone_number):
    url = "https://app.notify.lk/api/v1/send"
    params = {
        "user_id": NOTIFYLK_USER_ID,
        "api_key": NOTIFYLK_API_KEY,
        "sender_id": "NotifyDEMO",
        "to": phone_number,
        "message": f"Your registration is succesful. Please use the below link to proceed with the KYC Verification Process. Link: https://kyc.myshmps.com "
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("SMS sent successfully")
    else:
        print("Failed to send SMS")
        print("Response content:", response.content)
        
def send_smsche(phone_number):
    url = "https://app.notify.lk/api/v1/send"
    params = {
        "user_id": NOTIFYLK_USER_ID,
        "api_key": NOTIFYLK_API_KEY,
        "sender_id": "NotifyDEMO",
        "to": phone_number,
        "message": f"Success,You are Verfied. Thank You! "
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("SMS sent successfully")
    else:
        print("Failed to send SMS")
        print("Response content:", response.content)
        
def send_smsche1(phone_number):
    url = "https://app.notify.lk/api/v1/send"
    params = {
        "user_id": NOTIFYLK_USER_ID,
        "api_key": NOTIFYLK_API_KEY,
        "sender_id": "NotifyDEMO",
        "to": phone_number,
        "message": f"Failed, Please use the below link to proceed with the KYC Verification Process Again. Link: https://kyc.myshmps.com "
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("SMS sent successfully")
    else:
        print("Failed to send SMS")
        print("Response content:", response.content)
        


def calculate_estimated_skin_thickness(age, gender, weight, bmi):
    # Assign factors based on user input using if-else conditions
    if age < 30:
        age_factor = 1
    elif 30 <= age < 50:
        age_factor = 2
    else:
        age_factor = 3
    
    if gender == 'male':
        gender_factor = 3
    else:
        gender_factor = 2
    
    if weight < 60:
        weight_factor = 1
    elif 60 <= weight < 80:
        weight_factor = 2
    else:
        weight_factor = 3
    
    if bmi < 25:
        bmi_factor = 1
    elif 25 <= bmi < 30:
        bmi_factor = 2
    else:
        bmi_factor = 3
    
  
     
    # Weights for each factor (Hypothetical)
    age_weight = 0.2
    gender_weight = 0.3
    weight_weight = 0.2
    bmi_weight = 0.2
    
    
    # Calculate estimated skin thickness
    estimated_skin_thickness = (
        (age_factor * age_weight) +
        (gender_factor * gender_weight) +
        (weight_factor * weight_weight) +
        (bmi_factor * bmi_weight) 
    )
    
    estimated_skin_thickness = int(round(estimated_skin_thickness)*12)

    return estimated_skin_thickness


@app.route("/submit", methods=["POST"])
def submit_form():
    
    if request.method == 'POST':
        first_degree = int(request.form['first_degree'])
        second_degree = int(request.form['second_degree'])
        third_degree = int(request.form['third_degree'])

        # Compute the DPF score
        dpf_score = first_degree * 0.7 + second_degree * 0.2 + third_degree * 0.1
        dpf_score = round(dpf_score, 2)

        height = float(request.form["height"])
        weight = float(request.form["weight"])
        bmii = weight / ((height / 100) ** 2)
        
        age = int(request.form['age']) 
        gender= request.form.get("gender")
        weight= int(request.form['weight']) 
        bmi=bmii
        
        # Calculate estimated skin thickness
    estimated_skin_thickness = calculate_estimated_skin_thickness(age, gender, weight, bmi)
    
  
    estimated_insulin_level = 0
           
     # Convert height to meters
    
    
    age = int(request.form['age'])
    weight = int(request.form['weight'])
    height = int(request.form['height'])    
    gender = int(request.form['gender'])
    alcoholIntake = int(request.form['alcohol'])
    tobaccousage = int(request.form['tobacco'])
    physicalActivityType = int(request.form['exercisetype'])
    physical = int(request.form['excercise'])
    generic = int(request.form['family1'])

    input_data = (
        age, weight, height, bmi, gender, alcoholIntake, tobaccousage,
        physicalActivityType, physical, generic
    )

    input_data_as_numpy_array = np.asarray(input_data)


    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = svm_model.predict(input_data_reshaped)
    prediction1 = qda.predict(input_data_reshaped)
    prediction2 = svm_model.predict(input_data_reshaped)

    if (prediction[0] == 0 and prediction1[0] == 0):
      result = 'Not Healthy'
    elif (prediction1[0] == 0 and prediction2[0] == 0):
      result = 'Not Healthy'
    elif (prediction[0] == 0 and prediction2[0] == 0):
      result = 'Not Healthy'
    elif (prediction[0] == 1 and prediction2[0] == 1):
      result = 'Healthy'
    elif (prediction1[0] == 1 and prediction2[0] == 1):
      result = 'Healthy'
    elif (prediction[0] == 0 and prediction2[0] == 0):
      result = 'Healthy'
      
        # Extract form data
    form_data = {
        "patient_id": int(request.form.get("nic")),
        "name": request.form.get("name"),
        "Age": int(request.form.get("age")),
        "phone_number": int(request.form.get("number")),
        "sex": request.form.get("gender"),
        "district": request.form.get("district"),
        "tobacco_usage": int(request.form.get("tobacco")),
        "occupation_physical_activeness": int(request.form.get("occupation")),
        "type_of_exercise": int(request.form.get("exercisetype")),
        "exercise_hours_per_week": int(request.form.get("excercise")),
        "Pregnancies": int(request.form.get("pregnancy")),
        "BloodPressure": int(request.form.get("pressure")),
        "SkinThickness": estimated_skin_thickness,
        "gestational_diabetes": int(request.form.get("sugarpreg")),
        "chest_pressure_squeezing_running": int(request.form.get("chestpain1")),
        "sharp_chest_pain": int(request.form.get("chestpain2")),
        "other_chest_pain": int(request.form.get("chestpain3")),
        "no_chest_pain": int(request.form.get("chestpain4")),
        "family_history": int(request.form.get("family1")),
        "DiabetesPedigreeFunction": dpf_score,
        "diabetes_pedigree_function":dpf_score,
        "increased_thirst_and_urination": int(request.form.get("thirst")),
        "Insulin": estimated_insulin_level,
        "high_cholesterol": int(request.form.get("choles")),
        "high_blood_pressure": int(request.form.get("high")),
        "sedentary_lifestyle": int(request.form.get("life")),
        "increased_hunger": int(request.form.get("hunger")),
        "weight_loss": int(request.form.get("wloss")),
        "fatigue": int(request.form.get("tired")),
        "blurred_vision": int(request.form.get("vision")),
        "slow_healing_sores_or_infections": int(request.form.get("heal")),
        "areas_of_darkened_skin": int(request.form.get("dark")),
        "chest_pain_during_exercise": int(request.form.get("chest")),
        "BMI" : bmii,
        "initialhealthcondition" : result
    }
    

    # Insert form data into MongoDB
    formdata.insert_one(form_data)

    return render_template('ocr.html')


@app.route("/submitform", methods=["POST"])
def submit_form_personal():
    # Extract form data
    regform_data = {
        "patient_id": int(request.form.get("nic")),
        "fname": request.form.get("firstname"),
        "lname": request.form.get("lastname"),
        "namenic": request.form.get("nicName"),
        "sex": request.form.get("sex"),
        "dob": request.form.get("dob"),
        "password": request.form.get("password"),
        "email": request.form.get("email"),
        "addressnic": request.form.get("nicaddress"),
        "district": request.form.get("district"),
        "City": request.form.get("City"),
        "mobile": request.form.get("mobile"),
        "land": request.form.get("land"),
        "activationsatus": 0
    }
    
    phone_number=int(request.form.get("mobile"))
    send_sms(phone_number)
    
    regformdata.insert_one(regform_data)
    
    return render_template('thankyouu.html')

@app.route("/uploadocr", methods=["POST"])
def submit_form_ocr():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return "Both image files are required"
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return "Both image files must be selected"

        if file1 and file2:
            image1 = Image.open(file1)
            image2 = Image.open(file2)

            ocr = OCR()
            extracted_text1 = ocr.extract(image1)
            extracted_text2 = ocr.extract(image2)

            glucoselevel = extract_numeric_values(extracted_text2)
            glucoselevel_str = ', '.join(map(str, glucoselevel))
            
            # Process extracted_text1 and extracted_text2 using your patterns
            name_pattern = r"NAME\s+(.*?)\s+Date:"
            cholesterol_pattern = r"TOTAL CHOLESTEROL (\d+\.\d+) mg/dL"

            name_match1 = re.search(name_pattern, extracted_text1)
            cholesterol_match1 = re.search(cholesterol_pattern, extracted_text1)

            name_match2 = re.search(name_pattern, extracted_text2)
            cholesterol_match2 = re.search(cholesterol_pattern, extracted_text2)

            if name_match1 and cholesterol_match1:
                name1 = name_match1.group(1)
                total_cholesterol1 = cholesterol_match1.group(1)
            else:
                name1 = "Name not found"
                total_cholesterol1 = "Cholesterol not found"

            if name_match2 and cholesterol_match2:
                name2 = name_match2.group(1)
                total_cholesterol2 = cholesterol_match2.group(1)
            else:
                name2 = "Name not found"
                total_cholesterol2 = "Cholesterol not found"
                
        form_data = {
            'nic': int(request.form.get('nic')),
            'Glucose': int(glucoselevel_str),
            'total_cholesterol': float(total_cholesterol1),
        }

            # Return the results as needed
        formdata.update_one({'patient_id': int(request.form.get('nic'))}, {'$set': {'Glucose': int(glucoselevel_str)}})
        formdata.update_one({'patient_id': int(request.form.get('nic'))}, {'$set': {'total_cholesterol': float(total_cholesterol1)}})

        return 'Form data submitted successfully'
    

#anuja component begining

# Replace this connection string with your MongoDB Atlas connection string
connection_string = 'mongodb+srv://anujan:kasun123@shmps.drhzt7x.mongodb.net/?retryWrites=true&w=majority'
client2 = MongoClient(connection_string)
db = client2['insurance_db']
collection2 = db['form_data']
collection_insurance = db['insurance']
collection_sms_info = db['sms']

# MongoDB connection information for the first code
connection_string1 = 'mongodb+srv://anujan:kasun123@shmps.drhzt7x.mongodb.net/?retryWrites=true&w=majority'
client1 = MongoClient(connection_string1)
db1 = client1['insurance_db']
collection12 = db1['persons']
ins_collection1 = db1['ins_persons']
login_collection1 = db1['login_attempts']
collection_sms_info1 = db1['sms']

NOTIFY_ENDPOINT = "https://app.notify.lk/api/v1/send"
API_KEY = "DpubzXHTqfjV6W5wUngb"

def send_smsii(phone_number, message):
    data = {
        "user_id": "25513",
        "api_key": API_KEY,
        "sender_id": "NotifyDEMO",
        "to": phone_number,
        "message": message
    }
    response = requests.post(NOTIFY_ENDPOINT, data=data)
    
    # Check if the response indicates success or failure
    if response.status_code != 200:
        print(f"Error sending SMS: {response.text}")
        return None
    
    response_data = response.json()
    if 'status' in response_data and response_data['status'] != "success":
        print(f"Error from notify.lk: {response_data['message']}")
        
    return response_data

# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('insurancen.csv')

# encoding sex column
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)

# encoding 'smoker' column
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)

# encoding 'region' column
occupations_mapping = {
    'Actor/Actress': 0,
    'Teacher': 1,
    'Salesperson': 2,
    'Construction worker': 3,
    'Human resources manager': 4,
    'Architect': 5,
    'Tailor': 6,
    'Librarian': 7,
    'Entrepreneur': 8,
    'Carpenter': 9,
    'Journalist': 10,
    'Hotel manager': 11,
    'Social worker': 12,
    'Artist': 13,
    'Banker': 14,
    'Scientist': 15,
    'Event planner': 16,
    'Economist': 17,
    'Mechanic': 18,
    'Financial analyst': 19,
    'Lawyer': 20,
    'Retired': 21,
    'Marketing executive': 22,
    'Pharmacist': 23,
    'Consultant': 24,
    'Driver': 25,
    'Veterinarian': 26,
    'Government officer': 27,
    'Researcher': 28,
    'Doctor': 29,
    'Psychologist': 30,
    'NGO worker': 31,
    'Software developer': 32,
    'Receptionist': 33,
    'Musician': 34,
    'Tour guide': 35,
    'Engineer': 36,
    'Accountant': 37,
    'Electrician': 38,
    'Photographer': 39,
    'Fashion designer': 40,
    'Businessperson': 41,
    'IT professional': 42,
    'Flight attendant': 43,
    'Nurse': 44,
    'Graphic designer': 45,
    'Police officer': 46,
    'Professor': 47,
    'Chef': 48,
    'Farmer': 49,
    'Athlete':50
    
}
# Now perform the replacement using the mapping
insurance_dataset.replace({'occupation': occupations_mapping}, inplace=True)

# encoding heart disease column
insurance_dataset.replace({'heart issue':{'yes':1,'no':0}}, inplace=True)

# encoding diabetics column
insurance_dataset.replace({'diabetics':{'yes':1,'no':0}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Creating the linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, Y_train)

# Creating the Gradient Boosting Regressor model
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, Y_train)

# prediction on training data using both models
training_data_prediction_linear = linear_regressor.predict(X_train)
training_data_prediction_gb = gb_regressor.predict(X_train)

# Combine predictions from both models using a weighted average (you can change the weights as needed)
training_data_prediction_combined = (0.7 * training_data_prediction_linear) + (0.3 * training_data_prediction_gb)

# R squared value and Mean Squared Error on training data
r2_train = r2_score(Y_train, training_data_prediction_combined)
mse_train = mean_squared_error(Y_train, training_data_prediction_combined)
print('Training Data - R squared value: ', r2_train)
print('Training Data - Mean Squared Error: ', mse_train)

# prediction on test data using both models
test_data_prediction_linear = linear_regressor.predict(X_test)
test_data_prediction_gb = gb_regressor.predict(X_test)

# Combine predictions from both models using a weighted average (you can change the weights as needed)
test_data_prediction_combined = (0.7 * test_data_prediction_linear) + (0.3 * test_data_prediction_gb)

# R squared value and Mean Squared Error on test data
r2_test = r2_score(Y_test, test_data_prediction_combined)
mse_test = mean_squared_error(Y_test, test_data_prediction_combined)
print('Test Data - R squared value: ', r2_test)
print('Test Data - Mean Squared Error: ', mse_test)

# Assuming the input_data variable contains the input features in the same order as in the DataFrame
input_data = (52, 0, 40.3, 2, 0, 22, 0, 1)

# Convert input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make predictions using both models
prediction_linear = linear_regressor.predict(input_data_reshaped)
prediction_gb = gb_regressor.predict(input_data_reshaped)

# Combine predictions from both models using a weighted average (you can change the weights as needed)
prediction_combined = (0.7 * prediction_linear) + (0.3 * prediction_gb)

print('The insurance cost is LKR ', prediction_combined[0])

@app.route('/cost_prediction')
def cost_prediction():
    # Your logic here
    return render_template('predict.html')


#@app.route('/')
#def inde():
   # return render_template('predict.html')



@app.route('/next_page')
def next_page():
    return render_template('next_page.html')  # Replace 'next_page.html' with the actual template filename



@app.route('/predicts', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        patient_id = int(request.form.get('patient_id'))  # Assuming you have a form field named 'patient_id'
        
        # Fetch patient data from MongoDB based on patient_id
        form_data = collection2.find_one({'id': patient_id})
        
        if form_data:
            # Extract relevant features from patient_data and reshape
            input_data = np.array([
                int(form_data['age']),
                int(form_data['gender']),
                float(form_data['bmi']),
                int(form_data['child']),
                int(form_data['tobacco']),
                int(form_data['occupation']),
                int(form_data['heart_issue']),
                int(form_data['diabetics'])
            ]).reshape(1, -1)

            
            # Make predictions using both models
            prediction_linear = linear_regressor.predict(input_data)
            prediction_gb = gb_regressor.predict(input_data)
            
            # Combine predictions using a weighted average
            prediction_combined = (0.7 * prediction_linear) + (0.3 * prediction_gb)
            
            predicted_cost = prediction_combined[0]
           
           # Update the predicted cost in the existing document
            collection2.update_one({'id': patient_id}, {'$set': {'predicted_cost': predicted_cost}})

            return render_template('result.html', predicted_cost=predicted_cost)
        else:
            return 'Patient ID not found in the database'

    


# Load the dataset
data = pd.read_csv('insurance_plan.csv')

# Convert categorical features to numerical
encoder = LabelEncoder()
data['heart_disease'] = encoder.fit_transform(data['heart_disease'])
data['diabetes'] = encoder.fit_transform(data['diabetes'])
data['suitable_insurance_plan'] = encoder.fit_transform(data['suitable_insurance_plan'])

# Select features and target
features = ['desired_coverage', 'age', 'children', 'heart_disease', 'diabetes']
target = 'suitable_insurance_plan'

X = data[features]
y = data[target]

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

@app.route('/plan', methods=['POST'])
def plan():
 patient_id = int(request.form.get('patient_id'))  # Assuming you have a form field named 'patient_id'
        
        # Fetch patient data from MongoDB based on patient_id
 form_data = collection2.find_one({'id': patient_id})

    # Transform patient data and make predictions
 if form_data:
        # Transform patient_data into input_data format
        input_data = [[
            form_data['predicted_cost'],
            form_data['age'],
            form_data['child'],
            form_data['heart_issue'],
            form_data['diabetics']
        ]]
        
        predicted_plan_encoded = model.predict(input_data)[0]
        predicted_plan = encoder.inverse_transform([predicted_plan_encoded])[0]
       
        update_query = {'id': int(patient_id)}
        update_data = {'$set': {'plan': predicted_plan}}
        collection2.update_one(update_query, update_data)
        
        return render_template('plan.html', predicted_plan=predicted_plan)
 else:
        return "Patient data not found"
@app.route('/request')
def index9():
    return render_template('enter.html')

@app.route('/fetch_insurance_plan', methods=['POST'])
def fetch_insurance_plan():
    try:
        # Retrieve patient_id from the POST request
        patient_id = request.form.get('patient_id')

        # Fetch insurance plan from MongoDB using patient_id
        patient = collection2.find_one({'id': int(patient_id)})
        if patient:
            insurance_plan = patient.get('plan')
            print("Retrieved insurance plan:", insurance_plan)
            patient_phone_number = patient.get('phone_number')

            # Fetch insurance company details using the insurance plan
            insurance_data = collection_insurance.find_one({'plan': insurance_plan})
            if insurance_data:
                insurance_company_id = insurance_data.get('company_id')
                insurance_company_phone = int(insurance_data.get('phone_number'))
                print("Retrieved insurance company phone:", insurance_company_phone)

                message = f"Patient {patient_id} ({patient.get('name')}) has requested their insurance plan: {insurance_plan}. Please go to http://127.0.0.1:5000/get_data/{insurance_company_id}"

                send_smsii(insurance_company_phone,message)

                # Store information in MongoDB (if needed)
                sms_info = {
                    'insurance_company_id': insurance_company_id,
                    'patient_id': patient_id,
                    'insurance_plan': insurance_plan,
                    'patient_contact_number': patient_phone_number

                }
                collection_sms_info.insert_one(sms_info)
                return render_template('thankyouu.html')

            else:
                return "Insurance company data not found"

    except Exception as e:
        return str(e)

@app.route('/get_data/<id>', methods=['GET'])
def get_data_by_id1(id):
    
    data = collection_sms_info.find({'insurance_company_id': int(id)})

    return render_template('dash.html', data=data)

NOTIFY_LK_API_KEY="DpubzXHTqfjV6W5wUngb"

@app.route('/send-smss', methods=['POST'])
def send_smsi():
    try:
        data = request.json
        patient_id = data.get('patient_id')
        patient_contact_number = data.get('patient_contact_number')

        # Your SMS content
        sms_content = 'Your appointment has been accepted.'

        # Notify.lk API endpoint
        api_url = 'https://app.notify.lk/api/v1/send'

        # Send SMS using Notify.lk API
        response = requests.post(api_url, json={
            "api_key": API_KEY,
            'to': patient_contact_number,
            'message': sms_content,
            'sender': 'NotifyDEMO',
            "user_id": "25513"
        })
        
        
        print('SMS sent:', response.json())
        
        return jsonify({'success': True})

    except Exception as e:
        print('Error sending SMS:', str(e))
        return jsonify({'success': False, 'error': str(e)}), 500
    

#second part



@app.route("/loggin", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("user")
        password = request.form.get("pass")

        user = collection12.find_one({"email": email, "password": password})

        ins_info = ins_collection1.find_one({"email": email})

        if ins_info:
            ins_id = ins_info["id"]
            cmp_id = ins_info["company_id"]
            # Check the number of unsuccessful login attempts
            login_attempts = login_collection1.count_documents({"ins_id": ins_id, "cmp_id": cmp_id, "success": False})

            if login_attempts >= 3:
                # Account locked out
                return render_template("final_login.html", error="Your account is locked out due to too many login attempts.")

        if user:
            # Successful login
            if ins_info:
                # Insert the successful login attempt
                login_collection1.insert_one({"ins_id": ins_id, "email": email, "cmp_id": cmp_id, "success": True})
                return redirect(url_for("dashboard", username=user["email"]))
            else:
                # Ins info not found, handle accordingly
                return render_template("final_login.html", error="Invalid credentials")
        else:
            # Invalid credentials
            if ins_info:
                # Insert the unsuccessful login attempt
                login_collection1.insert_one({"ins_id": ins_id, "email": email, "cmp_id": cmp_id, "success": False})
            return render_template("final_login.html", error="Invalid credentials")

    return render_template("final_login.html")


@app.route("/dashboard/<username>")
def dashboard(username):
    return render_template("dasha.html", username=username)

@app.route("/verify/<company_id>/<insurance_id>")
def verify(company_id, insurance_id):
    company = ins_collection1.find_one({"company_id": int(company_id)})
    insurance = ins_collection1.find_one({"id": int(insurance_id)})

    if company and insurance:
        verification_result = {"company_id": int(company_id), "insurance_id": int(insurance_id), "result": 1}
        login_collection1.update_one(
            {"cmp_id": int(company_id), "ins_id": int(insurance_id)},
            {"$set": {"result": 1}},
            upsert=True
        )
        return jsonify({"status": "success", "message": "Verification successful."})
    else:
        verification_result = {"company_id": int(company_id), "insurance_id": int(insurance_id), "result": 0}
        login_collection1.update_one(
            {"company_id": int(company_id), "insurance_id": int(insurance_id)},
            {"$set": {"result": 0}},
            upsert=True
        )
        return jsonify({"status": "error", "message": "Verification failed."})

@app.route("/verifyme", methods=["POST"])
def verifyme():
    id_value = request.form.get("idInput")
    
    # Query the MongoDB collection for the verification result
    verification_result = login_collection1.find_one({"ins_id": int(id_value)})
    
    if verification_result and verification_result.get("result") == 1:
        cmp_id = verification_result.get("cmp_id")  # Retrieve company ID from verification_result
        return redirect(url_for("get_data_by_id", cmp_id=cmp_id))  
    else:
        return redirect(url_for("failure"))

@app.route('/get_data/<cmp_id>', methods=['GET'])
def get_data_by_id(cmp_id):
    data = collection_sms_info.find({'insurance_company_id': int(cmp_id)})
    return render_template('id.html', data=data, cmp_id=cmp_id)  # Pass company_id to the template


@app.route("/success")
def success():
    return render_template('dash.html')

@app.route("/failure")
def failure():
    return "Verification Failed! Redirected to the failure page."
API_KEY = "DpubzXHTqfjV6W5wUngb"

@app.route('/send-smss1', methods=['POST'])
def send_smss():
    try:
        data = request.json
        patient_id = data.get('patient_id')
        patient_contact_number = data.get('patient_contact_number')

        # Your SMS content
        sms_content = 'Your appointment has been accepted.'

        # Notify.lk API endpoint
        api_url = 'https://app.notify.lk/api/v1/send'

        # Send SMS using Notify.lk API
        response = requests.post(api_url, json={
           
            "user_id": "25513",
            "api_key": API_KEY,
            'sender_id': 'NotifyDEMO',
            'to': patient_contact_number,
            'message': sms_content,
           
        })
        # If SMS sending is successful, delete the MongoDB record
        if response.status_code == 200 and response.json().get('status') == 'success':
            collection_sms_info.delete_one({"patient_id": int(patient_id)})  # Assuming patient_id should be an integer
            print("Deleted MongoDB record.")  # Debug print
        
        print('SMS sent:', response.json())
        return jsonify({'success': True})

    except Exception as e:
        print('Error sending SMS:', str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/test")
def test():
    return render_template('chu.html')

@app.route('/submitinsu', methods=['POST'])
def submit_form1():
    form_data = {
        'id': int(request.form.get('id')),
        'age': int(request.form.get('age')),
        'gender': request.form.get('gender'),
        'phone_number':request.form.get('phone_number'),
        'bmi':float(request.form.get('bmi')),
         'child':int(request.form.get('child')),
         'tobacco':(request.form.get('tob')),
         'occupation': request.form.get('occupation'),
         'heart_issue':(request.form.get('heart_issue')),
         'diabetics':(request.form.get('diabetics')),
    }
   
    collection2.insert_one(form_data)
    
    return render_template('thankyou.html')

#sms.py integration

NOTIFY_ENDPOINT = "https://app.notify.lk/api/v1/send"
API_KEY = "DpubzXHTqfjV6W5wUngb"

def send_sms1(phone_number, message):
    data = {
        "user_id": "25513",
        "api_key": API_KEY,
        "sender_id": "NotifyDEMO",
        "to": phone_number,
        "message": message
    }
    response = requests.post(NOTIFY_ENDPOINT, data=data)
    
    # Check if the response indicates success or failure
    if response.status_code != 200:
        print(f"Error sending SMS: {response.text}")
        return None
    
    response_data = response.json()
    if 'status' in response_data and response_data['status'] != "success":
        print(f"Error from notify.lk: {response_data['message']}")
        
    return response_data

@app.route('/test1')
def test1():
    return render_template('enter.html')

@app.route('/fetch_insurance_plan', methods=['POST'])
def fetch_insurance_plan1():
    try:
        # Retrieve patient_id from the POST request
        patient_id = request.form.get('patient_id')

        # Fetch insurance plan from MongoDB using patient_id
        patient = collection2.find_one({'id': int(patient_id)})
        if patient:
            insurance_plan = patient.get('plan')
            print("Retrieved insurance plan:", insurance_plan)
            patient_phone_number = patient.get('phone_number')

            # Fetch insurance company details using the insurance plan
            insurance_data = collection_insurance.find_one({'plan': insurance_plan})
            if insurance_data:
                insurance_company_id = insurance_data.get('company_id')
                insurance_company_phone = insurance_data.get('phone_number')
                print("Retrieved insurance company phone:", insurance_company_phone)

                message = f"Patient {patient_id} ({patient.get('name')}) has requested their insurance plan: {insurance_plan}. Please go to http://127.0.0.1:5000/get_data/{insurance_company_id}"

                send_sms1(insurance_company_phone, message)

                # Store information in MongoDB (if needed)
                sms_info = {
                    'insurance_company_id': insurance_company_id,
                    'patient_id': patient_id,
                    'insurance_plan': insurance_plan,
                    'patient_contact_number': patient_phone_number

                }
                collection_sms_info.insert_one(sms_info)
                return "SMS sent successfully"

            else:
                return "Insurance company data not found"

    except Exception as e:
        return str(e)

@app.route('/get_data/<id>', methods=['GET'])
def get_data_by_id2(id):
    
    data = collection_sms_info.find({'insurance_company_id': int(id)})

    return render_template('id.html', data=data)


#Iresh component begins
warnings.filterwarnings('ignore')
MONGO_URI = "mongodb+srv://iresh678:iresh123@cluster0.vsmqddt.mongodb.net/mydatabase?retryWrites=true&w=majority"
client3 = MongoClient(MONGO_URI)
db = client3["mydatabase"]
patient_collection = db["patient_data"] 


# Load the CSV data to a pandas dataframe
heart_data = pd.read_csv('heart_disease_data.csv')

# Split the dataset into features (X) and target (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create the GridSearchCV object for Naive Bayes
nb_grid_search = GridSearchCV(GaussianNB(), {}, cv=5, return_train_score=True, verbose=1)
nb_grid_search.fit(X_train, Y_train)

# Get the best Naive Bayes model and its accuracy
best_nb_model = nb_grid_search.best_estimator_
best_nb_accuracy = nb_grid_search.best_score_

print("Best Naive Bayes Accuracy: {:.2f}%".format(best_nb_accuracy * 100))

# Define the parameter grid for Logistic Regression
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Solver algorithm
    'max_iter': [100, 200, 500],  # Maximum number of iterations
}

# Create the GridSearchCV object for Logistic Regression
lr_grid_search = GridSearchCV(LogisticRegression(), lr_param_grid, cv=5, return_train_score=True, verbose=1)
lr_grid_search.fit(X_train, Y_train)

# Get the best Logistic Regression model and its accuracy
best_lr_model = lr_grid_search.best_estimator_
best_lr_accuracy = lr_grid_search.best_score_

print("Best Logistic Regression Accuracy: {:.2f}%".format(best_lr_accuracy * 100))

# Create the ensemble model using VotingClassifier
estimators = [
    ('nb', best_nb_model),
    ('lr', best_lr_model)
]

hybrid_model = VotingClassifier(estimators=estimators, voting='hard')

# Fit the hybrid model on the training data
hybrid_model.fit(X_train, Y_train)

# Make predictions on the test data
predictions = hybrid_model.predict(X_test)

# Calculate accuracy on the test data
accuracy = accuracy_score(Y_test, predictions)
print("Hybrid Model Accuracy: {:.2f}%".format(accuracy * 100))

# Hardcoded prediction
input_data = (56,1,3,120,193,0,0,162,0,1.9,1,0,3) 
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = hybrid_model.predict(input_data_reshaped)
print(prediction)

if prediction[0]==0:
    print('The person does not have a Heart disease')
else:
    print('The person has heart disease')


#risk level 1 

def calculate_heart_disease_risk(age, gender, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, exercise_induced_angina):
    # Assign points for each risk factor
    age_points = 0
    gender_points = 0
    chest_pain_points = 0
    resting_blood_pressure_points = 0
    cholesterol_points = 0
    fasting_blood_sugar_points = 0
    exercise_induced_angina_points = 0

    # Age points
    if age >= 45:
        age_points = 4
    elif age >= 35:
        age_points = 2

    # Gender points (1 for male, 0 for female)
    if gender == "male":
        gender_points = 3

    # Chest pain points (Typical Angina 0, Atypical Angina 1, Non-Anginal Pain 2, Asymptomatic 3)
    chest_pain_points = chest_pain_type

    # Resting blood pressure points (measured in mmHg)
    if resting_blood_pressure >= 140:
        resting_blood_pressure_points = 2
    elif resting_blood_pressure >= 120:
        resting_blood_pressure_points = 1

    # Cholesterol points (measured in mg/dL)
    if cholesterol >= 240:
        cholesterol_points = 3
    elif cholesterol >= 200:
        cholesterol_points = 2

    # Fasting blood sugar points (1 for fasting blood sugar > 120 mg/dl, 0 for fasting blood sugar <= 120 mg/dl)
    if fasting_blood_sugar:
        fasting_blood_sugar_points = 2

    # Exercise induced angina points (1 for yes, 0 for no)
    if exercise_induced_angina:
        exercise_induced_angina_points = 2

    # Calculate total risk score
    total_points = age_points + gender_points + chest_pain_points + resting_blood_pressure_points + cholesterol_points + fasting_blood_sugar_points + exercise_induced_angina_points

    # Define risk levels based on total points
    if total_points <= 7:
        risk_level = "Low"
    elif total_points <= 14:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Calculate the percentage risk based on total points
    max_possible_points = 20  # Adjust as per the maximum possible points
    risk_percentage = (total_points / max_possible_points) * 100

    return risk_percentage, risk_level

#get user data
age = 30
gender = 'male'
chest_pain_type = 0
resting_blood_pressure = 240
cholesterol =300
fasting_blood_sugar = 1
exercise_induced_angina = 1

risk_percentage, risk_level = calculate_heart_disease_risk(age, gender, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, exercise_induced_angina)
print("Risk Percentage:", risk_percentage, "%")
print("Risk Level:", risk_level)

def get_recommendationss(risk_level):
    if risk_level == "Low":
        return "Your heart disease risk is low. Keep up a healthy lifestyle to maintain your heart health. Here are some recommendations:\br\n1. Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.\n2. Engage in regular physical activity, such as brisk walking, jogging, or swimming.\n3. Avoid smoking and limit alcohol consumption.\n4. Monitor your blood pressure and cholesterol levels regularly.\n5. Manage stress through relaxation techniques or hobbies."
    elif risk_level == "Moderate":
        return ("Your heart disease risk is moderate. Consider making lifestyle changes to reduce your risk."
                "\n\nHere are some recommendations:"
                "\n1. Adopt a heart-healthy diet with less saturated and trans fats, and lower sodium intake.")
    else:
        return "Your heart disease risk is high. It's crucial to take immediate action and consult a healthcare professional for guidance. Here are some recommendations:\n\n1. Follow a heart-healthy diet with minimal saturated and trans fats, and reduced salt intake.\n2. Engage in regular exercise under medical supervision.\n3. Quit smoking and limit alcohol intake completely.\n4. Monitor blood pressure, cholesterol, and blood sugar levels closely.\n5. Collaborate with your healthcare provider to develop a comprehensive plan to manage your risk factors."



#risk calculator 2 

def custom_calculate_risk(exercise_hours, chest_squeeze, sharp_chest_pain, other_chest_pain, unusual_fatigue, exercise_induced_chest_pain, family_history):
    # Calculate points for each factor
    exercise_hours_points = 0
    if 5 <= exercise_hours <= 6:
        exercise_hours_points = 1
    elif 8 <= exercise_hours <= 10:
        exercise_hours_points = 2

    chest_pain_points = chest_squeeze * 1 + sharp_chest_pain * 2 + other_chest_pain * 0
    unusual_fatigue_points = unusual_fatigue * 1
    exercise_induced_chest_pain_points = exercise_induced_chest_pain * 2
    family_history_points = family_history * 2

    # Calculate total points
    total_points = (
        exercise_hours_points
        + chest_pain_points
        + unusual_fatigue_points
        + exercise_induced_chest_pain_points
        + family_history_points
    )

    # Define risk levels based on total points
    if total_points <= 5:
        risk_level = "Low"
    elif total_points <= 10:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Calculate the percentage risk based on total points
    max_possible_points = 14  # Adjust as per the maximum possible points
    risk_percentages = (total_points / max_possible_points) * 100

    return risk_level, risk_percentages

exercise_hours = 9.5
chest_squeeze = 0
sharp_chest_pain = 0
other_chest_pain = 1
unusual_fatigue = 1
exercise_induced_chest_pain = 1
family_history = 3

# Calculate and display results
risk_level, risk_percentages = custom_calculate_risk(exercise_hours, chest_squeeze, sharp_chest_pain, other_chest_pain, unusual_fatigue, exercise_induced_chest_pain, family_history)
print(f"Risk Level: {risk_level}")
print(f"Risk Percentage: {risk_percentages:.2f}%")

def get_custom_recommendations(risk_level):
    if risk_level == "Low":
        return "Your custom heart disease risk is low. Keep up with your healthy exercise routine and lifestyle. Here are some recommendations:\br\n1. Maintain a balanced diet to support your fitness efforts.\n2. Stay consistent with your exercise routine, and consider mixing in different types of physical activity.\n3. Stay hydrated and get adequate rest.\n4. Monitor your heart health and overall well-being regularly."
    elif risk_level == "Moderate":
        return "Your custom heart disease risk is moderate. Consider adjusting your exercise routine and lifestyle for better heart health. Here are some recommendations:\n\n1. Focus on a well-rounded fitness routine that includes cardiovascular, strength, and flexibility exercises.\n2. Pay attention to your body's signals during exercise and adapt your routine as needed.\n3. Prioritize recovery with proper nutrition, sleep, and stress management.\n4. Consult a fitness professional for personalized guidance."
    else:
        return "Your custom heart disease risk is high. It's important to consult a healthcare professional and make significant changes to your exercise and lifestyle. Here are some recommendations:\n\n1. Work closely with a medical and fitness team to create a safe and effective exercise plan.\n2. Include regular cardiovascular exercise and supervised strength training.\n3. Prioritize cardiac health by managing intensity and monitoring for any adverse symptoms.\n4. Follow medical advice and get regular check-ups to track progress."

#Display nearest facilities pharmacy and hospital

# OpenCage Geocoding API endpoint
endpoint = 'https://api.opencagedata.com/geocode/v1/json'
api_key = '33365c941d474dc086c98c6cb87823a3'



# Function to calculate distance

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Function to get location coordinates using OpenCage Geocoding API
def get_location_coordinates(location):
    params = {
        'q': location,
        'key': api_key,
        'limit': 1
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lon = data['results'][0]['geometry']['lng']
        return lat, lon
    return None, None 

# Function to get nearby hospitals from MongoDB

def get_nearby_hospitals(location, radius):
    lat1, lon1 = get_location_coordinates(location)
    if lat1 is None or lon1 is None:  # Check for None values
        return None, None, []

    hospitals = get_hospitals_from_db()

    nearby_hospitals = []
    for hospital, (lat2, lon2) in hospitals.items():
        if lat2 is None or lon2 is None:  # Check for None values in hospitals' coordinates
            continue
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        if distance <= radius:
            nearby_hospitals.append({"name": hospital, "lat": lat2, "lon": lon2, "distance": distance})

    return lat1, lon1, nearby_hospitals

# Function to get nearby pharmacies from MongoDB
def get_nearby_pharmacy(location, radius):
    lat1, lon1 = get_location_coordinates(location)
    if lat1 is None or lon1 is None:  # Check for None values
        return None, None, []

    pharmacies = get_pharmacy_from_db()

    nearby_pharmacies = []
    for pharmacy, (lat2, lon2) in pharmacies.items():
        if lat2 is None or lon2 is None:  # Check for None values in pharmacies' coordinates
            continue
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        if distance <= radius:
            nearby_pharmacies.append({"name": pharmacy, "lat": lat2, "lon": lon2, "distance": distance})

    return lat1, lon1, nearby_pharmacies

# Function to get hospitals from MongoDB
def get_hospitals_from_db():
    hospitals_data = {}
    for hospital in hospitals_collection.find():
        hospitals_data[hospital['name']] = (hospital['lat'], hospital['lon'])
    return hospitals_data

# Function to get pharmacies from MongoDB
def get_pharmacy_from_db():
    pharmacy_data = {}
    for pharmacy in pharmacy_collection.find():
        pharmacy_data[pharmacy['name']] = (pharmacy['lat'], pharmacy['lon'])
    return pharmacy_data

@app.route('/heartd')
def heartd():
    return render_template('main.html')

@app.route('/heartmy')
def predim():
    return render_template('index1.html')

@app.route('/pred',methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
       
        user_input_id = request.form['patient_id']
        
        patient_data = patient_collection.find_one({'patient_id': int(user_input_id)})
        # Print the retrieved patient_data
        print("Retrieved patient_data:", patient_data)
        
        if patient_data:
            # Extract the necessary features from the patient_data dictionary
            necessary_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach','exang', 'st', 'slope', 'ca', 'thal']
            features = [patient_data[feature] for feature in necessary_features]
            # Convert the extracted features into a numpy array
        
        input_data_as_numpy_array_patient = np.asarray(features)
        input_data_reshaped_patient = input_data_as_numpy_array_patient.reshape(1, -1)

        # Now, you can use input_data_reshaped_patient for prediction
        prediction = hybrid_model.predict(input_data_reshaped_patient)
     
        if prediction[0] == 0:
            result = 'The person does not have a heart disease.'
        else:
            result = 'The person has a heart disease.'

        return render_template('render1.html', result=result)
  

@app.route('/input_dat')
def input():
    return render_template('input1.html')

# Connect to MongoDB
connection_string = "mongodb+srv://kasun312:kasun123@shmps.vpmeq0h.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)
db = client['shmps']
collection = db['patient_data']
collection1 = db['pdf_info']
doctors_col = db['doctors']
keys_col = db['keys']
requests_col = db['requests']
patients_col = db['patient_data']

@app.route('/met', methods=['GET'])
def meter():
    patient_id = request.args.get('patient_id')

    if not patient_id:
        return "Patient ID not provided", 400

    patient_data = collection.find_one({'patient_id': int(patient_id)})

    if patient_data is None:
        return "Patient not found", 404

    age = int(patient_data['Age'])
    gender = float(patient_data['sex'])
    chest_pain_type = bool(patient_data['family_history'])
    resting_blood_pressure = bool(patient_data['BloodPressure'])
    cholesterol = bool(patient_data['high_cholesterol'])
    fasting_blood_sugar = bool(patient_data['Glucose'])
    exercise_induced_angina = bool(patient_data['chest_pain_during_exercise'])
    
    
    # Calculate the risk
    risk_percentage, risk_level = calculate_heart_disease_risk(age, gender, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, exercise_induced_angina)

    recommendations = get_recommendationss(risk_level)

    # Input data for the scenario (Exercise Enthusiast's Routine)
    exercise_hours = float(patient_data['exercise_hours_per_week'])
    chest_squeeze = int(patient_data['chest_pressure_squeezing_running'])
    sharp_chest_pain = float(patient_data['sharp_chest_pain'])
    other_chest_pain = int(patient_data['other_chest_pain'])
    unusual_fatigue = int(patient_data['fatigue'])
    exercise_induced_chest_pain = int(patient_data['no_chest_pain'])
    family_history = int(patient_data['family_history'])

    # Calculate custom risk
    custom_risk_level, custom_risk_percentages = custom_calculate_risk(exercise_hours, chest_squeeze, sharp_chest_pain, other_chest_pain, unusual_fatigue, exercise_induced_chest_pain, family_history)

    custom_recommendations = get_custom_recommendations(custom_risk_level)

    return render_template('meter1.html', custom_risk_level=custom_risk_level, custom_risk_percentages=custom_risk_percentages, risk_percentage=risk_percentage, risk_level=risk_level, recommendations=recommendations, custom_recommendations=custom_recommendations)


# Main route to handle form submission
@app.route('/hos', methods=['GET', 'POST'])
def hos():
    if request.method == 'POST':
        location = request.form['location']
        facility_type = request.form['facilityType']
        radius = 10  # Initialize radius with the default value

        if facility_type == 'hospitals':
            lat, lon, nearby_facilities = get_nearby_hospitals(location, radius)
            return render_template('hos1.html', nearby_facilities=nearby_facilities, lat=lat, lon=lon, radius=radius)
        elif facility_type == 'pharmacies':
            lat, lon, nearby_facilities = get_nearby_pharmacy(location, radius)  # Change variable name here
            return render_template('phr1.html', nearby_facilities=nearby_facilities, lat=lat, lon=lon, radius=radius)  # Change variable name here
    
    return render_template('hos.html')



#pdf generating process
@app.route('/gen')
def gen():
    return render_template('gen1.html')



@app.route('/generate_report', methods=['POST'])
def generate_report():
    patient_id = request.form.get('patient_id')

    if patient_id:
        patient_data = collection.find_one({'patient_id': int(patient_id)})
        if not patient_data:
            return "Patient not found."
        
        ref_number = ''.join(random.choices(string.digits, k=8))

        # Render the HTML template with patient data
        with open('templates/repo.html', 'r') as template_file:
            template = template_file.read()
        rendered_html = render_template_string(template, 
                                               patient_id=patient_data['patient_id'], 
                                               Age=patient_data['Age'], 
                                               Name=patient_data['name'],
                                               Sex=patient_data['sex'], 
                                               cholesterol=patient_data['high_cholesterol'],   
                                               BloodPressure=patient_data['BloodPressure'],      
                                               Glucose=patient_data['Glucose'],
                                               no_chest_pain=patient_data['no_chest_pain'],
                                               risk1=patient_data['result_2'],
                                               risklevel=patient_data['risk_level'],
                                               Precentage=patient_data['sym_precentage'],
                                               date=datetime.datetime.now().strftime('%Y-%m-%d'))

        # Convert rendered HTML to PDF
        pdf_buffer = io.BytesIO()
        pisa.CreatePDF(rendered_html, dest=pdf_buffer)

        
        # Encrypt the PDF
        password = generate_random_password()
        encrypted_pdf_buffer = encrypt_pdf(pdf_buffer, password)
        
        phone_number = patient_data.get('phone_number', None) # Assuming 'phone_number' is the field name in the database
        if phone_number:
            send_password_via_sms(phone_number, password)
        else:
            print(f"No phone number found for patient ID: {patient_id}")

        return send_file(
            encrypted_pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'{patient_id}_{ref_number}report.pdf'
        )
    else:
        return "Please enter a patient ID."

#face detection part

cnt = 0
pause_cnt = 0
justscanned = False
 
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier( "E:/iresh/Progress/resources/haarcascade_frontalface_default.xml")   
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]
 
    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "E:/iresh/Progress/dataset"
    
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
 
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
 
    return redirect('/exe')
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
 
        global justscanned
        global pause_cnt
 
        pause_cnt += 1
 
        coords = []
 
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
 
            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1
 
                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w
 
                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
 
                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
 
                mycursor.execute("select a.img_person, b.prs_name, b.prs_skill "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]
 
                if int(cnt) == 30:
                    cnt = 0
 
                    mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('"+str(date.today())+"', '" + pnbr + "')")
                    mydb.commit()
 
                    cv2.putText(img, pname , (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)
 
                    justscanned = True
                    pause_cnt = 0
 
            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
 
                if pause_cnt > 80:
                    justscanned = False
 
            coords = [x, y, w, h]
        return coords
 
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img
 
    faceCascade = cv2.CascadeClassifier("E:/iresh/Progress/resources/haarcascade_frontalface_default.xml")
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
 
    wCam, hCam = 400, 400
 
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
 
    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)
 
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Check if a person has been recognized and set the flag
        if justscanned:
            break

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()  # Release the video capture obje
 
 
 
@app.route('/exe')
def exe():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()
 
    return render_template('fa.html', data=data)
 
@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))
 
    return render_template('addprsn.html', newnbr=int(nbr))
 
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')
 
    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()
 
    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))
 
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)
 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added, a.is_verified "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return render_template('fr_page.html', data=data)
 
 
@app.route('/countTodayScan')
def loadData():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("SELECT a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, b.prs_nbr, "
                     "DATE_FORMAT(a.accs_added, '%H:%i:%s'), a.is_verified "
                     "FROM accs_hist a ,  prs_mstr b"
                     "LEFT JOIN prs_mstr b ON a.accs_prsn = b.prs_nbr "
                     "WHERE a.accs_date = CURDATE() "
                     "ORDER BY a.accs_id DESC")
    data = mycursor.fetchall()

    for row in data:
        accs_id, accs_prsn, prs_name, prs_skill, prs_nbr, accs_added, is_verified = row

        if prs_nbr is not None and accs_prsn == prs_nbr:
            verified_score = 1
        else:
            verified_score = 0

        # Update the 'is_verified' field in the 'accs_hist' table
        update_query = "UPDATE accs_hist SET is_verified = %s WHERE accs_prsn = %s"
        update_values = (verified_score, accs_prsn)
        mycursor.execute(update_query, update_values)
        mydb.commit()

    return jsonify(response="Verification status updated.")
 
@app.route('/verifyID/<int:input_id>', methods=['GET'])
def verifyID(input_id):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()

    # Check if the entered ID exists in the accs_hist table
    mycursor.execute("SELECT COUNT(*) FROM accs_hist WHERE accs_prsn = %s", (input_id,))
    accs_count = mycursor.fetchone()[0]

    # Check if the entered ID exists in the prs_mstr table
    mycursor.execute("SELECT COUNT(*) FROM prs_mstr WHERE prs_nbr = %s", (input_id,))
    prs_count = mycursor.fetchone()[0]

    if accs_count > 0 and prs_count > 0:
        # Delete the access history for the given accs_prsn
        delete_query = "DELETE FROM accs_hist WHERE accs_prsn = %s"
        mycursor.execute(delete_query, (input_id,))
        mydb.commit()

        response_message = "Verification Successful"
    else:
        response_message = "Verification Failed"

    return jsonify({
        "message": response_message
     })

@app.route('/my_log')
def mylog():
    return render_template('user_login.html')

#kasun's session management
app.secret_key = 'a3c2ab9890b89eebb9c9ef3deacf2452'

@app.route('/user_login', methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        # Assuming user info comes in form data
        username = request.form.get('username')
        password = request.form.get('password')

               # Query MongoDB to validate user in the patient collection
        patient_user = regformdata.find_one({"email": username, "password": password})
        if patient_user:
            session['user_type'] = 'patient'
            session['patient_id'] = patient_user['patient_id']
            session['patient_name'] = patient_user['namenic']  # Assuming 'namenic' is the field for patient_name
            return redirect(url_for('mydash'))

        # Query MongoDB to validate user in the doctor collection
        doctor_user = doctors_col.find_one({"email": username, "password": password})
        if doctor_user:
            session['user_type'] = 'doctor'
            session['doctor_id'] = doctor_user['doctor_id']
            session['doctor_name'] = doctor_user['name']
            return redirect(url_for('docdash'))

    return render_template('user_login.html')


@app.route('/mydash')
def mydash():
    if 'patient_id' in session and 'patient_name' in session:
        return render_template('user_dashboard.html', patient_id=session['patient_id'], patient_name=session['patient_name'])
    else:
        return redirect(url_for('user_login'))

@app.route('/docdash')
def docdash():
    if 'doctor_id' in session and 'doctor_name' in session:
        return render_template('doc_dash.html', doctor_id=session['doctor_id'], doctor_name=session['doctor_name'])
    else:
        return redirect(url_for('user_login'))
    
import subprocess


    
if __name__ == '__main__':
    app.run(debug=True)
