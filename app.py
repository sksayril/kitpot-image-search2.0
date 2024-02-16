from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import pandas as pd

from model import load_model, find_similar_images,extract_filenames
from flask_restful import Resource, Api
import os

import difflib
from flask_cors import CORS
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
# import os
# import json
mongo_url = "mongodb://fastwhistlev4-live:fast5!fastwhistle@3.109.178.154:27017/"
# mongo_url = os.getenv('MONGO_URL')

database_name = "live-surat"
collection_name = "salesmanactivities"

pipeline = [
    {"$unwind": "$data.payment.oustandingInfo"},
    # {
    #     "$lookup": {
    #         "from": "collection_name",
    #         "localField": "data.distributorId",
    #         "foreignField": "_id",
    #         "as": "distributor",
    #     },
    # },
{
    "$project": {
        "distributor_id": "$data.payment.oustandingInfo.distributorId",
        "partyCode": "$data.payment.oustandingInfo.data.partyCode",
        "partyName": "$data.payment.oustandingInfo.data.partyName",
        "billNo": "$data.payment.oustandingInfo.data.billNo",
        "billDate": {
            "$dateFromString": {
                "dateString": "$data.payment.oustandingInfo.data.billDate",
            },
        },
        "billAmount": {
            "$convert": {
                "input": "$data.payment.oustandingInfo.data.billAmount",
                "to": "double",
                "onError": 0.0, 
                "onNull": 0.0   
            }
        },
        "days": {
            "$toInt": "$data.payment.oustandingInfo.data.days",
        },
        "paidAmount": {
            "$convert": {
                "input": "$data.payment.oustandingInfo.data.paidAmt",
                "to": "double",
                "onError": 0.0, 
                "onNull": 0.0   
            }
        },
        "reachTime": {
            "$toDate": "$data.reachedTime",
        },
    },
}

    
]

# Connect to MongoDB
client = MongoClient(mongo_url)
database = client[database_name]
collection = database[collection_name]

# Perform aggregation
result = list(collection.aggregate(pipeline))
import pandas as pd
df = pd.DataFrame(result)

# Close the MongoDB connection
client.close()
df['reachTime'] = pd.to_datetime(df['reachTime'])
df['day_of_week'] = df['reachTime'].dt.day_name()
df['date'] = df['reachTime'].dt.date
df['time'] = df['reachTime'].dt.time

app = Flask(__name__)
CORS(app)
model_filename = 'payment_prediction_model.pkl'
pipeline_filename = 'payment_prediction_pipeline.pkl'
pipeline = None
 
# Load and preprocess the data
df['invoiceDate'] = pd.to_datetime(df['billDate'], dayfirst=True)
df['paymentDate'] = pd.to_datetime(df['date'])
df['days_until_payment'] = (df['paymentDate'] - df['invoiceDate']).dt.days
df['invoiceMonth'] = df['invoiceDate'].dt.month
df['invoiceDayOfWeek'] = df['invoiceDate'].dt.dayofweek

def preprocess_data(df):
    """Preprocess the data for modeling."""
    features = ['distributor_id', 'partyCode', 'billAmount', 'invoiceMonth', 'invoiceDayOfWeek', 'timeOfDay']
    if not all(feature in df.columns for feature in features):
        raise ValueError("Dataframe must contain all required features: " + ", ".join(features))
    return df[features], df['days_until_payment']

# Preprocess reachTime to categorize into "Morning" or "Evening"
def categorize_time_of_day(reachTime):
    hour = reachTime.hour
    return "Morning" if hour < 12 else "Evening"
df['timeOfDay'] = df['reachTime'].apply(categorize_time_of_day)

average_payment_per_party = df.groupby('partyCode')['billAmount'].mean().reset_index()

def train_model(features, target):
    """Train the model and return the mean absolute error on the test set."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    numeric_features = ['billAmount', 'invoiceMonth', 'invoiceDayOfWeek']
    categorical_features = ['distributor_id', 'partyCode']
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    global pipeline  # Make pipeline a global variable
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    
    pipeline.fit(X_train, y_train)

    with open('payment_prediction_pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

    y_pred = pipeline.predict(X_test)
    return mean_absolute_error(y_test, y_pred)
# Function to calculate the next weekday
def calculate_next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days_ahead)

# Function to find the most common payment day
def find_common_payment_day(distributor_id, partyCode):
    filtered_data = df[(df['distributor_id'] == distributor_id) & (df['partyCode'] == partyCode)]
    if filtered_data.empty:
        return None, "No common payment day found"

    common_day = int(filtered_data['invoiceDayOfWeek'].mode()[0])
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return common_day, days[common_day]

@app.route('/train', methods=['POST'])
def train():
    features, target = preprocess_data(df)
    mae = train_model(features, target)
    return jsonify({"message": "Model trained successfully", "mean_absolute_error": mae})
def call_train_api():
    try:
        # Example of calling your /train API
        response = requests.post('http://43.205.191.73:5000/train')
        if response.status_code == 200:
            print("Training API called successfully")
        else:
            print("Failed to call training API")
    except Exception as e:
        print("Error calling training API:", e)

def run_script_py():
    try:
        # Run script.py file
        subprocess.run(["python", "autohit.py"])
    except Exception as e:
        print("Error running script.py:", e)

@app.route('/find_data', methods=['POST'])
def find_data():
    df = request.get_json()
    if 'distributor_id' not in df or 'date' not in df:
        return jsonify({'error': 'Missing distributor_id or date in request'}), 400

    distributor_id = df['distributor_id']
    date = df['date']
    
    # Path to the directory containing JSON files
    directory = 'resposedata'
    found_entries = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                file_data = json.load(file)
                for entry in file_data:
                    if entry.get('distributor_id') == distributor_id and \
                       (entry.get('next_common_payment_date') == date or \
                       entry.get('predicted_payment_date') == date):
                        found_entries.append(entry)

    # Add an additional feature to indicate if the predicted payment date matches the common payment day
    for entry in found_entries:
        if entry['predicted_payment_date'] and entry['common_payment_day_of_week']:
            predicted_payment_day = datetime.strptime(entry['predicted_payment_date'], '%Y-%m-%d').strftime('%A')
            entry['predicted_payment_matches_common_day'] = predicted_payment_day == entry['common_payment_day_of_week']
        else:
            entry['predicted_payment_matches_common_day'] = None

    return jsonify(found_entries)
@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({"error": "Model not trained. Please train the model first by calling the /train endpoint."})

    request_data = request.get_json()
    predictions = []
    for distributor_id, partyCode, invoiceDate, invoiceAmount in zip(
            request_data['distributor_ids'], request_data['partyCodes'],
            request_data['invoiceDates'], request_data['invoiceAmounts']):
        if partyCode not in df['partyCode'].unique():
            predictions.append({
                "distributor_id": distributor_id,
                "partyCode": partyCode,
                "invoiceDate": invoiceDate,
                "predicted_days_until_payment": "Not applicable",
                "predicted_payment_date": "Not applicable",
                "common_payment_day_of_week": "Not applicable",
                "avg_payment_amount": "Not applicable",
                "common_time_of_day": "Not applicable"
            })
            continue

        invoice_date_parsed = pd.to_datetime(invoiceDate)
        input_data = pd.DataFrame({
            'distributor_id': [distributor_id],
            'partyCode': [partyCode],
            'billAmount': [invoiceAmount],
            'invoiceMonth': [invoice_date_parsed.month],
            'invoiceDayOfWeek': [invoice_date_parsed.dayofweek],
            'timeOfDay': [df[df['partyCode'] == partyCode]['timeOfDay'].mode().iloc[0]]
        })

        predicted_days = pipeline.predict(input_data)[0]
        predicted_payment_date = invoice_date_parsed + timedelta(days=int(predicted_days))
        common_day_num, common_day_name = find_common_payment_day(distributor_id, partyCode)

        if common_day_name != "No common payment day found":
            actual_date_of_weekday = calculate_next_weekday(predicted_payment_date, common_day_num).strftime('%Y-%m-%d')
        else:
            actual_date_of_weekday = "Not applicable"

        avg_payment_amount = average_payment_per_party[average_payment_per_party['partyCode'] == partyCode]['billAmount'].iloc[0]

        predictions.append({
            "distributor_id": distributor_id,
            "partyCode": partyCode,
            "invoiceDate": invoiceDate,
            "predicted_days_until_payment": predicted_days,
            "predicted_payment_date": predicted_payment_date.strftime('%Y-%m-%d'),
            "common_payment_day_of_week": common_day_name,
            "actual_payment_date_based_on_weekday": actual_date_of_weekday,
            "avg_payment_amount": avg_payment_amount,
            "common_time_of_day": input_data['timeOfDay'].iloc[0]
        })

    return jsonify(predictions)

    # return jsonify({"predictions": predictions})
# Initialize scheduler

data = pd.read_pickle('kitpot.searchingdata14.pkl')
titles = data['DATA'].tolist()

lowercase_titles = [str(title).lower() if isinstance(title, str) else '' for title in titles]
data = list(filter(None, lowercase_titles))

def find_closest_match(input_word, data):
    if input_word in data:
        return [input_word]

    closest_match = difflib.get_close_matches(input_word, data)
    return closest_match

def process_search_term(search_term, data):
    words = search_term.split(' ')
    processed_words = []

    for word in words:
        
        if word.lower() in data:
            processed_words.append(word)
        else:
            closest_match = find_closest_match(word.lower(), data)
            processed_words.append(closest_match[0] if closest_match else word)

    processed_sentence = ' '.join(processed_words)
    return processed_sentence

@app.route('/get_closest_match', methods=['GET'])
def get_closest_match():
    search_term = request.args.get('search_key', default='', type=str)
    
    if not search_term:
        response = {
            'message': 'Search term is empty.',
        }
        return jsonify(response)

    processed_sentence = process_search_term(search_term, data)
    
    if processed_sentence:
        response = {
            'result': processed_sentence,
        }
    else:
        response = {
            'message': 'No similar words found.',
        }

    return jsonify(response)


@app.route('/similar_images', methods=['POST'])
def get_similar_images():
    if 'imageSearch' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['imageSearch']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        similar_images = find_similar_images(filepath)
        actualFormatedData=extract_filenames(similar_images)
        # actualFormatedData = convert_file_paths(similar_images)
        os.remove(filepath)
        
        return jsonify({'result': actualFormatedData})
    else:
        return jsonify({'error': 'File format not supported'})

scheduler = BackgroundScheduler()
# Schedule calling train API every 10 minutes
# scheduler.add_job(call_train_api, 'interval', minutes=10)
# Schedule running script.py every 20 minutes
# scheduler.add_job(run_script_py, 'interval', minutes=10)
# Start the scheduler
scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)

