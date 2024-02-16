import requests
from pymongo import MongoClient
import json
import os
from datetime import datetime

# MongoDB connection string
mongo_url = "mongodb://fastwhistlev4-live:fast5!fastwhistle@3.109.178.154:27017/"
database_name = "live-surat"
collection_name = "outstandings"

# Establish a connection to MongoDB
client = MongoClient(mongo_url)
db = client[database_name]
collection = db[collection_name]

# Define the aggregation pipeline
pipeline = [
    {
        "$project": {
            # "_id": 0,  # Exclude the _id field
            "distributorId": {"$toString": "$distributorId"},  # Convert ObjectId to string
            "partyCode": "$data.partyCode",
            "partyName": "$data.partyName",
            "billNo": "$data.billNo",
            "billAmount": "$data.billAmount",
            "billDate": {"$substr": ["$data.billDate", 0, 10]}  # Extract YYYY-MM-DD from the string
        }
    },
    {
        "$group": {
            "_id": "$distributorId",
            "distributor_ids": {"$push": "$distributorId"},
            "partyCodes": {"$push": "$partyCode"},
            "invoiceDates": {"$push": "$billDate"},
            "invoiceAmounts": {"$push": "$billAmount"}
        }
    }
]

# Perform aggregation
result = list(collection.aggregate(pipeline))

# Create a directory to store individual files
output_directory = "individual_results"
os.makedirs(output_directory, exist_ok=True)

# Iterate through the result and save each distributor's data into separate files
for distributor_data in result:
    distributor_id = distributor_data["_id"]
    output_file = os.path.join(output_directory, f"{distributor_id}.json")
    
    formatted_invoice_dates = []
    for date in distributor_data["invoiceDates"]:
        try:
            formatted_date = datetime.strptime(date, "%d-%m-%Y").strftime("%Y-%m-%d")
        except ValueError:
            formatted_date = date
        formatted_invoice_dates.append(formatted_date)
    
    formatted_result = {
        "distributor_ids": distributor_data["distributor_ids"],
        "partyCodes": distributor_data["partyCodes"],
        "invoiceDates": formatted_invoice_dates,
        "invoiceAmounts": distributor_data["invoiceAmounts"]
    }

    # Save the formatted result to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(formatted_result, json_file, indent=4)

    print(f"Result for distributor ID {distributor_id} saved to {output_file}")
    
    # HTTP request to the API
    api_url = "http://43.205.191.73:5000/predict"
    response = requests.post(api_url, json=formatted_result)
    
    # Save response data to a folder
    response_data_folder = "resposedata"
    os.makedirs(response_data_folder, exist_ok=True)
    response_file = os.path.join(response_data_folder, f"{distributor_id}.json")
    with open(response_file, "w") as response_json_file:
        json.dump(response.json(), response_json_file, indent=4)
    
    print(f"Response data for distributor ID {distributor_id} saved to {response_file}")
