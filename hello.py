# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# # Load the training data
# df = pd.read_csv('training.csv', names=['description', 'type'], header=None)

# # Create the TF-IDF vectorizer and transform the training data
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

# # Train the SVC model with the RBF kernel
# svm_type_classifier = SVC(kernel='rbf')
# svm_type_classifier.fit(tfidf_matrix, df['type'])

# # Save the trained model
# joblib.dump((svm_type_classifier, tfidf_vectorizer), 'trained_model.joblib')


# import joblib

# # Load the pre-trained model
# svm_type_classifier, tfidf_vectorizer = joblib.load('trained_model.joblib')

# def classify_description(description):
#     tfidf_new = tfidf_vectorizer.transform([description])
#     predicted_type = svm_type_classifier.predict(tfidf_new)[0]
#     return predicted_type

# import re

# def preprocess(text):
#     text = re.sub(r'[^\w\s\']',' ',text)
#     text = re.sub(r' +',' ',text)
#     return text.strip().lower()

# text_predict = """In a quiet suburban neighborhood, a daring burglary unfolded under the cover of darkness. A skilled thief, equipped with tools of the trade, discreetly 
# entered a residence through an unlocked back door. Moving with calculated precision, the intruder swiftly ransacked the home, absconding with valuable jewelry and electronic devices.
# The residents, unaware of the intrusion until morning, were left shocked and violated by the audacious crime that had taken place within the sanctity of their own home. 
# Law enforcement was promptly notified, initiating an investigation into the burglary that left the community on edge."""

# text_predict = preprocess(text_predict)
# text_predict = text_predict.replace("\n", " ")
# # text_predict = "attempted to murder a woman from a car driver"

# predicted_type = classify_description(text_predict)

# # Display the predicted Category and Type
# print(f"Predicted Type: {predicted_type}")




# import pyttsx3


# def Speak(Text):
#     engine = pyttsx3.init("sapi5")
#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[0].id)
#     engine.setProperty('rate', 170)
#     print("")
#     print(f"You : {Text}.")
#     print("")
#     engine.say(Text)
#     engine.runAndWait()

# from selenium import webdriver
# from selenium.webdriver.support.ui import Select
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from time import sleep


# chrome_options = Options()
# chrome_options.add_argument('--log-level=3')
# chrome_options.headless = False

# # Update the path to your Chrome executable
# Path = "Database\\chromedriver.exe"
# driver = webdriver.Chrome(Path, options=chrome_options)
# driver.maximize_window()

# website = r"https://ttsmp3.com/text-to-speech/British%20English/"
# driver.get(website)
# ButtonSelection = Select(driver.find_element(by=By.XPATH,value='/html/body/div[4]/div[2]/form/select'))
# ButtonSelection.select_by_visible_text('British English / Brian')
# #Speak("Hello, I am speaking using text-to-speech!")

# def Speak(Text):
    
#     lengthoftext = len(str(Text))

#     if lengthoftext == 0:
#         pass

#     else:
#         print("")
#         print(f"AI : {Text}")
#         print("")
#         Data = str(Text)
#         xpathofsec = '/html/body/div[4]/div[2]/form/textarea'
#         driver.find_element(By.XPATH,value=xpathofsec).send_keys(Data)
#         driver.find_element(By.XPATH,value='//*[@id="vorlesenbutton"]').click()
#         driver.find_element(By.XPATH,value="/html/body/div[4]/div[2]/form/textarea").clear()

#         if lengthoftext>=30:
#             sleep(4)

#         elif lengthoftext>=40:
#             sleep(6)

#         elif lengthoftext>=55:
#             sleep(8)

#         elif lengthoftext>=70:
#             sleep(10)

#         elif lengthoftext>=100:
#             sleep(13)

#         elif lengthoftext>=120:
#             sleep(14)

#         else:
#             sleep(2)


# Speak("Welcome Vedant Sir")

# from pymongo import MongoClient

# # Connect to the MongoDB servers
# # source_client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/")
# # destination_client = MongoClient("mongodb+srv://himanshu:himanshu@cluster0.gaoteru.mongodb.net/")
# source_client = MongoClient("mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/Crime")
# destination_client = MongoClient("mongodb+srv://himanshu:himanshu@cluster0.gaoteru.mongodb.net/Crime")

# # Select the source and destination databases and collections
# source_db = source_client['Crime']
# destination_db = destination_client['Crime']

# source_collection = source_db['CrimeDetails']
# destination_collection = destination_db['CrimeDetails']

# i=1
# # Iterate through each document in the source collection and insert it into the destination collection
# for document in source_collection.find():
#     destination_collection.insert_one(document)
#     print(i)
#     i = i+1

# # Close the MongoDB connections
# source_client.close()
# destination_client.close()


# import pymongo

# # Connect to MongoDB
# client = pymongo.MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/")
# db = client["Crime"]
# collection = db["CrimeDetails"]

# # Update documents
# update_result = collection.update_many(
#     {"Landmark": "Select"},
#     {"$set": {"Landmark": "SELECT"}}
# )

# # Print the number of documents updated
# print(f"Number of documents updated: {update_result.modified_count}")

# # Close the MongoDB connection
# client.close()

# import pymongo
# import random

# # Connect to MongoDB
# client = pymongo.MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
# db = client["Crime"]
# collection = db["CrimeDetails"]

# # List of police stations
# # police_stations = ['BHOSARI', 'BHOSARI MIDC', 'CHIKHALI','PIMPRI','CHINCHWAD','NIGADI','CHAKAN','ALANDI','DIGHI','MHALUNGE','SANGVI','WAKAD','HINJEWADI','RAVET','DEHUROAD','TALEGAON DABHADE','SHIRGAON','TALEGAON MIDC']

# # Update documents with a random police station
# # for document in collection.find():
# #     random_station = random.choice(police_stations)
# #     collection.update_one(
# #         {"_id": document["_id"]},
# #         {"$set": {"Police_Station": random_station}}
# #     )

# # # Print a message indicating the update
# # print(f"Police Station field added to all documents with random values.")
# query = {"Date_of_Crime": {"$regex": "^2018-"}}
# update = {"$set": {"Date_of_Crime": "2024-01-14"}}

# result = collection.update_many(query, update)

# print(f"Matched {result.matched_count} documents and modified {result.modified_count} documents.")
# # Close the MongoDB connection
# client.close()
# from pymongo import MongoClient 
# import ssl

# def copy_collection(source_uri, destination_uri, source_db, destination_db, collection_name, dest_coll):
#     # Connect to the source MongoDB
#     # source_client = MongoClient(source_uri)
#     source_client = MongoClient(source_uri)

#     source_database = source_client[source_db]
#     source_collection = source_database[collection_name]

#     # Connect to the destination MongoDB
#     destination_client = MongoClient(destination_uri)
#     destination_database = destination_client[destination_db]
#     destination_collection = destination_database[dest_coll]

#     # Copy documents from source collection to destination collection
#     destination_collection.insert_many(source_collection.find())     

#     # Close connections
#     source_client.close()
#     destination_client.close()

# if __name__ == "__main__":
#     # Specify your source and destination MongoDB URIs
#     source_mongodb_uri = "mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/?retryWrites=true&w=majority"
#     destination_mongodb_uri = "mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority"
    
#     # source_mongodb_uri = "mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority"
#     # destination_mongodb_uri = "mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/?retryWrites=true&w=majority"

#     # Specify the source and destination databases and collection name
#     source_database_name = "Crime1"
#     destination_database_name = "Crime"
#     collection_to_copy = "UserDetails1"
#     dest_coll = "UserDetails"

#     # Copy the collection
#     copy_collection(source_mongodb_uri, destination_mongodb_uri, source_database_name, destination_database_name, collection_to_copy,dest_coll)

#     print(f"Collection '{collection_to_copy}' copied from '{source_mongodb_uri}' to '{destination_mongodb_uri}'.")



# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient('mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority')  # Update the connection string with your MongoDB URI
# db = client['Crime']  # Replace 'your_database_name' with your actual database name
# collection = db['CrimeDetails']  # Replace 'your_collection_name' with your actual collection name

# # Retrieve all records from the collection
# all_records = list(collection.find())

# # Identify the records to keep (first 96 records)
# records_to_keep = all_records[:96]

# # Delete the remaining records from the collection
# for record in all_records[96:]:
#     collection.delete_one({'_id': record['_id']})

# print("Remaining records after deletion:", collection.count_documents({}))



# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient('mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority') 
# database = client['Crime']  # Replace 'your_database_name' with your actual database name
# collection = database['CrimeDetails']  # Replace 'your_collection_name' with your actual collection name

# # Get all documents from the collection
# documents = collection.find()

# # Update the firno field for each document
# for index, document in enumerate(documents, start=1):
#     # Update the firno field with the incremented value
#     collection.update_one(
#         {'_id': document['_id']},  # Assuming '_id' is the unique identifier field
#         {'$set': {'FIR_No': index}}
#     )
#     print(document['_id'])

# print('Firno field updated successfully.')

# # Close the MongoDB connection
# client.close()


# from pymongo import MongoClient 
# import ssl

# def copy_collection(source_uri, destination_uri, source_db, destination_db, collection_name, dest_coll):
#     # Connect to the source MongoDB
#     # source_client = MongoClient(source_uri)
#     source_client = MongoClient(source_uri)

#     source_database = source_client[source_db]
#     source_collection = source_database[collection_name]

#     # Connect to the destination MongoDB
#     destination_client = MongoClient(destination_uri)
#     destination_database = destination_client[destination_db]
#     destination_collection = destination_database[dest_coll]

#     # Copy documents from source collection to destination collection
#     destination_collection.insert_many(source_collection.find())     

#     # Close connections
#     source_client.close()
#     destination_client.close()

# if __name__ == "__main__":
#     # Specify your source and destination MongoDB URIs
#     source_mongodb_uri = "mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority"
#     destination_mongodb_uri = "mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority"
    
#     # source_mongodb_uri = "mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority"
#     # destination_mongodb_uri = "mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/?retryWrites=true&w=majority"

#     # Specify the source and destination databases and collection name
#     source_database_name = "Crime"
#     destination_database_name = "Crime"
#     collection_to_copy = "CrimeDetails"
#     dest_coll = "CrimeDetails"

#     # Copy the collection
#     copy_collection(source_mongodb_uri, destination_mongodb_uri, source_database_name, destination_database_name, collection_to_copy,dest_coll)

#     print(f"Collection '{collection_to_copy}' copied from '{source_mongodb_uri}' to '{destination_mongodb_uri}'.")




















# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient('mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority')
# db = client['Crime']
# collection = db['CrimeDetails']

# # Define a function to check null, None, or invalid values
# def check_invalid_values():
#     invalid_documents = []

#     # Query the collection
#     cursor = collection.find()

#     # Iterate through the documents
#     for doc in cursor:
#         if doc['Longitude'] is None or doc['Latitude'] is None or doc['Pincode'] is None:
#             invalid_documents.append(doc)
#         elif not isinstance(doc['Longitude'], (float, int)) or not isinstance(doc['Latitude'], (float, int)) or not isinstance(doc['Pincode'], str):
#             invalid_documents.append(doc)
    
#     return invalid_documents

# # Count or list the documents with null, None, or invalid values
# invalid_documents = check_invalid_values()
# print("Number of documents with null, None, or invalid values:", len(invalid_documents))
# for doc in invalid_documents:
#     print(doc)



# from pymongo import MongoClient
# import pandas as pd

# # Connect to MongoDB
# client = MongoClient('mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority')
# db = client['Crime']
# collection = db['CrimeDetails']

# # Retrieve data from MongoDB and load it into a DataFrame
# data = list(collection.find())
# df = pd.DataFrame(data)

# # Define the path for the CSV file
# csv_file_path = 'data.csv'

# # Write the DataFrame to a CSV file
# df.to_csv(csv_file_path, index=False)

# print("CSV file has been created successfully.")


# import pandas as pd
# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient('mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority')
# db = client['Crime']  # Replace 'your_database' with your database name
# collection = db['CrimeDetails']  # Replace 'your_collection' with your collection name

# # Read the CSV file into a pandas DataFrame
# csv_file = 'output1.csv'  # Replace 'your_file.csv' with the path to your CSV file
# df = pd.read_csv(csv_file)

# # Convert the DataFrame to a list of dictionaries (each dictionary represents a document)
# data = df.to_dict(orient='records')

# # Insert the data int   o MongoDB
# collection.insert_many(data)

# # Close the MongoDB connection
# client.close()

# print("Data imported successfully!")


# def get_unique_key(d):
#     return (d['Longitude'], d['Latitude'], d['Pincode'])

# # Your list of dictionaries
# list_of_dicts = [{'Landmark': 'MALL', 'Longitude': 73.79484777389996, 'Latitude': 18.574327203299998, 'Pincode': 427381.36, 'Crime': 'Pocket Theft'}, {'Landmark': 'MALL', 'Longitude': 73.79484777389996, 'Latitude': 18.574327203299998, 'Pincode': 427381.36, 'Crime': 'Chain Theft'}, {'Landmark': 'MALL', 'Longitude': 73.79484777389996, 'Latitude': 18.574327203299998, 'Pincode': 427381.36, 'Crime': 'Bicycle Theft'}, {'Landmark': 'MALL', 'Longitude': 73.79484777389996, 'Latitude': 18.574327203299998, 'Pincode': 427381.36, 'Crime': 'Two-wheeler Theft'}, {'Landmark': 'MALL', 'Longitude': 73.79484777389996, 'Latitude': 18.574327203299998, 'Pincode': 427381.36, 'Crime': 'Four-wheeler Theft'}, {'Landmark': 'MALL', 'Longitude': 73.79484777389996, 'Latitude': 18.574327203299998, 'Pincode': 427381.36, 'Crime': 'Other Vehicle Theft'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Vehicle Parts Theft'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Other Theft'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Commercial Robbery'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Technical Robbery'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Preparing to Robbery'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Other Robbery'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Daytime Burglary'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Night Burglary'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Culpable Homicide'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Forcible Theft'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Rape'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Murder'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Attempt to Murder'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Betrayal'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Riot'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Injury'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Molestation'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Gambling'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Prohibition'}, {'Landmark': 'OTHER PLACES', 'Longitude': 73.80036191069996, 'Latitude': 18.578226061899997, 'Pincode': 427411.98, 'Crime': 'Other'}]

# # Use a set to keep track of unique keys
# unique_keys = set()

# # List to store unique dictionaries
# unique_dicts = []

# for d in list_of_dicts:
#     key = get_unique_key(d)
#     if key not in unique_keys:
#         unique_keys.add(key)
#         unique_dicts.append(d)

# print(unique_dicts)



import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority')
db = client['Crime']  # Replace 'Crime' with your database name
collection = db['CrimeDetails']  # Replace 'CrimeDetails' with your collection name

# Read the CSV file into a pandas DataFrame with the appropriate encoding
csv_file = 'data_csv_files/10000ready.csv'  # Replace 'crimedata.csv' with the path to your CSV file
df = pd.read_csv(csv_file, encoding='ISO-8859-1')

# Convert the DataFrame to a list of dictionaries (each dictionary represents a document)
data = df.to_dict(orient='records')

# Insert the data into MongoDB
collection.insert_many(data)

# Close the MongoDB connection
client.close()

print("Data imported successfully!")
