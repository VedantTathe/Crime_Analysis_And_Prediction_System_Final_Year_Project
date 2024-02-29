from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
db = client["Crime"]
collection = db["UserDetails"]

# Define the new attribute and its value
new_attribute = "Status"
new_value = "Activated"

# Update all existing records to add the new attribute
collection.update_many({}, {"$set": {new_attribute: new_value}})

# Close the connection
client.close()
