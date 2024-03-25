from pymongo import MongoClient
from datetime import datetime
from pymongo.errors import DuplicateKeyError
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, AutoReconnect

class CrimeOperations:
    def addnewcrime(self,fname,lname,age,dob,addhar,gen,ph,hight,address,firno,CrimeType,nop,CrimeWeapons,cdate,Timeperiod,vehicle,vehicleNo,personsit,wearperson,discVehi,crimesp,pincd,discCrimeSP,Status,Landmark,Longitude,Latitude,PS):
        stat={}
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db=client["Crime"]
            coll=db["CrimeDetails"]
            # dobA = datetime.strptime(dob, '%Y-%m-%d') if dob else None
            # cdateA = datetime.strptime(cdate, '%Y-%m-%d') if cdate else None
            dic={}
            dic['First_Name']=fname
            dic['Last_Name']=lname
            dic['Age']=age
            # dic['Date-of-Birth']=dobA
            dic['Date-of-Birth']=dob
            # dic['Aadhar No.']=addhar
            dic['Aadhar_No']=addhar
            dic['Gender']=gen
            # dic['Phone No.']=ph
            dic['Phone_No']=ph
            dic['Height']=hight
            dic['Address']=address
            # dic['FIR No.']=firno
            dic['FIR_No']=firno
            dic['Type_of_Crime']=CrimeType
            dic['Number_Of_Person_Involve']=nop
            dic['Weapons_use_for_Crime']=CrimeWeapons
            # dic['Date_of_Crime']=cdateA
            dic['Date_of_Crime']=cdate
            dic['Time_Period_of_Crime']=Timeperiod
            dic['Vehicle_use']=vehicle
            # dic['Vehicle No.']=vehicleNo
            dic['Vehicle_No']=vehicleNo
            # dic['No. of Person  on Vehicle']=personsit
            dic['No_of_Person_on_Vehicle']=personsit
            dic['Discription_of_Person sitting_on_Vehicle']=wearperson
            dic['Discription_of_Vehicle']=discVehi
            dic['Crime_Spot']=crimesp
            dic['Pincode']=pincd
            dic['Discription_of Crime_Spot']=discCrimeSP
            dic['Status']=Status
            dic['Landmark']=Landmark
            dic['Longitude']=Longitude
            dic['Latitude']=Latitude
            dic['Police_Station']=PS
            coll.create_index('FIR_No', unique=True)
            coll.insert_one(dic)
            stat['msg']='Registered Successfully'
        except DuplicateKeyError as err:
            print("exception: "+str(err))
            stat['err']='FIR Number already exist'
        except Exception as err:
            print("exception: "+str(err))
            stat['err']='Oops! Something went wrong'
        return stat

    def count(self):
        stat={'msg':None,
        'err':None
        }
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db=client["Crime"]
            coll=db["CrimeDetails"]
            count = coll.count_documents({})
            print('count',count)
            detected = coll.count_documents({'Status':'Detected'})
            notdetected = coll.count_documents({'Status':'Not Detected'})
            
            stat['Crime_Registered'] = count
            stat['Crime_Detected'] = detected
            stat['Crime_Not_Detected'] = notdetected


            db=client["Crime"]
            coll=db["UserDetails"]

            
        except Exception as err:
            print("exception: "+str(err))
            stat['err']='Oops! Something went wrong, Check Your Internet Connection..!'
        return stat

    def searchC(self, key):
        stat = {'err':None}
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["CrimeDetails"]
            
            # Construct the query
            query = {"FIR_No": key}
            
            # # Find documents matching the query
            data = coll.find_one(query)
            
            if data is None:
                # If no documents match the query, return a message
                stat['err'] = "No matching documents found for FIR_No " + str(key)
            else:
                # Print the documents (for debugging purposes)
                stat['data'] = data       

            return stat
        except Exception as e:
            print("Exception in search: " + str(e))
        return

    def crimedata_list(request):
        stat = {'err':None}
        user_data = []
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["CrimeDetails"]
            users = coll
            user_data = users.find()
            stat['userdata'] = user_data
        except Exception as err:
            print("exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        # print("stat"+stat)
        return stat

    def criminals_list(self):
        stat = {'err': None, 'criminals_list': []}
        try:
            print("In model.py Criminals List")

             #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client['Crime']
            collection = db['CrimeDetails']

             # Specify the fields to retrieve in the projection
            projection = {'First_Name': 1, 'Last_Name': 1, 'Type_of_Crime': 1, 'FIR_No':1, 'Police_Station':1}

            # Fetch criminals list from MongoDB with the specified projection
            criminals = collection.find({}, projection)

            # Now 'criminals' contains a cursor with the data. You can convert it to a list if needed.
            criminals_list = list(criminals)
            # print("Criminals List:", criminals_list)

            # You may want to pass the 'criminals_list' to your template context
            stat['criminals_list'] = criminals_list
            #stat['criminals_list'] = criminals


        except ConnectionFailure as err:
            print("Connection Failure:", str(err))
            stat['err'] = 'ConnectionFailure: Unable to connect to the MongoDB server. Please check your connection settings.'

        except Exception as err:
            print("Exception in model.py:", str(err))
            stat['err'] = 'Oops! Something went wrong'

        finally:
            client.close()  # Close the MongoDB client

        return stat

    def criminal_detail(self,firstname,lastname):
        stat = {'err': None}
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client['Crime']
            coll = db['CrimeDetails']

            query = {"First_Name": firstname, "Last_Name": lastname}
            found_criminal = coll.find(query)

            all_data = []

            for x in found_criminal:
                all_data.append(x)

            stat['all_data'] = all_data

        except Exception as err:
            print("Exception in model.py:", str(err))
            stat['err'] = 'Oops! Something went wrong'

        finally:
            client.close()  # Close the MongoDB client

        return stat
    
    def crimesearch(self, key1, key2, key3):
        stat = {'err': None}
        dic = {}
        data = []

        try:
            # Connect to MongoDB
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            # Select the database and collection
            db = client["Crime"]
            coll = db["CrimeDetails"]
            print("Here inthe search")
            print(key1,key2,key3)
            print("============")

            # Construct the query
            query = {"First_Name": key1, "Last_Name": key2, "Type_of_Crime": key3}

            # Find documents matching the query
            cursor = coll.find(query)

            # Iterate through the matching documents and add them to the data list
            for document in cursor:
                data.append(document)

            # Print the number of documents found
            print("Number of matching documents:", len(data))

            if not data:
                # If no documents match the query, return a message
                stat['err'] = "No matching documents found"
            else:
                # Return the data
                stat['data'] = data

        except Exception as e:
            print("Exception in search:", str(e))
            stat['err'] = "An error occurred during the search."
        
        return stat
    
    def updateCriminalData(self,fname, lname, age, dob, addhar, gender, phone, hight, address, firno, CrimeType, nop, CrimeWeapons, cdate, Timeperiod, vehicle, vehicleNo, personsit, wearperson, discVehi, crimesp, pincd, discCrimeSP, Status, Landmark,Longitude,Latitude,PS):
        dic = []
        stat = {'err':None,'msg':None}
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
                # Select the database and collection
            db = client["Crime"]
            coll = db["CrimeDetails"]

            fn =int(firno)

            query = {'FIR_No':fn}
            # query = {'FIR_No':firno}

            new_values = {
            "$set": {
                "First_Name": fname.upper(),
                "Last_Name": lname.upper(),
                "Age": age,
                "Date_of_Birth": dob,
                "Aadhar_No": addhar,  # Replace space with underscore
                "Gender": gender.upper(),
                "Phone_No": phone,  # Replace space with underscore
                "Height": hight,
                "Address": address.upper(),
                "Type_of_Crime": CrimeType.upper(),  # Replace spaces with underscores
                "Number_Of_Person_Involve": nop,  # Replace spaces with underscores
                "Weapons_use_for_Crime": CrimeWeapons.upper(),  # Replace spaces with underscores
                "Date_of_Crime": cdate,
                "Time_Period_of_Crime": Timeperiod,  # Replace spaces with underscores
                "Vehicle_use": vehicle.upper(),  # Replace spaces with underscores
                "Vehicle_No": vehicleNo.upper(),  # Replace spaces with underscores
                "No_of_Person_on_Vehicle": personsit.upper(),  # Replace spaces with underscores
                "Discription_of_Person_sitting_on_Vehicle": wearperson.upper(),  # Replace spaces with underscores
                "Discription_of_Vehicle": discVehi.upper(),  # Replace spaces with underscores
                "Crime_Spot": crimesp.upper(),  # Replace underscore with space
                "Pincode": pincd,
                "Discription_of_Crime_Spot": discCrimeSP.upper(),  # Replace spaces with underscores
                "Status": Status,
                "Landmark": Landmark.upper(),
                "Longitude": Longitude,
                "Latitude": Latitude,
                "Police_Station": PS,
            }
            }    
            # Update the document
            result = coll.update_one(query, new_values)
            
            print("Matched Count:", result.matched_count)

            if result.matched_count > 0:
                    stat['msg'] = " Data Updated Successfully"
                    print(stat['msg'])
            else:
                    stat['err'] = "Data not found or status already updated"
            
            print("stat = ",stat)

        except Exception as err:
            print("Exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        finally:
            client.close()  # Close the MongoDB client

        return stat

    def searchfirno(self,fir_no):
            stat = {'err': None}
            dic = {}
            data = []

            try:
                # Connect to MongoDB
                #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
                # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
                client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
                # client=MongoClient("mongodb://localhost:27017/")
                # Select the database and collection
                db = client["Crime"]
                coll = db["CrimeDetails"]
                # print("FOR the Fir Number")

                # fn =  int(fir_no)
                fn = int(fir_no)
                # Construct the query
                query = {"FIR_No":fn}

                # Find documents matching the query
                document = coll.find_one(query)
                # print("Cursor down", cursor)

                # Iterate through the matching documents and add them to the data list
                # for document in cursor:
                #     # print("Cursor in ")
                #     # print("\n",document)
                #     data.append(document)

                # Print the data (for debugging purposes
                    # print("for a in data")
                    # print("Data Array:", data)
                if not document:
                    # If no documents match the query, return a message
                    stat['err'] = "No matching documents found"
                else:
                    # Return the data
                    stat['data'] = document
                    print("document = ",document)
                    print("stat = ",stat)


                return stat

            except Exception as e:
                print("Exception in search: " + str(e))
                stat['err'] = "An error occurred during the search."
            
            return stat
    
    def genFirno(self):
        stat={
            'err': None
        }
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db=client["Crime"]
            coll=db["CrimeDetails"]
            
            document_with_max_firno = coll.find_one(sort=[('FIR_No', -1)])

            # Access the firno value
            greatest_firno = document_with_max_firno['FIR_No'] if document_with_max_firno else None

            print(f"The greatest firno is: {greatest_firno} {type(greatest_firno)}")
            greatest_firno = int(greatest_firno) + 1
            stat['firno']=greatest_firno
            
        except Exception as err:
            print("exception: "+str(err))
            stat['err']='Oops! Something went wrong, Check Your Internet Connection..!'
        return stat

    def delete_criminal_data(self,fir_no):
        stat = {'err': None}
            

        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            # Select the database and collection
            db = client["Crime"]
            coll = db["CrimeDetails"]
            # print("FOR the Fir Number")

            # fn =  int(fir_no)
            fn = int(fir_no)
            # Construct the query
            query = {"FIR_No":fn}

            # Find documents matching the query
            result = coll.delete_one(query)
            if result.deleted_count == 1:
                print("Document deleted successfully")
                stat['msg'] = "Document deleted Successfully"
            else:
                print("Document not found or not deleted")
                stat['msg'] = "Document not found or not deleted"


            return stat

        except Exception as e:
            print("Exception in search: " + str(e))
            stat['err'] = "An error occurred during the search."
                
        return stat

    def searchC2(self, text):
        stat = {'err': None}
        result = []
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            collection = db["CrimeDetails"]
            # Construct a regex pattern for case-insensitive search
            # pattern = f".*{text}.*"
            pattern = fr"\b{text}\b"
            # Specify the fields you want to retrieve
            projection = {
                "First_Name": 1,
                "Last_Name": 1,
                "Type_of_Crime": 1,
                "FIR_No": 1,
                "Police_Station": 1,
                # Add more fields as needed
            }
            # Perform the search using regex and apply projection
            query = {
                "$or": [
                    {"First_Name": {"$regex": pattern, "$options": "i"}},
                    {"Last_Name": {"$regex": pattern, "$options": "i"}},
                    {"Type_of_Crime": {"$regex": pattern, "$options": "i"}},
                    {"Police_Station": {"$regex": pattern, "$options": "i"}},
                ]
            }
            # Execute the query with projection
            result1 = collection.find(query, projection)
            # Convert result1 to a list
            result = list(result1)
            print("result (model) = ", result)
            stat['result'] = result
        except Exception as e:
            print("Exception in search: " + str(e))
            stat['err'] = "An error occurred during the search."
        return stat

    def statuschanged(self, firno, param):
        result = None
        
        try:
            print("In the model of statrus")
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["CrimeDetails"]
            filter_criteria = {"FIR_No": firno}

            print(param)
            
            if param == "Detected":
                update_data = {"$set": {'Status': 'Detected'}}
                print("Detected")
            else:
                update_data = {"$set": {'Status': 'Not Detected'}}
                print("Not Detected")

            result = coll.update_one(filter_criteria, update_data)

            if result.modified_count > 0:
                msg = "Update successful"
            else:
                msg = "No documents were modified"

        except ConnectionFailure as cf_error:
            print(f"Connection error: {cf_error}")
            msg = "Error connecting to MongoDB"
        except Exception as generic_error:
            print(f"Exception: {generic_error}")
            msg = "Oops! Something went wrong"
        finally:
            if client:
                client.close()  # Close the MongoDB client

        return result, msg

    def getpolicestation(self):
        stat = {'err': None, 'criminals_list': []}
        try:
            print("In model.py Criminals List")

            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client['Crime']
            collection = db['CrimeDetails']

             # Specify the fields to retrieve in the projection
            projection = {'Police_Station':1}

            # Fetch criminals list from MongoDB with the specified projection
            criminals = collection.find({}, projection)

            # Now 'criminals' contains a cursor with the data. You can convert it to a list if needed.
            criminals_list = list(criminals)
            # print("Criminals List:", criminals_list)

            # You may want to pass the 'criminals_list' to your template context
            stat['criminals_list'] = criminals_list
            #stat['criminals_list'] = criminals


        except ConnectionFailure as err:
            print("Connection Failure:", str(err))
            stat['err'] = 'ConnectionFailure: Unable to connect to the MongoDB server. Please check your connection settings.'

        except Exception as err:
            print("Exception in model.py:", str(err))
            stat['err'] = 'Oops! Something went wrong'

        finally:
            client.close()  # Close the MongoDB client

        return stat

    def criminals_list_as_per_police_station(self, selected_station):
        stat = {'err': None, 'criminals_list': []}
        try:
            print("In model.py Criminals List")

            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client['Crime']
            collection = db['CrimeDetails']

             # Specify the fields to retrieve in the projection
            projection = {'First_Name': 1, 'Last_Name': 1, 'Type_of_Crime': 1, 'Police_Station':1}

            # Fetch criminals list from MongoDB with the specified projection
            criminals = collection.find({'Police_Station': selected_station}, projection)

            # Now 'criminals' contains a cursor with the data. You can convert it to a list if needed.
            criminals_list = list(criminals)
            # print("Criminals List:", criminals_list)

            # You may want to pass the 'criminals_list' to your template context
            stat['criminals_list'] = criminals_list
            #stat['criminals_list'] = criminals


        except ConnectionFailure as err:
            print("Connection Failure:", str(err))
            stat['err'] = 'ConnectionFailure: Unable to connect to the MongoDB server. Please check your connection settings.'

        except Exception as err:
            print("Exception in model.py:", str(err))
            stat['err'] = 'Oops! Something went wrong'

        finally:
            client.close()  # Close the MongoDB client

        return stat

class UserOperations:
    
    def register(self,fname,lname, uname, email, pswd, policestation):
        stat = {}
        try:
            print("hello")
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["UserDetails"]
            cdate = datetime.now()
            dic = {}
            dic['FName'] = fname
            dic['LName'] = lname
            dic['Username'] = uname
            dic['Email'] = email
            dic['Password'] = pswd
            dic['Police_Station'] = policestation
            # dic['Country'] = country
            dic['Date'] = cdate
            dic['Status'] = 'Activated'
            dic['utype'] = 'normal'

            existing_user = coll.find_one({"Username": uname})
            existing_email = coll.find_one({"Email": email})

            if existing_user or existing_email:
                stat['err'] = 'User already exists... Please try to Login'
            else:
                coll.insert_one(dic)
                stat['msg'] = 'User Registered Successfully'
        except Exception as err:
            print("exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        return stat
        
    
    def login(self, uname, pswd):
        stat = {
            'msg':None,
            'err':None
        }
        try:
            # Connect to the MongoDB server
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            
            # Select the database and collection
            db = client["Crime"]
            coll = db["UserDetails"]

            # Check if the user exists in the database
            existing_user = coll.find_one({"Username": uname, "Password": pswd})

            if existing_user:
                # User is found, you can consider them logged in
                print(existing_user['Status'])
                if existing_user['Status'] == "Activated":
                    stat['msg'] = 'Login successful'
                    stat['fname'] = existing_user['FName']
                    stat['policestation'] = existing_user['Police_Station']
                else:
                    stat['err'] = 'Account is Deactivated'
                    print(stat)
                    return stat
            else:
                # User not found, login failed
                stat['err'] = 'Invalid credentials'

        except Exception as err:
            print("exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'

        return stat
    
    
    def updateUserData(self, uname, email, password, policestation):
        stat = {}
        stat['err'] = None
        stat['msg'] = None

        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["UserDetails"]

            query = {'Username': uname}
            update_data = {'$set': {'Email': email, 'Password': password,'Police_Station':policestation}}

            print("Query:", query)
            print("Update Data:", update_data)

            result = coll.update_one(query, update_data)

            print("Matched Count:", result.matched_count)

            if result.matched_count > 0:
                stat['msg'] = "User Updated Successfully"
            else:
                stat['err'] = "User not found or status already updated"

        except Exception as err:
            print("Exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        finally:
            client.close()  # Close the MongoDB client

        return stat

    
    def userdata_list(request):
        stat = {'err':None}
        user_data = []
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["UserDetails"]
            users = coll
            user_data = users.find()
            stat['userdata'] = user_data
        except Exception as err:
            print("exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        # print("stat"+stat)
        return stat

    def updateStatus(self, uname, ustatus):
        stat = {}
        stat['err'] = None
        stat['msg'] = None
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["UserDetails"]

            filter_criteria = {"Username": uname}
            if ustatus == "Activated":
                update_data = {"$set": {"Status": "Deactivated"}}
            else:
                update_data = {"$set": {"Status": "Activated"}}

            result = coll.update_one(filter_criteria, update_data)

            if result.matched_count > 0:
                stat['msg'] = "Success"
            else:
                stat['err'] = "User not found or status already updated"

        except Exception as err:
            print("exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        finally:
            client.close()  # Close the MongoDB client

        return stat

    def updateType(self, uname, ustatus):
        stat = {}
        stat['err'] = None
        stat['msg'] = None
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["UserDetails"]

            filter_criteria = {"Username": uname}
            update_data = {"$set": {"utype": ustatus}}

            result = coll.update_one(filter_criteria, update_data)

            if result.matched_count > 0:
                stat['msg'] = "Success"
            else:
                stat['err'] = "User not found or status already updated"

        except Exception as err:
            print("exception: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        finally:
            client.close()  # Close the MongoDB client

        return stat
     
    def getUserData(self, uname):
        stat = {}
        stat['err'] = None
        stat['msg'] = None
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["UserDetails"]

            query = {'Username': uname}

            result = coll.find(query)

            print(result)
            stat['user']=list(result)[0]
            print(stat)
        except Exception as err:
            print("exception getUserData model: " + str(err))
            stat['err'] = 'Oops! Something went wrong'
        finally:
            client.close()  # Close the MongoDB client

        return stat
 
  
    def checkType(self, name):
        default_utype = None
        # client = None
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            db = client["Crime"]
            coll = db["UserDetails"]

            default_utype = 'normal'

            result = coll.find_one({"Username": name})

            if result:
                utype = result.get("utype", default_utype)
                print(f"The 'utype' value for Username '{name}' is: {utype}")
            else:
                print(f"No document found for Username '{name}'.")
                utype = default_utype
            
        except Exception as err:
            print("Exception: " + str(err))
            utype = default_utype
            
        return utype

    def checkStatus(self, name):
        try:
            #client = MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
            # client = MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            client = MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
            # client=MongoClient("mongodb://localhost:27017/")
            
            db = client["Crime"]
            coll = db["UserDetails"]

            default_status = 'Activated'

            result = coll.find_one({"Username": name})

            if result:
                status = result.get("Status", default_status)
                print(f"The 'Status' value for Username '{name}' is: {status}")
            else:
                print(f"No document found for Username '{name}'.")
                status = default_status

        except Exception as err:
            print("Exception: " + str(err))
            status = default_status
        finally:
            client.close()  # Close the MongoDB client

        return status
