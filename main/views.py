from django.shortcuts import render, redirect
from .models import CrimeOperations
from .models import UserOperations
from django.contrib.auth.decorators import login_required
import json
import re
import joblib
from django.conf import settings
import os
from django.http import HttpResponse
import pyttsx3
import speech_recognition as sr
from time import sleep
import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np    
from datetime import datetime
from django.http import JsonResponse
from sklearn.metrics import mean_squared_error
from collections import Counter
import threading
from datetime import datetime, timedelta
from django.core.paginator import Paginator

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense




def speak(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 170)
    print("")
    print(f"You : {Text}.")
    print("")
    engine.say(Text)
    engine.runAndWait()

def speechrecognition(text):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.....")
        speak(text)
        print(text)
        r.pause_threshold = 1
        audio = r.listen(source,0,8)

    try:
        print("Recognizing....")
        speak("Recognizing.....")
        query = r.recognize_google(audio,language="en")

        print(f"==> Vedant : {query}")
            
         
        if "skip" in query:
            return "skip"
        
        if "blank" in query or "empty" in query:
            return ""        
            
        if "exit" in query:
            return "terminate"
            
        return query.upper()
    except:
        speak("Dose Not Recognized Properly, Please Repeat Again.....")
        speechrecognition(text)

def readFromVoiceSeparate1(request):
    data = {}
    speak("Welcome Sir, I am Sarthi")
    sleep(0.5)
    speak("Your Voice Assistant")
    sleep(0.5)
    speak("Please tell me following details...")
    txt = speechrecognition("Listeneing for First Name")
    data['fname'] = txt
    txt = speechrecognition("Listeneing for Last Name")
    data['lname'] = txt
    try:
            
        txt = speechrecognition("Listeneing for Age")
        data['age'] = txt
    except:
        print("invalid age..")
    txt = speechrecognition("Listeneing for Date of Birth")
    data['dob'] = txt
    txt = speechrecognition("Listeneing for Gender")
    data['gender'] = txt
    try:
        txt = speechrecognition("Listeneing for Phone Number")
        txt = txt.replace(" ", "")
        if len(txt) < 10:
            need = 10-len(txt)
            txt = txt[:len(txt)]+"0"*need
        elif len(txt) > 10:
            txt = txt[:9]
        
    except:
        print("invalid phone number")
    data['phone'] = txt
    txt = speechrecognition("Listeneing for Height")
    data['height'] = txt
    txt = speechrecognition("Listeneing for Address")
    data['addrs'] = txt
    txt = speechrecognition("Listeneing for FIR Number")
    data['FIR_No'] = txt
    txt = speechrecognition("Listeneing for Type of Crime")
    data['Type_of_Crime'] = txt
    txt = speechrecognition("Listeneing for Number of Person Involve")
    data['Number_Of_Person_Involve'] = txt
    txt = speechrecognition("Listeneing for Weapons use for Crime")
    data['Weapons_use_for_Crime'] = txt
    txt = speechrecognition("Listeneing for Date of Crime")
    data['Date_of_Crime'] = txt
    txt = speechrecognition("Listeneing for Time Period of Crime")
    data['Time_Period_of_Crime'] = txt
    txt = speechrecognition("Listeneing for Vehicle Used")
    data['Vehicle_use'] = txt
    txt = speechrecognition("Listeneing for Vehicle Number")
    data['Vehicle_No'] = txt
     

    return data


def readFromVoiceSeparate(request,param):
    data = {}
    
    txt = speechrecognition("Listeneing for "+param)
    if txt == None:
        txt = ""

    if "Phone" in param or "Age" in param or "Aadhar-Number" in param or "Height" in param or "F-I-R-Number" in param or "Number-Of-Person-Involved" in param:
        txt = txt.replace(" ", "")
        txt = re.findall(r'\d+', txt)

        # Convert the list of strings to a list of integers
        txt = [int(num) for num in txt]
        
    
    print("PARAM:- ",param)
    print("VALUE:- ",txt)

    data['query'] = txt

    return JsonResponse(data)

def readDescriptionFromVoice(request):
    data={}
    stat={}
    text = ""
    text_predict = ""
    if request.POST.get('submitbtn'):
        text_predict = request.POST.get('crime_description')
        text = text_predict
        print("text = ",text)
    else:
        text = speechrecognition("Listening For Crime Description")
        text_predict = text
    # Load the pre-trained model
    svm_type_classifier, tfidf_vectorizer = joblib.load('C:/CrimeAnalysis_ML/trained_model.joblib')
    def classify_description(description):
        tfidf_new = tfidf_vectorizer.transform([description])
        predicted_type = svm_type_classifier.predict(tfidf_new)[0]
        return predicted_type
    def preprocess(text):
        text = re.sub(r'[^\w\s\']',' ',text)
        text = re.sub(r' +',' ',text)
        return text.strip().lower()
    # text_predict = """In a quiet suburban neighborhood, a daring burglary unfolded under the cover of darkness. A skilled thief, equipped with tools of the trade, discreetly 
    # entered a residence through an unlocked back door. Moving with calculated precision, the intruder swiftly ransacked the home, absconding with valuable jewelry and electronic devices.
    # The residents, unaware of the intrusion until morning, were left shocked and violated by the audacious crime that had taken place within the sanctity of their own home. 
    # Law enforcement was promptly notified, initiating an investigation into the burglary that left the community on edge."""
    text_predict = preprocess(text_predict)
    text_predict = text_predict.replace("\n", " ")
    # text_predict = "attempted to murder a woman from a car driver"
    predicted_type = classify_description(text_predict)
    # Display the predicted Category and Type
    
    predicted_type = predicted_type.upper()
    data['type'] = predicted_type
    
    print(f"Predicted Type: {predicted_type}")
    ####################Crime Description Data Extraction
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # text = """In the quiet neighborhood of Maple Street, a heinous crime occurred. The suspect, John Doe, a 35 year old male with a date of birth on 4/15/1987, was involved in an armed robbery. The crime took place on 7/12/2023 between 2 PM and 4 PM.
    # The suspect, armed with a firearm, robbed a convenience store at 123 Elm Street. The FIR number for the case is FIR-2023-567. Two other individuals were part of the crime, one wielding a knife and the other acting as the getaway driver.
    # The getaway vehicle, a red sedan with license plate number ABC-123, had three people inside. The driver was a tall man of 165cm (5'5"), wearing a black cap and sunglasses. The passenger in the front seat was a woman with a blue scarf, and in the back seat, there was a man in a green jacket.
    # The crime spot was near a prominent landmark, the city park. The pincode for the area is 567896. The incident was captured on CCTV cameras, and the police are actively investigating. The latitude and longitude of the crime spot are 40.7128 N and 74.0060 West.  Phone Number is 9922334455
    # """
    # text = "Rajesh Kumar attempted to murder Pravin"
    import string
    def remove_punctuation(text):
        # Define a translation table that excludes periods (full stops)
        translator = str.maketrans('', '', string.punctuation.replace('.', ''))
        
        # Apply the translation to the text
        text = text.translate(translator)
        
        return text
    # Example usage:
    text_with_punctuation = text
    text = remove_punctuation(text_with_punctuation)
    # print("text = ",text) 
    doc = nlp(text)
    
    
    pattern = r'\d+'
    names = []
    
    first_name = ""
    last_name = ""
    # for ent in doc.ents:
    #     if ent.label_ == "PERSON":
    #         match = re.search(pattern, ent.text)
    #         if not match:
    #             print("entities: ",ent.text)
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            match = re.search(pattern, ent.text)
            if not match:
                name_parts = ent.text.split()
                if len(name_parts) > 1:
                    first_name = name_parts[0]
                    last_name = " ".join(name_parts[1:])
                    names.append((first_name, last_name))
    print("Names extracted (First Name, Last Name):")
    for first_name, last_name in names:
        print("First Name: "+first_name)
        print("Last Name: "+last_name)
    #
    data['fname'] = first_name
    data['lname'] = last_name
    age = None
    for ent in doc.ents:
        if ent.label_ == "DATE" or ent.label_ == "CARDINAL":
            if "Age" in ent.text or "age" in ent.text or "year" in ent.text or "years" in ent.text or "old" in ent.text or "years-old" in ent.text or "year-old" in ent.text:
                pattern = r'\d+'
                numbers = re.findall(pattern, ent.text)
                    
                for number in numbers:
                    # print(number)
                    age = number
            else:
                pattern = r'\d+'
                numbers = re.findall(pattern, ent.text)
                for number in numbers:
                    age = number          

    print("age=",age)
    
    #
    data['age'] = age
    time_period = None
    for ent in doc.ents:
        if ent.label_ == "TIME":
            print("time_period = ",ent.text)
            time_period = ent.text.upper()
    #
    data['time_period'] = time_period
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            print(ent.text)
    addrs = ""
    for ent in doc.ents:
        if ent.label_ == "FAC":
            addrs = addrs+ent.text+", "
            
    # print("addrs = ",addrs)
    addrs = addrs+",   ,"
    i = len(addrs)-1
    addr = list(addrs)
    while True:
        print(addr[i])
        if addr[i]==" " or addr[i]==",":
            addr[i]=''
        else:
            break    
        i = i-1
    addrs=''.join(addr)
    print("address = ",addrs)
    #
    data['addrs'] = addrs
    height = None
    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            # print(ent.text)
            pattern = r'\d+'
            # Use re.findall to find all numbers in the text
            numbers = re.findall(pattern, ent.text)
            # Print the extracted numbers
            for number in numbers:
                print("Hieght found:", number)
            
            if "feet" in ent.text:
                conversion_factor = 30.48
                number = number * conversion_factor
                print("height converted to cm from feet")
            height = number
    #
    data['height'] = height
    mobile = None
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            if len(ent.text) == 10:
                mobile = int(ent.text)
    #             print(ent.text)
    # Mobile = None
    if mobile == None:
        for ent in doc.ents:
            if ent.label_ == "FAC":
    #             print(ent.text)
                if "Phone" in ent.text or "phone" in text:
                    pattern = r'\d+'
                    numbers = re.findall(pattern, ent.text)
                    
                    for number in numbers:
                        if len(str(number)) == 10:
                            mobile = int(number)
    print(type(mobile))
    print("Mobile Number: ",mobile)
    #
    data['mobile'] = mobile
    pincode = None
    for ent in doc.ents:
        if ent.label_ == "DATE":
            if len(ent.text) == 6:
                # print(ent.text)
                pincode = ent.text
    #
    data['pincode'] = pincode
    pattern = r'(\d+\.\d+\s+(?:N|S|North|South))|(\d+\.\d+\s+(?:W|E|West|East))'
    # Search for latitude and longitude in the text
    matches = re.findall(pattern, text)
    # Extracted latitude and longitude values
    latitude = None
    longitude = None
    for match in matches:
        if match[0]:  # Latitude matched
            latitude = match[0]
        elif match[1]:  # Longitude matched
            longitude = match[1]
    print("Latitude:", latitude)
    print("Longitude:", longitude)
    #
    data['Latitude'] = latitude
    data['Longitude'] = longitude
    gender = None
    txt = text.lower()
    if "male" in txt:
        gender = "male"
    elif "female" in txt:
        gender = "female"
    
    #
    data['gender'] = gender
    
    obj = CrimeOperations()
    xo = obj.genFirno() 
    stat['data'] = data
    stat.update(xo)
    return stat

def trainModelForHotspotsPrediction(request):
    #     !pip install pymongo
    # !pip install pandas
    # # !pip install joblib
    # import pymongo
    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.metrics import mean_squared_error
    # from sklearn.preprocessing import LabelEncoder
    # from sklearn.impute import SimpleImputer
    # import numpy as np
    # from sklearn.externals import joblib

    # Connect to MongoDB
    # client = pymongo.MongoClient("mongodb+srv://tathevedant70:VedantMadhav@cluster0.30s18ki.mongodb.net/?retryWrites=true&w=majority")
    client = pymongo.MongoClient("mongodb+srv://vedant:vedant@cluster0.3glbf3u.mongodb.net/?retryWrites=true&w=majority")
    #client = pymongo.MongoClient("mongodb+srv://pratham:swisshy@prathamclus.l5pmia2.mongodb.net/?retryWrites=true&w=majority")
    print("started training")
    # client = pymongo.MongoClient("mongodb+srv://tanmayshambharkar22:tanmay@cluster0.sptyi.mongodb.net/?retryWrites=true&w=majority")
            
    # Retrieve data from MongoDB
    db = client.Crime
    collection = db.CrimeDetails
    cursor = collection.find()

    # data = []
    # for doc in cursor:
    #     data.append(doc)

    data = list(cursor)

    # Convert MongoDB cursor to Pandas DataFrame
    columns_to_select = ['Type_of_Crime', 'Pincode', 'Landmark', 'Date_of_Crime', 'Time_Period_of_Crime', 'Latitude', 'Longitude']
    df = pd.DataFrame(data, columns=columns_to_select)

    # Handle missing values
    df = df.replace(['', 'Select', 'SELECT', None], np.NAN)
    imputer = SimpleImputer(strategy='mean')

    print(df[['Latitude', 'Longitude', 'Pincode']])
    
    df[['Latitude', 'Longitude', 'Pincode']] = imputer.fit_transform(df[['Latitude', 'Longitude', 'Pincode']])
    df[['Pincode']] = df[['Pincode']].astype(int)

    # Convert date and time columns to datetime objects
    # df['Timestamp'] = pd.to_datetime(df['Date_of_Crime'] + ' ' + df['Time_Period_of_Crime'])
    df['Date_of_Crime'] = pd.to_datetime(df['Date_of_Crime'])

    df['Day_of_Week'] = df['Date_of_Crime'].dt.dayofweek
    df['Year'] = df['Date_of_Crime'].dt.year
    df['Month'] = df['Date_of_Crime'].dt.month
    df['Week'] = df['Date_of_Crime'].dt.isocalendar().week
    df[['Crime_Start_Time', 'Crime_End_Time']] = df['Time_Period_of_Crime'].str.split('-', expand=True)
    df[['Crime_Start_Hour','x']] = df['Crime_Start_Time'].str.split(':', expand=True)
    df[['Crime_End_Hour','y']] = df['Crime_End_Time'].str.split(':', expand=True)
    df.drop(['Crime_Start_Time', 'Crime_End_Time','x','y','Time_Period_of_Crime','Date_of_Crime'], axis=1, inplace=True)

    # Identify NaN values along columns
    # nan_columns = df.columns[df.isna().any()].tolist()
    # inf_columns = [col for col in df.columns if np.any(np.isinf(pd.to_numeric(df[col], errors='coerce')))]
    # print("Columns with NaN:", nan_columns)
    # print("Columns with infinity:", inf_columns)

    # Convert categorical variables to numerical using Label Encoding
    # label_encoder = LabelEncoder()
    # mytoclist = ['Pocket Theft','Chain Theft','Bicycle Theft','Two-wheeler Theft','Four-wheeler Theft','Other Vehicle Theft','Vehicle Parts Theft','Other Theft','Commercial Robbery','Technical Robbery','Preparing to Robbery','Other Robbery','Daytime Burglary','Night Burglary','Culpable Homicide','Forcible Theft','Rape','Murder','Attempt to Murder','Betrayal','Riot','Injury','Molestation','Gambling','Prohibition','Other']
    # toc_df = pd.DataFrame(mytoclist, columns=['Type_of_Crime'])
    # x = label_encoder.fit_transform(toc_df['Type_of_Crime'])
    # print(x)


    label_encoder = LabelEncoder()
    df['Type_of_Crime'] = label_encoder.fit_transform(df['Type_of_Crime'])
    df['Landmark'] = label_encoder.fit_transform(df['Landmark'])

    # Handle missing values for Type_of_Crime and Landmark
    imputer = SimpleImputer(strategy='mean')
    df[['Type_of_Crime', 'Landmark']] = imputer.fit_transform(df[['Type_of_Crime', 'Landmark']])
    df[['Landmark','Type_of_Crime']] = df[['Landmark','Type_of_Crime']].astype(int)

    X = df[['Day_of_Week', 'Week', 'Month', 'Year', 'Type_of_Crime', 'Crime_Start_Hour', 'Crime_End_Hour']]
    y_longitude = df[['Longitude']]
    y_latitude = df[['Latitude']]
    y_pincode = df[['Pincode']]
    y_landmark = df[['Landmark']]

    X_train, X_test, y_longitude_train, y_longitude_test, y_latitude_train, y_latitude_test, y_landmark_train, y_landmark_test, y_pincode_train, y_pincode_test = train_test_split(
        X, y_longitude, y_latitude, y_landmark, y_pincode, test_size=0.2, random_state=42
    )

    y_longitude_train = y_longitude_train.values.ravel()
    y_latitude_train = y_latitude_train.values.ravel()
    y_pincode_train = y_pincode_train.values.ravel()
    y_landmark_train = y_landmark_train.values.ravel()


    # Choose a machine learning model (Random Forest Regressor)
    model_longitude = RandomForestRegressor()
    model_latitude = RandomForestRegressor()
    model_landmark = RandomForestRegressor()
    model_pincode = RandomForestRegressor()

    model_longitude.fit(X_train, y_longitude_train)
    model_latitude.fit(X_train, y_latitude_train)
    model_pincode.fit(X_train, y_pincode_train)
    model_landmark.fit(X_train, y_landmark_train)

    label_encoder_data = {
    'classes_': label_encoder.classes_.tolist(),
    # Add any other necessary information
    }

    # Extract relevant information from the DataFrame
    model_data = {
        'X_test': X_test.to_dict(orient='records'),  # Convert X_test DataFrame to a list of dictionaries
        'y_longitude_test': y_longitude_test.to_dict(orient='records'),
        'y_latitude_test': y_latitude_test.to_dict(orient='records'),
        'y_landmark_test': y_landmark_test.to_dict(orient='records'),
        'y_pincode_test': y_pincode_test.to_dict(orient='records'),
        'label_encoder': label_encoder_data,
    }

    # Set the variables in the session
    request.session['model_data'] = model_data

    # Save the trained models to files
    joblib.dump(model_longitude, 'C:/CrimeAnalysis_ML/model_longitude.pkl')
    joblib.dump(model_latitude, 'C:/CrimeAnalysis_ML/model_latitude.pkl')
    joblib.dump(model_pincode, 'C:/CrimeAnalysis_ML/model_pincode.pkl')
    joblib.dump(model_landmark, 'C:/CrimeAnalysis_ML/model_landmark.pkl')
    print("training completed")

def predictHotspots(request,dow,week,month,year,toc,csh,ceh):
    # Load the models from files when needed
    print("prediction started")
    loaded_model_longitude = joblib.load('C:/CrimeAnalysis_ML/model_longitude.pkl')
    loaded_model_latitude = joblib.load('C:/CrimeAnalysis_ML/model_latitude.pkl')
    loaded_model_pincode = joblib.load('C:/CrimeAnalysis_ML/model_pincode.pkl')
    loaded_model_landmark = joblib.load('C:/CrimeAnalysis_ML/model_landmark.pkl')

    model_data = request.session.get('model_data')

    if not model_data:
        print('Model data not found. Please train the models first.')
        trainModelForHotspotsPrediction(request)
        
        loaded_model_longitude = joblib.load('C:/CrimeAnalysis_ML/model_longitude.pkl')
        loaded_model_latitude = joblib.load('C:/CrimeAnalysis_ML/model_latitude.pkl')
        loaded_model_pincode = joblib.load('C:/CrimeAnalysis_ML/model_pincode.pkl')
        loaded_model_landmark = joblib.load('C:/CrimeAnalysis_ML/model_landmark.pkl')
        # return {'probabilities': None}

    # Convert the list of dictionaries back to DataFrames
    X_test = pd.DataFrame(model_data['X_test'])
    y_longitude_test = pd.DataFrame(model_data['y_longitude_test'])
    y_latitude_test = pd.DataFrame(model_data['y_latitude_test'])
    y_landmark_test = pd.DataFrame(model_data['y_landmark_test'])
    y_pincode_test = pd.DataFrame(model_data['y_pincode_test'])
 
    label_encoder_data = model_data['label_encoder']
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_data['classes_'])


    # Make predictions using the loaded models
    # predictions_longitude_loaded = loaded_model_longitude.predict(X_test)
    # predictions_latitude_loaded = loaded_model_latitude.predict(X_test)
    # predictions_pincode_loaded = loaded_model_pincode.predict(X_test)
    # predictions_landmark_loaded = loaded_model_landmark.predict(X_test)

    # Replace the following values with your own data
    custom_data = {
    'Day_of_Week': [dow],
    'Week': [week],
    'Month': [month],
    'Year': [year],
    'Type_of_Crime': [toc],
    'Crime_Start_Hour': [csh],
    'Crime_End_Hour': [ceh]
    }


    # Create a DataFrame with your custom data
    custom_df = pd.DataFrame(custom_data)

    # print(custom_df)

    # Ensure that the columns in custom_df match the columns used during training
    # Add any necessary preprocessing steps here to align with the training data

    # Make predictions using the loaded models
    predictions_longitude_loaded = loaded_model_longitude.predict(custom_df)
    predictions_latitude_loaded = loaded_model_latitude.predict(custom_df)
    predictions_pincode_loaded = loaded_model_pincode.predict(custom_df)
    predictions_landmark_loaded = loaded_model_landmark.predict(custom_df)

    #   Reshape predictions to match the shape of the test data
    # predictions_longitude_loaded = predictions_longitude_loaded.reshape(-1, 1)
    # predictions_latitude_loaded = predictions_latitude_loaded.reshape(-1, 1)
    # predictions_pincode_loaded = predictions_pincode_loaded.reshape(-1, 1)
    # predictions_landmark_loaded = predictions_landmark_loaded.reshape(-1, 1)

    # print(predictions_longitude_loaded)
    # print(predictions_latitude_loaded)
    # print(predictions_pincode_loaded)
    # print(predictions_landmark_loaded)


    predictions_longitude_decoded = predictions_longitude_loaded
    predictions_latitude_decoded = predictions_latitude_loaded
    predictions_pincode_decoded = predictions_pincode_loaded
    predictions_landmark_decoded = label_encoder.inverse_transform(predictions_landmark_loaded.astype(int))


    # Print or use the decoded predictions
    # print("Decoded Landmark Predictions:", predictions_landmark_decoded)
    # print("Decoded Longitude Predictions:", predictions_longitude_decoded)
    # print("Decoded Latitude Predictions:", predictions_latitude_decoded)
    # print("Decoded Pincode Predictions:", predictions_pincode_decoded)
    print("prediction completed")

    return([predictions_landmark_decoded,predictions_longitude_decoded,predictions_latitude_decoded,predictions_pincode_decoded,csh,ceh])


def getTomorrowData(mydate):
    # Get tomorrow's date
    tomorrow_date = mydate

    # Extract components
    day_of_week = tomorrow_date.weekday()  # 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
    week_number = tomorrow_date.isocalendar()[1]  # ISO week number
    month = tomorrow_date.month
    year = tomorrow_date.year

    # Print the results
    print(f"Day of the week: {day_of_week}")
    print(f"Week number: {week_number}")
    print(f"Month: {month}")
    print(f"Year: {year}")

    mylist = [day_of_week, week_number, month, year]
    return mylist









def index(request):
    # print(request.session.get('all_hotspots'))

    if 'user' in request.session:
        obj=CrimeOperations()
        stat=obj.count()
        dic = stat

        #check type of user
        name = request.session.get('user')
        v = UserOperations()
        utype = v.checkType(name)
        request.session['utype'] = utype
        print(request.session.get('user'))
        print(utype)

        
        # trainModelForHotspotsPrediction(request)
        # fname = None

        # for index, char in enumerate(name):
        #     if char.isupper():
        #         # Found the first capital letter, extract the substring
        #         fname = str(name[:index])

        # if request.session.get('new') == 1:
        #     fname = request.session.get('fname')
        #     text = "Welcome "+fname+" Sir"
        #     speak(text)
        #     sleep(0.5)
        #     speak("I am Sarthi")
        #     sleep(0.5)
        #     speak("Your Voice Assistant")
        #     sleep(0.5)
        #     print(request.session.get('new'))
        #     request.session['new'] = 0
        #     print(request.session.get('new'))

        
        return render(request, "index.html/",dic)
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

def welcomeMsg(request):
    if 'user' in request.session:
        
        if request.session.get('new') == 1:
            fname = request.session.get('fname')
            text = "Welcome "+fname+" Sir"
            speak(text)
            sleep(0.5)
            speak("I am Sarthi")
            sleep(0.5)
            speak("Your Voice Assistant")
            sleep(0.5)
            
            # training_thread = threading.Thread(target=trainModelForHotspotsPrediction, args=(request,))
            # training_thread.start()
            # print("next")

            print(request.session.get('new'))
            request.session['new'] = 0
            print(request.session.get('new'))
            
        
        return render(request, "index.html/")
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

def predicthotspot(request):
    if 'user' in request.session:
        trainModelForHotspotsPrediction(request)
        dic={}
        try:
            mydate = datetime.now() + timedelta(days=1)
            mylist = getTomorrowData(mydate)
            predictions_list = []
            all_hotspots = []
    
            mytoclist_mapped={ "Pocket Theft": 18, "Chain Theft": 3, "Bicycle Theft": 2, "Two-wheeler Theft": 24, "Four-wheeler Theft": 8, "Other Vehicle Theft": 17, "Vehicle Parts Theft": 25, "Other Theft": 16, "Commercial Robbery": 4, "Technical Robbery": 23, "Preparing to Robbery": 19, "Other Robbery": 15, "Daytime Burglary": 6, "Night Burglary": 13, "Culpable Homicide": 5, "Forcible Theft": 7, "Rape": 21, "Murder": 12, "Attempt to Murder": 0, "Betrayal": 1, "Riot": 22, "Injury": 10, "Molestation": 11, "Gambling": 9, "Prohibition": 20, "Other": 14 }
            timeperiod={0:2,2:4,4:6,6:8,8:10,10:12,12:14,14:16,16:18,18:20,20:22,22:24}
           
            for crime, value in mytoclist_mapped.items():
                print(f"************* CRIME = {crime} *******************")
                for s, e in timeperiod.items():
                    # print(f"------Time = {s}-{e} ----------")
                    predictionlist = predictHotspots(request, mylist[0], mylist[1], mylist[2], mylist[3], value, s, e)
                    predictions_list.append(predictionlist)
    
            # max_probability = 0  # Initialize before the inner loop
            # for crime, value in mytoclist_mapped.items():
            #     print(f"************* CRIME = {crime} *******************")
            #     for s, e in timeperiod.items():
            #         print(f"------Time = {s}-{e} ----------")
            #         predictions = predictHotspots(request, mylist[0], mylist[1], mylist[2], mylist[3], value, s, e)
            #         # Check if predictions are None
            #         if predictions['probabilities'] is not None:
            #             predictions_probabilities = predictions['probabilities']
            #             # Find the index with the maximum probability
            #             max_prob_index = np.argmax(predictions_probabilities)
            #             # Get the maximum probability for the current time period
            #             max_probability_for_time = predictions_probabilities[max_prob_index]
            #             # Check if the current time period has a higher probability
            #             if max_probability_for_time > max_probability:
            #                 max_probability = max_probability_for_time
            #                 most_probable_time = (s, e)
            #     print("Most Probable Time Period:", most_probable_time)
            #     print("Max Probability:", max_probability)

            # dic={'msg':'Your Task Completed Please Check the'}
            # print(predictions_list)

                def extract_landmark_info(landmark_data):
                    return (
                        landmark_data[0][0],
                        float(landmark_data[1][0]),
                        float(landmark_data[2][0]),
                        float(landmark_data[3][0])
                    )

                # Extracting landmark information for each entry in the list
                landmark_info_list = [extract_landmark_info(landmark) for landmark in predictions_list]

                # Counting occurrences of each unique combination
                landmark_counter = Counter(landmark_info_list)

                # Finding the most common combination
                most_common_landmark = landmark_counter.most_common(1)[0][0]

                print("Most Common Landmark:")
                print("Landmark:", most_common_landmark[0])
                print("Longitude:", most_common_landmark[1])
                print("Latitude:", most_common_landmark[2])
                print("Pincode:", most_common_landmark[3])
                print("Count:", landmark_counter[most_common_landmark])

                all_hotspotsdic = {}
                all_hotspotsdic['Landmark'] =  most_common_landmark[0]
                all_hotspotsdic['Longitude'] = most_common_landmark[1]
                all_hotspotsdic['Latitude'] = most_common_landmark[2]
                all_hotspotsdic['Pincode'] = most_common_landmark[3]
                all_hotspotsdic['Crime'] = crime
                
                all_hotspots.append(all_hotspotsdic)
                
            # request.session['all_hotspots'] = all_hotspots


            # all_hotspotsex = set(all_hotspots)
            # print(all_hotspots)
            
            
            def get_unique_key(d):
                return (d['Longitude'], d['Latitude'], d['Pincode'])

            # Use a set to keep track of unique keys
            unique_keys = set()

            # List to store unique dictionaries
            unique_dicts = []

            for d in all_hotspots:
                key = get_unique_key(d)
                if key not in unique_keys:
                    unique_keys.add(key)
                    unique_dicts.append(d)

            print(unique_dicts)
            request.session['all_hotspots'] = unique_dicts


            dic['msg'] = "Please double-check the results, as they may not be accurate every time. \nKeep in mind that predictions might vary."
            print(dic)
            speak("Your Task Completed. Please Check the Results")
                
        except Exception as ex:
            print("Exception: ",str(ex))
                        
        return render(request, "index.html/",dic)
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

def predicthotspotcall(request):
    training_thread = threading.Thread(target=trainModelForHotspotsPrediction, args=(request,))
    training_thread.start()

    training_thread.join()
    
    predict_tread = threading.Thread(target=predicthotspot, args=(request,))
    predict_tread.start()

    return render(request, "index.html/",{'msg':'Success, Training the model... It will take some time, till you can explore other features'})

def predicthotspots_of_date(request):
    dic = {'err':None,'msg':None}
    if request.method == "POST":
        # try:
        if True:
            mydate_str = request.POST.get('mydate')
            crimetype = request.POST.get('CrimeType').title()
            timeperiod = request.POST.get('Timeperiod')

            # Parse the date string into a datetime object
            mydate = datetime.strptime(mydate_str, '%Y-%m-%d')

            # Format the datetime object to display only the date portion
            mydate = mydate.date()

            # Print the formatted date
            print('Date:', mydate)

            # Get tomorrow's date components
            mylist = getTomorrowData(mydate)  
            print(crimetype)      
            print(mylist)

            # Mapping of crime type to integer
            mytoclist_mapped = {"Pocket Theft": 18, "Chain Theft": 3, "Bicycle Theft": 2, "Two-wheeler Theft": 24, "Four-wheeler Theft": 8, "Other Vehicle Theft": 17, "Vehicle Parts Theft": 25, "Other Theft": 16, "Commercial Robbery": 4, "Technical Robbery": 23, "Preparing to Robbery": 19, "Other Robbery": 15, "Daytime Burglary": 6, "Night Burglary": 13, "Culpable Homicide": 5, "Forcible Theft": 7, "Rape": 21, "Murder": 12, "Attempt to Murder": 0, "Betrayal": 1, "Riot": 22, "Injury": 10, "Molestation": 11, "Gambling": 9, "Prohibition": 20, "Other": 14 }

            # Get the integer representation of the crime type
            ctype = mytoclist_mapped.get(crimetype)  

            # Extract start and end times from the time period
            start_time, end_time = timeperiod.split(" - ")

            # Extract the hour values from the start and end times
            s = int(start_time.split(":")[0])
            e = int(end_time.split(":")[0])

            print(ctype)
            print(s,e)

            predictionlist = predictHotspots(request, mylist[0], mylist[1], mylist[2], mylist[3], ctype, s, e)
                    
            print("prediction result: \n")
            print(predictionlist)

            dic.update({
                'date':mydate,
                'crimetype': crimetype,
                'timeperiod':timeperiod,
                'landmark': predictionlist[0][0],  # Assuming only one location in the array
                'latitude': predictionlist[1][0],
                'longitude': predictionlist[2][0],
                'pincode': predictionlist[3][0],
                'start_time': predictionlist[4],
                'end_time': predictionlist[5],
                'msg':'Successfully Predicted the Future Hotspot'
            })
            
            return render(request, "futurehotspots.html",dic)

        # except Exception as e:
        #     print("Error parsing date:", e)
        #     dic['err']  = e
        #     return render(request, "futurehotspots.html",dic)
    else:
        return render(request, "futurehotspots.html")

            

def registerCrime(request):
    if 'user' in request.session:
        if request.method!="POST":
            print(request.session.get('user'))
            obj = CrimeOperations()
            stat = obj.genFirno()        
            print(stat)
            return render(request, "RegisterCrime.html/",stat)
        else:
            stat={'err':None,'msg':None}
            data={}

            # if request.POST.get('micforallbtn'):
                
                # data = readFromVoiceSeparate(request)

                # stat['data'] = data              
                
                # print("\n\ndata = ",data)
            # try:
            # if request.POST.get('micforallbtn'):
            #     print("exiting")
            # else:               
                # print("readDescriptionFromVoice: ")
                # x=readDescriptionFromVoice(request)
                # stat.update(x)

            print("readDescriptionFromVoice: ")
            x=readDescriptionFromVoice(request)
            stat.update(x)
            
            stat['msg'] = "Please double-check the results, as they may not be accurate every time. \nKeep in mind that predictions might vary."
            
            speak("Your Task Completed. Please Check the Results")
                
            # except Exception as ex:
            #     print("exception: "+str(ex))
            #     stat['err'] = "Opps! Something went wrong"
            return render(request, "RegisterCrime.html/",stat)
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

def addcrime(request):
    dic={}
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    if request.method=="POST":
        # try:
            fname = request.POST.get("fname").upper()
            lname = request.POST.get("lname").upper()

            age_str = request.POST.get("age")
            age = int(age_str) if age_str else None  # Use None for empty input

            dob = request.POST.get("dob")

            addhar_str = request.POST.get("addhar")
            addhar = int(addhar_str) if addhar_str else None

            gen = request.POST.get("gender").upper()
            ph = request.POST.get("phone")
            hight = request.POST.get("hight")
            address = request.POST.get("address").upper()

            firno_str = request.POST.get("firno")
            firno = int(firno_str) if firno_str else None

            CrimeType = request.POST.get("CrimeType").upper()

            nop_str = request.POST.get("nop")
            nop = int(nop_str) if nop_str else None

            CrimeWeapons = request.POST.get("CrimeWeapons").upper()
            cdate = request.POST.get("Cdate")
            Timeperiod = request.POST.get("Timeperiod")
            vehicle = request.POST.get("vehicle").upper()

            vehicleNo = request.POST.get("vno").upper()

            personsit = request.POST.get("vnp").upper()
            wearperson = request.POST.get("wearp").upper()
            discVehi = request.POST.get("discVehi").upper()
            crimesp = request.POST.get("crimesp").upper()

            pincd_str = request.POST.get("pincd")
            pincd = int(pincd_str) if pincd_str else None

            discCrimeSP = request.POST.get("discsp").upper()
            Status = request.POST.get("Status")
            Landmark = request.POST.get("Landmark").upper()
            Longitude = request.POST.get("Longitude")
            Latitude = request.POST.get("Latitude")
            Police_s = request.POST.get("Police_Station").upper()

        
            obj=CrimeOperations()
            
            stat=obj.addnewcrime(fname,lname,age,dob,addhar,gen,ph,hight,address,firno,CrimeType,nop,CrimeWeapons,cdate,Timeperiod,vehicle,vehicleNo,personsit,wearperson,discVehi,crimesp,pincd,discCrimeSP,Status,Landmark,Longitude,Latitude,Police_s)
            dic=stat
        
        # except Exception as err: 
        #     dic['status']='Oops! Something went wrong'
        #     print("Error"+str(err));        
    return render(request, "RegisterCrime.html/",dic)

def show(request):
    # data_list = []
    # context = {}
    # obj = CrimeOperations()  # Instantiate once
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})


    # try:
    #     stat1 = obj.crimedata_list()  # Assuming userdata_list returns a cursor
    #     if stat1['err'] is None:
    #         cursor = stat1['userdata']
    #         for document in cursor:
    #             data_dict = {
    #                 'Username': document['Username'],
    #                 'Email': document['Email'],
    #                 'Password': document['Password'],
    #                 'Status': document['Status'],
    #                 # 'country': document['country'],
    #                 # Add more key-value pairs as needed
    #             }
    #             data_list.append(data_dict)
    #     else:
    #         context['err'] = stat1['err']
    # except Exception as err:
    #     print("view catch: ", err)

    # context['data_list'] = data_list
    return render(request, "ShowCrime.html/")

def searchcrime(request):
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    # speaking_thread = threading.Thread(target=speak, args=("WELCOME TO SEARCH CRIME WEB PAGE HERE YOU CAN SEARCH CRIMES",))
    # speaking_thread.start()

    return render(request, "SearchCrime.html/")

def searchc(request):
    if request.method=="POST":
        key= int(request.POST.get("searchInput"))
        print("Key",key)
        obj=CrimeOperations()
        stat=obj.searchC(key)
        if stat['err'] is not None:
            return render(request, "SearchCrime.html/",stat)

    return render(request, "SearchResult.html/",stat)
 
def login(request):
    dic = {'msg':None,'err':None}
    if 'user' in request.session:
        del request.session['user']
        del request.session['new']
        # request.session.flush()

    if request.method == "POST":        
        uname = request.POST.get("username")
        pswd = request.POST.get("pass")

        print(uname, pswd)

        obj = UserOperations()
        stat = obj.login(uname, pswd)
        dic.update(stat)

        if dic['msg']:
            # Redirect to 'index.html' after successful login
            # dic['msg']="Login Successful"
            print(dic)
            request.session['user'] = uname
            request.session['fname'] = dic['fname']
            request.session['new'] = 1
            print("Session Set..!")
            return redirect('/index.html',dic)
            # return render(request,'index.html/',dic)
        elif dic['err']:
            # dic['err']="Invalid Credentials"
            return render(request, "login.html/",dic)
    return render(request, "login.html/",dic)

def register(request):
    dic={}
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    if request.method=="POST":
        try:
            fname = request.POST.get("fname")
            lname = request.POST.get("lname")
            uname = request.POST.get("uname")
            email = request.POST.get("email")
            pswd = request.POST.get("password")
            
            obj=UserOperations()
            
            stat=obj.register(fname,lname,uname,email,pswd)
            dic=stat
        
        except Exception as err: 
            dic['err']='Oops! Something went wrong'
            print("Error"+str(err))
    return render(request, "register.html/",dic)

def updateuser(request):
            
    stat={}
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    uname = request.GET['uname']
    stat['uname'] = uname
    obj = UserOperations()
    stat = obj.getUserData(uname)
    print(stat)
    
    if request.method == "POST":
        obj = UserOperations()
        print(request.POST)
        s = obj.updateUserData(request.POST.get('uname'),request.POST.get('email'),request.POST.get('password'))
        stat.update(s)
        if stat['msg']:
            query_params = f"msg={stat['msg']}"
            return redirect(f"/viewuser.html/?{query_params}")
        
    return render(request, "updateuser.html/", stat)

def viewusers(request):
    data_list = []
    
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    context = request.GET.dict()
    print(context)
    obj = UserOperations()  # Instantiate once


    if request.method == "POST":
        try:
            print(request.POST)
            for key in request.POST:
                if key.startswith("deactivate_"):
                    uname = key[len("deactivate_"):]
                    stat = obj.updateStatus(uname, "Activated")
                    print(uname, "hi")
                    # ustat = v.checkStatus(name)
                    # if ustat == "Deactivated":
                    #     return render(request, "login.html/",{'err':'Your Account is Deactivated...!'})
                elif key.startswith("activate_"):
                    uname = key[len("activate_"):]
                    stat = obj.updateStatus(uname, "Deactivated")
                    print(uname, "h")
                elif key.startswith("normal_"):
                    uname = key[len("normal_"):]
                    stat = obj.updateType(uname,"admin")
                    print(uname, "hii")
                elif key.startswith("admin_"):
                    uname = key[len("admin_"):]
                    stat = obj.updateType(uname,"normal")
                    print(uname, "hiii")
                    
                    # name = request.session.get('user')
                    # v = UserOperations()
                    # utype = v.checkType(name)
                    # print(request.session.get('user'))
                    # print(utype)
                    # context['utype'] = utype
                    # if utype == 'normal':
                    #     request.session['utype'] = utype 
                    #     return render(request, "index.html/",{'err':'You are not admin...!'})
                
            context.update(stat)
            print(context)

        except Exception as err:
            print("view catch post: ", err)
            context['err'] = "Oops! Something went wrong"

    try:
        stat1 = obj.userdata_list()  # Assuming userdata_list returns a cursor
        if stat1['err'] is None:
            cursor = stat1['userdata']
            for document in cursor:
                data_dict = {
                    'Username': document['Username'],
                    'Email': document['Email'],
                    'Password': document['Password'],
                    'Status': document['Status'],
                    'utype': document['utype'],
                    # 'country': document['country'],
                    # Add more key-value pairs as needed
                }
                data_list.append(data_dict)
        else:
            context['err'] = stat1['err']
    except Exception as err:
        print("view catch: ", err)

    context['data_list'] = data_list
    print(context)
    return render(request, "viewuser.html", context)

def chartjs(request):
    data = {}
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    if request.method == "POST":
        clicked_button = request.POST.get('special_btn', '')

        # print(clicked_button)
        # if(clicked_button):
        x = request.POST.get('x-axis')
        y = request.POST.get('y-axis')
        result = x + "/" + y
        data['result'] = result
        print("hi")
        # else:
        #     if(clicked_button == "HOTSPOT1"):
        #         data['result'] = "Count(Landmark)/Time_Period_of_Crime"

        # Identify which button is clicked
        
    
    return render(request, "chartjs.html", data)

def ViewCharts(request):
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    return render(request, "ViewCharts.html/")

# def viewcriminals(request):
#     context = {'criminals_data': []}
#     if 'user' in request.session:
#         print(request.session.get('user'))
#     else:
#         return render(request, "login.html/",{'err':'Please Login First...!'})

#     if request.method == "GET":
#         try:
#             msg = request.GET.get('msg', None)
            
#             obj = CrimeOperations()
#             response_data = obj.criminals_list()

#             # Check if the response_data has 'err' and 'criminals_list' keys
#             if isinstance(response_data, dict) and 'err' in response_data and 'criminals_list' in response_data:
#                 err = response_data['err']
#                 criminals_list = response_data['criminals_list']

#                 if err is None and isinstance(criminals_list, list):
#                     criminals_data = {}

#                     for criminal in criminals_list:
#                         first_name = criminal.get('First_Name', '')
#                         last_name = criminal.get('Last_Name', '')
#                         crime_type = criminal.get('Type_of_Crime', '')
#                         firno = criminal.get('FIR_No', '')

#                         # Create a unique key for each criminal based on First_Name and Last_Name
#                         key = f"{first_name} {last_name}"

#                         if first_name != None and last_name != None and first_name != '' and last_name != '':

#                             # If the key already exists, append the crime type, otherwise create a new entry
#                             if key in criminals_data:
#                                 if crime_type not in criminals_data[key]['crime_types']:
#                                     criminals_data[key]['crime_types'].append(crime_type)
#                             else:
#                                 criminals_data[key] = {
#                                     'first_name': first_name,
#                                     'last_name': last_name,
#                                     'crime_types': [crime_type],
#                                     'firno':firno,
#                                 }
                        
                        
#                     context['msg'] = msg
#                     context['criminals_data'] = list(criminals_data.values())
#                 else:
#                     # Handle the case where there is an error in the response
#                     context['error'] = 'Error in response data'

#         except Exception as err:
#             print("view catch: of criminal list ", err)
#             context['error'] = 'An error occurred while fetching criminals list'
#     # Render the template with the provided context
#     return render(request, "viewcriminal.html", context)

# def viewcriminals(request):
#     context = {'criminals_data': []}
#     if 'user' in request.session:
#         print(request.session.get('user'))
#     else:
#         return render(request, "login.html/",{'err':'Please Login First...!'})

#     if request.method == "GET":
#         try:
#             msg = request.GET.get('msg', None)
            
#             obj = CrimeOperations()
#             response_data = obj.criminals_list()

#             if isinstance(response_data, dict) and 'err' in response_data and 'criminals_list' in response_data:
#                 err = response_data['err']
#                 criminals_list = response_data['criminals_list']

#                 if err is None and isinstance(criminals_list, list):
#                     criminals_data = {}

#                     for criminal in criminals_list:
#                         first_name = criminal.get('First_Name', '')
#                         last_name = criminal.get('Last_Name', '')
#                         crime_type = criminal.get('Type_of_Crime', '')
#                         firno = criminal.get('FIR_No', '')

#                         key = f"{first_name} {last_name}"

#                         if first_name and last_name:
#                             if key in criminals_data:
#                                 if crime_type not in criminals_data[key]['crime_types']:
#                                     criminals_data[key]['crime_types'].append(crime_type)
#                             else:
#                                 criminals_data[key] = {
#                                     'first_name': first_name,
#                                     'last_name': last_name,
#                                     'crime_types': [crime_type],
#                                     'firno': firno,
#                                 }
                        
#                     context['msg'] = msg
#                     context['criminals_data'] = list(criminals_data.values())
#                 else:
#                     context['error'] = 'Error in response data'

#                 if isinstance(response_data, dict) and 'err' in response_data:
#                     err = response_data['err']

#                 if err is None:
#                     criminals_list = response_data.get('criminals_list', [])
#                     paginator = Paginator(criminals_list, 150)
#                     page_number = request.GET.get('page')
#                     page_obj = paginator.get_page(page_number)

#                     context['page_obj'] = page_obj
#                     context['msg'] = msg
#                 else:
#                     context['error'] = 'Error in response data'
#         except Exception as err:
#             print("An error occurred while fetching criminals list:", err)
#             context['error'] = 'An error occurred while fetching criminals list'

#     return render(request, "viewcriminal.html", context)

# def viewcriminals(request):
#     context = {'criminals_data': []}
#     if 'user' in request.session:
#         print(request.session.get('user'))
#     else:
#         return render(request, "login.html/",{'err':'Please Login First...!'})

#     if request.method == "GET":
#         try:
#             msg = request.GET.get('msg', None)
            
#             obj = CrimeOperations()
#             response_data = obj.criminals_list()

#             if isinstance(response_data, dict) and 'err' in response_data and 'criminals_list' in response_data:
#                 err = response_data['err']
#                 criminals_list = response_data['criminals_list']

#                 if err is None and isinstance(criminals_list, list):
#                     criminals_data = {}

#                     for criminal in criminals_list:
#                         first_name = criminal.get('First_Name', '')
#                         last_name = criminal.get('Last_Name', '')
#                         crime_type = criminal.get('Type_of_Crime', '')
#                         firno = criminal.get('FIR_No', '')

#                         key = f"{first_name} {last_name}"

#                         if first_name and last_name:
#                             if key in criminals_data:
#                                 if crime_type not in criminals_data[key]['crime_types']:
#                                     criminals_data[key]['crime_types'].append(crime_type)
#                             else:
#                                 criminals_data[key] = {
#                                     'first_name': first_name,
#                                     'last_name': last_name,
#                                     'crime_types': [crime_type],
#                                     'firno': firno,
#                                 }
                        
#                     context['msg'] = msg
#                     context['criminals_data'] = list(criminals_data.values())
#                 else:
#                     context['error'] = 'Error in response data'

#                 if isinstance(response_data, dict) and 'err' in response_data:
#                     err = response_data['err']

#                 if err is None:
#                     criminals_list = [criminal for criminal in criminals_data.values()]
#                     paginator = Paginator(criminals_list, 50)
#                     page_number = request.GET.get('page')
#                     page_obj = paginator.get_page(page_number)

#                     context['page_obj'] = page_obj
#                     context['msg'] = msg
#         except Exception as err:
#                 print("An error occurred while fetching criminals list:", err)
#                 context['error'] = 'An error occurred while fetching criminals list'

#         return render(request, "viewcriminal.html", context)

def viewcriminals(request):
    context = {'criminals_data': []}
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/", {'err': 'Please Login First...!'})

    if request.method == "GET":
        try:
            msg = request.GET.get('msg', None)

            obj = CrimeOperations()
            response_data = obj.criminals_list()

            if isinstance(response_data, dict) and 'err' in response_data and 'criminals_list' in response_data:
                err = response_data['err']
                criminals_list = response_data['criminals_list']

                if err is None and isinstance(criminals_list, list):
                    criminals_data = {}

                    unique_police_stations = []

                    for criminal in criminals_list:
                        first_name = criminal.get('First_Name', '')
                        last_name = criminal.get('Last_Name', '')
                        crime_type = criminal.get('Type_of_Crime', '')
                        firno = criminal.get('FIR_No', '')
                        ps = criminal.get('Police_Station', '')

                        key = f"{first_name} {last_name}"

                        if first_name and last_name:
                            if key in criminals_data:
                                if crime_type not in criminals_data[key]['crime_types']:
                                    criminals_data[key]['crime_types'].append(crime_type)
                            else:
                                criminals_data[key] = {
                                    'first_name': first_name,
                                    'last_name': last_name,
                                    'crime_types': [crime_type],
                                    'firno': firno,
                                    'police_station': ps,
                                }
                            if ps not in unique_police_stations:
                                unique_police_stations.append(ps)

                    print(unique_police_stations)
                    context['msg'] = msg
                    context['criminals_data'] = list(criminals_data.values())
                    context['police_stations'] = unique_police_stations
                else:
                    context['error'] = 'Error in response data'

                if isinstance(response_data, dict) and 'err' in response_data:
                    err = response_data['err']

                if err is None:
                    criminals_list = [criminal for criminal in criminals_data.values()]
                    paginator = Paginator(criminals_list, 150)
                    page_number = request.GET.get('page')
                    page_obj = paginator.get_page(page_number)

                    context['page_obj'] = page_obj
                    context['msg'] = msg
        except Exception as err:
            print("An error occurred while fetching criminals list:", err)
            context['error'] = 'An error occurred while fetching criminals list'

    return render(request, "viewcriminal.html", context)

def criminaldetail(request):
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    context = {}
    if request.method == "POST":
        for key in request.POST:
            if key.startswith("detail_"):
                n = key[len("detail_"):]
                s_values = n.split('-')
                f_name = s_values[0]
                l_name = s_values[1]
                firno = s_values[2]
                print("Firstname:", f_name)
                print("Lastname:", l_name)

        obj = CrimeOperations()
        criminal_data = obj.criminal_detail(f_name,l_name)

        if criminal_data['err'] == None:
            # for key, value in criminal_data['criminal_data'].items():
            #     print(f"{key}: {value}")
            context = criminal_data
        else:
            print(criminal_data)
        
        return render(request, "criminaldetail.html",context)

def updatecrime(request,parameter=''):
    context = {'criminals_data': []}
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/",{'err':'Please Login First...!'})

    if request.method == "GET":
        try:
            obj = CrimeOperations()
            response_data = obj.criminals_list()
            context['msg'] = parameter

            # Check if the response_data has 'err' and 'criminals_list' keys
            if isinstance(response_data, dict) and 'err' in response_data and 'criminals_list' in response_data:
                err = response_data['err']
                criminals_list = response_data['criminals_list']

                if err is None and isinstance(criminals_list, list):
                    criminals_data = {}

                    for criminal in criminals_list:
                        first_name = criminal.get('First_Name', '')
                        last_name = criminal.get('Last_Name', '')
                        crime_type = criminal.get('Type_of_Crime', '')
                        status = criminal.get('Status', '')

                        # Create a unique key for each criminal based on First_Name and Last_Name
                        key = f"{first_name} {last_name}"

                        if first_name != None and last_name != None and first_name != '' and last_name != '':

                            # If the key already exists, append the crime type, otherwise create a new entry
                            if key in criminals_data:
                                if crime_type not in criminals_data[key]['crime_types']:
                                    criminals_data[key]['crime_types'].append(crime_type)
                            else:
                                criminals_data[key] = {
                                    'first_name': first_name,
                                    'last_name': last_name,
                                    'crime_types': [crime_type],
                                }
                        
                        

                    context['criminals_data'] = list(criminals_data.values())
                else:
                    # Handle the case where there is an error in the response
                    context['error'] = 'Error in response data'

        except Exception as err:
            print("view catch: of criminal list ", err)
            context['error'] = 'An error occurred while fetching criminals list'
    # Render the template with the provided context
    return render(request, "UpdateCrime.html", context)

def showcrimedata(request):
    context = {}
    if request.method == "POST":
        try:
            fn = int(request.POST.get('firno'))
            # fn = param1
            print("---> FIR No : - ---->",fn)
            # For example, printing the parameter
            print(f"FIR_No from query parameters: {fn}")


            obj = CrimeOperations()
            data = obj.searchfirno(fn)

            context = data
            x = context['data']
            
            x['Type_of_Crime'] = (x['Type_of_Crime'].lower())


            context['data']  = x
            print("The context",context)
            
        except Exception as err:
                print("view catch: showcrimedata ", err)

    return render(request, "updatecriminaldata.html", context)
    
# def displaycrime(request,param1=0):
#     if request.method == "GET":
#         try:
#             print("In the get method")
            
#             if param1 == 0:
#                 param1 = request.GET.get('param1')

#             print(param1)
#             msg = request.GET.get('msg', None)
#             fir_no = param1
#             fn = fir_no
#             # For example, printing the parameter
#             print(f"FIR_No from query parameters: {fn}")


#             obj = CrimeOperations()
#             data = obj.searchfirno(fn)

#             if data['err']:
#                 return render(request, "displaycrime.html", {'err_message': data['err']})
            
#             data['msg'] = msg
#             print("data = ",data)
#         except Exception as err:
#                 print("view catch: displaycrime ", err)

#     return render(request, "SearchResult.html", data)


def displaycrime(request,param1=0):
    if request.method == "GET":
        try:
            print("In the get method")
            msg = request.GET.get('msg', None)
            
            if param1 == 0:
                param1 = request.GET.get('param1')

            print(param1)
            msg = request.GET.get('msg', None)
            fir_no = param1
            fn = fir_no
            # For example, printing the parameter
            print(f"FIR_No from query parameters: {fn}")


            obj = CrimeOperations()
            data = obj.searchfirno(fn)

            if data['err']:
                return render(request, "displaycrime.html", {'err_message': data['err']})
            
            data['msg'] = msg
            print("data = ",data)
        except Exception as err:
                print("view catch: displaycrime ", err)

    return render(request, "SearchResult.html", data)

def showupdate_combined(request):
        
        if request.method == "POST":
            try:
                print("in the post method")
                obj = CrimeOperations()
                print(request.POST)
                fname = request.POST.get('fname')
                lname = request.POST.get('lname')
                age = request.POST.get('age')
                dob = request.POST.get('dob')
                addhar = request.POST.get('addhar')
                gender = request.POST.get('gender')
                phone = request.POST.get('phone')
                hight = request.POST.get('hight')
                address = request.POST.get('address')
                firno = request.POST.get('firno')
                CrimeType = request.POST.get('CrimeType')
                nop = request.POST.get('nop')
                CrimeWeapons = request.POST.get('CrimeWeapons')
                cdate = request.POST.get('Cdate')
                Timeperiod = request.POST.get('Timeperiod')
                vehicle = request.POST.get('vehicle')
                vehicleNo = request.POST.get('vno')
                personsit = request.POST.get('vnp')
                wearperson = request.POST.get('wearp')
                discVehi = request.POST.get('discVehi')
                crimesp = request.POST.get('crimesp')
                pincd = request.POST.get('pincd')
                discCrimeSP = request.POST.get('discsp')
                Status = request.POST.get('Status')
                Landmark = request.POST.get('Landmark')
                Longitude = request.POST.get('Longitude')
                Latitude = request.POST.get('Latitude')
                PoliceStation = request.POST.get('Police_Station')
                data = obj.updateCriminalData(fname,lname,age,dob,addhar,gender,phone,hight,address,firno,CrimeType,nop,CrimeWeapons,cdate,Timeperiod,vehicle,vehicleNo,personsit,wearperson,discVehi,crimesp,pincd,discCrimeSP,Status,Landmark,Longitude,Latitude,PoliceStation)
                # print("data = ",data)

                return redirect('/displaycrime/{}/?msg={}'.format(firno,data['msg']))
                
       

            except Exception as err:
              print("view catch: show criminal list ", err)
        return render(request,"UpdateCrime.html/")

# def showupdate(request, param1, param2, param3):
#     context = {'data': []}

#     if 'user' in request.session:
#         print(request.session.get('user'))
#     else:
#         return render(request, "login.html/", {'err': 'Please Login First...!'})

#     if request.method == "GET":
#         try:
#             fn = param1
#             ln = param2
#             ct = param3

#             obj = CrimeOperations()
#             data = obj.crimesearch(fn, ln, ct)

#             if data.get('err'):
#                 return render(request, "showupdatecrime.html", {'err_message': data['err']})

#             for document in data.get('data', []):
#                 context['data'].append({
#                    'first_name': document.get("First_Name", ''),
#                             'last_name': document.get("Last_Name", ''),
#                             'age': document.get('Age', ''),
#                             'date_of_birth': document.get('Date-of-Birth', ''),
#                             'aadhar_no': document.get('Aadhar_No', ''),
#                             'gender': document.get('Gender', ''),
#                             'phone_no': document.get('Phone_No', ''),
#                             'height': document.get('Height', ''),
#                             'address': document.get('Address', ''),
#                             'fir_no': document.get('FIR_No', ''),
#                             'crime_types': document.get("Type_of_Crime", ''),
#                             'number_of_person_involved': document.get('Number_Of_Person_Involve', ''),
#                             'weapons_used': document.get('Weapons_use_for_Crime', ''),
#                             'date_of_crime': document.get('Date_of_Crime', ''),
#                             'time_period_of_crime': document.get('Time_Period_of_Crime', ''),
#                             'vehicle_use': document.get('Vehicle_use', ''),
#                             'vehicle_no': document.get('Vehicle_No', ''),
#                             'no_of_person_on_vehicle': document.get('No_of_Person_on_Vehicle', ''),
#                             'description_of_person_sitting_on_vehicle': document.get('Discription_of_Person_sitting_on_Vehicle', ''),
#                             'description_of_vehicle': document.get('Discription_of_Vehicle', ''),
#                             'crime_spot': document.get('Crime_Spot', ''),
#                             'pincode': document.get('Pincode', ''),
#                             'description_of_crime_spot': document.get('Discription_of_Crime_Spot', ''),
#                             'status': document.get('Status', ''),
#                             'landmark': document.get('Landmark', ''),
#                 })
                
#         except Exception as err:
#             print("view:showupdate ", err)

#     return render(request, "showupdatecrime.html", context)

# def showupdate(request, param1, param2, param3):
#     context = {'data': []}

#     if 'user' in request.session:
#         print(request.session.get('user'))
#     else:
#         return render(request, "login.html/", {'err': 'Please Login First...!'})

#     if request.method == "GET":
#         try:
#             fn = param1
#             ln = param2
#             ct = param3

#             obj = CrimeOperations()
#             data = obj.crimesearch(fn, ln, ct)
#             print("Length of data:", len(data.get('data', [])))

#             print("data : ",data)

#             if data.get('err'):
#                 return render(request, "showupdatecrime.html", {'err_message': data['err']})

#             for document in data.get('data', []):
#                 context['data'].append({
#                    'first_name': document.get("First_Name", ''),
#                             'last_name': document.get("Last_Name", ''),
#                             'fir_no': document.get('FIR_No', ''),
#                             'crime_types': document.get("Type_of_Crime", ''),
#                             'number_of_person_involved': document.get('Number_Of_Person_Involve', ''),
#                             'weapons_used': document.get('Weapons_use_for_Crime', ''),
#                             'date_of_crime': document.get('Date_of_Crime', ''),
#                             'time_period_of_crime': document.get('Time_Period_of_Crime', ''),
#                             'vehicle_use': document.get('Vehicle_use', ''),
#                             'vehicle_no': document.get('Vehicle_No', ''),
#                             'no_of_person_on_vehicle': document.get('No_of_Person_on_Vehicle', ''),
#                             'description_of_person_sitting_on_vehicle': document.get('Discription_of_Person_sitting_on_Vehicle', ''),
#                             'description_of_vehicle': document.get('Discription_of_Vehicle', ''),
#                             'crime_spot': document.get('Crime_Spot', ''),
#                             'pincode': document.get('Pincode', ''),
#                             'description_of_crime_spot': document.get('Discription_of_Crime_Spot', ''),
#                             'status': document.get('Status', ''),
#                             'landmark': document.get('Landmark', ''),
#                 })

#                 print("document : ",context['data'])
                
#         except Exception as err:
#             print("view:showupdate ", err)

#     return render(request, "showupdatecrime.html", context)

def showupdate(request, param1, param2, param3):
    context = {'data': []}

    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/", {'err': 'Please Login First...!'})

    if request.method == "GET":
        try:
            fn = param1
            ln = param2
            ct = param3

            obj = CrimeOperations()
            data = obj.crimesearch(fn, ln, ct)

            print("data : ", data)

            if data.get('err'):
                return render(request, "showupdatecrime.html", {'err_message': data['err']})

            for document in data.get('data', []):
                context['data'].append({
                    'first_name': document.get("First_Name", ''),
                    'last_name': document.get("Last_Name", ''),
                    'fir_no': document.get('FIR_No', ''),
                    'crime_types': document.get("Type_of_Crime", ''),
                    'number_of_person_involved': document.get('Number_Of_Person_Involve', ''),
                    'weapons_used': document.get('Weapons_use_for_Crime', ''),
                    'date_of_crime': document.get('Date_of_Crime', ''),
                    'time_period_of_crime': document.get('Time_Period_of_Crime', ''),
                    'vehicle_use': document.get('Vehicle_use', ''),
                    'vehicle_no': document.get('Vehicle_No', ''),
                    'no_of_person_on_vehicle': document.get('No_of_Person_on_Vehicle', ''),
                    'description_of_person_sitting_on_vehicle': document.get('Discription_of_Person_sitting_on_Vehicle', ''),
                    'description_of_vehicle': document.get('Discription_of_Vehicle', ''),
                    'crime_spot': document.get('Crime_Spot', ''),
                    'pincode': document.get('Pincode', ''),
                    'description_of_crime_spot': document.get('Discription_of_Crime_Spot', ''),
                    'status': document.get('Status', ''),
                    'landmark': document.get('Landmark', '')
                })

                print("document : ", context['data'])

            # Check if there is only one record in the context data
            if len(context['data']) == 1:
                 firno = context['data'][0]['fir_no']
                # Call another function within the view if there is only one record
                 return displaycrime(request, firno)  # Update with your desired function name
                
        except Exception as err:
            print("view:showupdate ", err)

    return render(request, "showupdatecrime.html", context)
   
def deletecrimedata(request):
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/", {'err': 'Please Login First...!'})

    context = {}
    if request.method == "POST":
        try:
            firno = int(request.POST.get('firno'))
            # firno = param1
            print("---> FIR No : - ---->",firno)
            # For example, printing the parameter
            print(f"FIR_No from query parameters: {firno}")


            obj = CrimeOperations()

            result = obj.searchfirno(firno)

            result_data = result.get('data', {})
            if result_data:
                # Access specific values from the nested dictionary
                fn = result_data.get('First_Name', '')
                ln = result_data.get('Last_Name', '')
                ct = result_data.get('Type_of_Crime', '')

                # Now you can print or use these variables as needed in your view
                print("First Name:", fn)
                print("Last Name:", ln)
                print("Type of Crime:", ct)
            else:
                # Handle the case where no data is found
                print("No data found for FIR No:", firno)

            #print("hmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm",result)
            # print("First Name",data['First_Name'])

            data = obj.delete_criminal_data(firno)

            msg = data.get('msg','')
            
            # print("message : ",msg)
            context['msg'] = msg

            # return redirect(f'/viewcriminal.html/?msg={msg}')
            return redirect(f'/SearchCrime.html/?msg={msg}')
            
            
        except Exception as err:
                print("view catch: showcrimedata ", err)

    return render(request, "updatecriminaldata.html", context)

def searchchart(request):
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html/", {'err': 'Please Login First...!'})

    return render(request,"searchchart.html/")
 
def searchchartresult(request):
    if request.method == "GET":
        # search_term = request.GET.get('search_term', '')
        search_term = request.GET.get('search_term', '').strip().lower()

        # charts_names = ['Robberies Registered in Past 3 years with respect to Landmark',' Crime Types with respect to its Status ','Number Of person involve with respect to Type of Crime ','Number of person involved with respect to vehicles','Crime Registered in locations (landmark) with respect to time period','Number of person involved with respect to age', 'Number of person involved with respect to gender', 'Pychart of Weapons Used for Crime','Number of person involved in crimes with respect to each year','Crimes Registered (Pocket theft, chain theft, bicycle theft) in past three year/ 3 years','Chain Theft Registered with respect to time period','Number of person on vehicle','Description of chain snatched','count of Chain theft in crime places / Landmark in last three years / 3 years','Robberies registered in police stations in last three years / 3 years','Crime registered in police stations in last three years / 3 years','Chain theft registered in police stations in last three years / 3 years']

        if search_term == '' or search_term == '':
            results = ['No Matching Results ....!']
        else:
            charts_data = {
               #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                'Crimes Registered (Pocket theft, chain theft, bicycle theft) in past three year':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4c5f-84c2-590df2b8c463&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Pychart of Chain Theft Registered with respect to time period':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-443d-827f-590df2b8c465&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Pychart of Number of Persons on Vehicle In Chain Snatching':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4e26-87cc-590df2b8c467&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Pychart of Description of chain snatched':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4a08-8dec-590df2b8c469&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Pychart of Chain theft in crime places / Landmark in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4b34-88b2-590df2b8c46b&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Robberies registered in police stations in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-404c-8c55-590df2b8c46d&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Crime registered in police stations in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4a99-88f5-590df2b8c46f&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Chain theft registered in police stations in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4d5d-856c-590df2b8c471&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',              
                'Robberies Registered in Past 3 years with respect to Landmark': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-47fd-84bf-590df2b8c473&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Number of person involved with respect to vehicles':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4b22-854e-590df2b8c47b&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Pychart of Weapons Used for Crime':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-446f-81d7-590df2b8c483&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Crime Types with respect to its Status': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4af1-8038-590df2b8c475&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Crime Registered in locations (landmark) with respect to time period':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-49a2-8fbf-590df2b8c47d&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Number of person involved in crimes with respect to each year':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-401d-8beb-590df2b8c485&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Crime Types with respect to its Status': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4b1b-89de-590df2b8c477&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Number of person involved with respect to gender':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-420e-89e1-590df2b8c47f&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Number Of person involve with respect to Type of Crime': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4cde-8300-590df2b8c479&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
                'Number of person involved with respect to age':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-zhgnb/embed/charts?id=65ab8c60-bbf1-4330-840c-590df2b8c481&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
            }

            # Search for the term in the keys of the charts_data dictionary
            results = {key: value for key, value in charts_data.items() if search_term in key.lower()}

        # Return the results as a JSON response
        return JsonResponse({'results': results})
    else:
        return render(request, "searchchart.html/")

from bson import ObjectId

def searchcrimeresult(request):
    if request.method == "GET":
        # search_term = request.GET.get('search_term', '')
        search_term = request.GET.get('search_term', '').strip().lower()
        results = []
        # charts_names = ['Robberies Registered in Past 3 years with respect to Landmark',' Crime Types with respect to its Status ','Number Of person involve with respect to Type of Crime ','Number of person involved with respect to vehicles','Crime Registered in locations (landmark) with respect to time period','Number of person involved with respect to age', 'Number of person involved with respect to gender', 'Pychart of Weapons Used for Crime','Number of person involved in crimes with respect to each year','Crimes Registered (Pocket theft, chain theft, bicycle theft) in past three year/ 3 years','Chain Theft Registered with respect to time period','Number of person on vehicle','Description of chain snatched','count of Chain theft in crime places / Landmark in last three years / 3 years','Robberies registered in police stations in last three years / 3 years','Crime registered in police stations in last three years / 3 years','Chain theft registered in police stations in last three years / 3 years']

        if search_term == '' or search_term == ' ':
            results = ['No Matching Results ....!']
        else:
            print(search_term)
            obj = CrimeOperations()
            stat = obj.searchC2(search_term)
            
            # print(stat['result'])
            # results = stat['result']
            # for x in stat['result']:
            #     # print(type(x))
            #     mylist = list(x.values())
            #     results.append(mylist)
            # for x in stat['result']:
            #     mylist = [str(value) if isinstance(value, ObjectId) else value for value in x.values()]
            #     results.append(mylist)

            for x in stat['result']:
                mylist = [x.get('FIR_No'),x.get('First_Name'), x.get('Last_Name'),x.get('Type_of_Crime'),x.get('Police_Station')]  # First Name and Last Name
                # mylist.extend([str(value) if isinstance(value, ObjectId) else value for key, value in x.items() if key not in ['First_Name', 'Last_Name']])
                results.append(mylist)

            print(results)
        return JsonResponse({'results': results})
    else:
        return render(request, "SearchCrime.html/")


def statuschangedata(request):
    data = {}
    if 'user' in request.session:
        print(request.session.get('user'))
    else:
        return render(request, "login.html", {'err': 'Please Login First...!'})

    # Check if the request method is POST
    if request.method == "POST":
        try:
            print("In the view of statrus")
            status_changed = request.POST.get('status_changed')
            print(status_changed)
            firno = int(request.POST.get('firno'))
            obj = CrimeOperations()
            data, msg = obj.statuschanged(firno, status_changed)
        except Exception as err:
            print("view catch: of criminal list ", err)
            data = {'error': 'An error occurred while fetching criminals list'}
        
    
        #msg_parameter = f"?msg={data if isinstance(data, str) else ''}"

        return redirect(f'displaycrime/{firno}/?msg={msg}')

    
    # Handle GET requests if needed
    return render(request, "SearchResult.html")  # Adjust this line as needed


def futurehotspots(request):
    return render(request,"futurehotspots.html/")



def sampleurl(request):
    # crime_data = pd.read_csv('datafile (1).csv')
    # X = crime_data[['Latitude', 'Longitude']].values
    # y = crime_data['Pincode'].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # Build the neural network model
    # model = Sequential([
    #     Dense(64, activation='relu', input_shape=(2,)),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='linear')  # Assuming pincode prediction is a regression task
    # ])

    # # Compile the model
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # # Train the model
    # model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # # Evaluate the model
    # loss, mae = model.evaluate(X_test_scaled, y_test)

    # print(f'Test Mean Absolute Error: {mae}')

    # # Make predictions
    # predictions = model.predict(X_train)
    # print(predictions)


    return render(request,"sampleurl.html/")






































   
# def searchchartresult(request):
#     if request.method == "GET":
#         # search_term = request.GET.get('search_term', '')
#         search_term = request.GET.get('search_term', '').strip().lower()

#         # charts_names = ['Robberies Registered in Past 3 years with respect to Landmark',' Crime Types with respect to its Status ','Number Of person involve with respect to Type of Crime ','Number of person involved with respect to vehicles','Crime Registered in locations (landmark) with respect to time period','Number of person involved with respect to age', 'Number of person involved with respect to gender', 'Pychart of Weapons Used for Crime','Number of person involved in crimes with respect to each year','Crimes Registered (Pocket theft, chain theft, bicycle theft) in past three year/ 3 years','Chain Theft Registered with respect to time period','Number of person on vehicle','Description of chain snatched','count of Chain theft in crime places / Landmark in last three years / 3 years','Robberies registered in police stations in last three years / 3 years','Crime registered in police stations in last three years / 3 years','Chain theft registered in police stations in last three years / 3 years']

#         if search_term == '' or search_term == '':
#             results = ['No Matching Results ....!']
#         else:
#             charts_data = {
#                 'Robberies Registered in Past 3 years with respect to Landmark': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2); resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653a7dca-cc82-4e83-8ad4-9369f0202c1f&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Crime Types with respect to its Status': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2); resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653a8295-cc82-445d-85ed-9369f02549f1&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 # 'Crime Types with respect to its Status': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2); resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653a7fac-e89e-4197-8653-87676cc1db48&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Number Of person involve with respect to Type of Crime': '<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2); resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653a85e7-ef00-4529-8c56-d1af183e468b&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Number of person involved with respect to vehicles':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653bd8da-eb82-4a7d-8465-ee7d62b7abcd&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Crime Registered in locations (landmark) with respect to time period':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653bda20-49fd-4d0c-8acd-d0503a641279&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Number of person involved with respect to gender':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653bda90-7139-43fd-86b6-37a95908d3cb&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Number of person involved with respect to age':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653bda90-7139-43fd-86b6-37a95908d3cb&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Pychart of Weapons Used for Crime':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653bdd06-a2e1-4f99-84df-2960108420b1&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Number of person involved in crimes with respect to each year':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653bdde8-4509-41bc-8721-e86924611cd9&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Crimes Registered (Pocket theft, chain theft, bicycle theft) in past three year':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=652caa0e-2c36-444d-8ad4-8b2792bcc181&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Pychart of Chain Theft Registered with respect to time period':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=652cab14-9abd-4551-8092-9ea64e216133&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Pychart of Number of Persons on Vehicle In Chain Snatching':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=65363818-765d-4feb-8967-a8b693e688af&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Pychart of Description of chain snatched':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=653638c2-765d-43c2-881c-a8b693e6e216&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Pychart of Chain theft in crime places / Landmark in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=65363949-829f-4b71-8a8b-b0fa6ed2258f&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Robberies registered in police stations in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=65a932ba-de1f-4636-82a7-7f60f87ada5b&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Crime registered in police stations in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=65a93389-a36b-4398-80a0-d623a21315e7&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',
#                 'Chain theft registered in police stations in last three years':'<iframe style="background: #FFFFFF;border: none;border-radius: 2px;box-shadow: 0 2px 10px 0 rgba(70, 76, 79, .2);" resize:both;overflow:auto; padding: 20px" width="100%" height="100%" src="https://charts.mongodb.com/charts-project-0-wsfea/embed/charts?id=65a93415-dea8-44f4-8642-2bd0257ced8d&maxDataAge=3600&theme=light&autoRefresh=true"></iframe>',              
                
                
#             }

#             # Search for the term in the keys of the charts_data dictionary
#             results = {key: value for key, value in charts_data.items() if search_term in key.lower()}

#         # Return the results as a JSON response
#         return JsonResponse({'results': results})
#     else:
#         return render(request, "searchchart.html/")


from django.template.loader import render_to_string

def get_police_station(request):
    context = {'criminals_data': []}
    print("In the get police station")
    
    if 'user' not in request.session:
        return render(request, "login.html/", {'err': 'Please Login First...!'})
    
    if request.method == 'GET':
        try:
            selected_station = request.GET.get('station')
            # selected_station = station
            print("Selected Police Station : ",selected_station)
            
            if selected_station:
                obj = CrimeOperations()
                response_data = obj.criminals_list_as_per_police_station(selected_station)
                station = obj.getpolicestation()


                print("in the if")
                criminals_data = {}
                unique_police_stations = []
                
                if isinstance(response_data, dict) and 'err' in response_data and 'criminals_list' in response_data:
                    err = response_data['err']
                    criminals_list = response_data['criminals_list']
                    policestation = station['criminals_list']

                    

                    if err is None and isinstance(policestation, list):
                        
                        for criminal in policestation:
                            ps = criminal.get('Police_Station', '')
                            if ps not in unique_police_stations:
                                unique_police_stations.append(ps)

                        

                    
                    if err is None and isinstance(criminals_list, list):
                        
                        for criminal in criminals_list:
                            first_name = criminal.get('First_Name', '')
                            last_name = criminal.get('Last_Name', '')
                            crime_type = criminal.get('Type_of_Crime', '')

                            key = f"{first_name} {last_name}"
                            if first_name and last_name:
                                if key in criminals_data:
                                    if crime_type not in criminals_data[key]['crime_types']:
                                        criminals_data[key]['crime_types'].append(crime_type)
                                else:
                                    criminals_data[key] = {
                                        'first_name': first_name,
                                        'last_name': last_name,
                                        'crime_types': [crime_type],
                                        'selected_station' : unique_police_stations,
                                    }

                        context['msg'] = request.GET.get('msg')
                        context['criminals_data'] = list(criminals_data.values())
                        context['police_stations'] = unique_police_stations

                        # Pagination
                        paginator = Paginator(context['criminals_data'], 150)
                        page_number = request.GET.get('page')
                        page_obj = paginator.get_page(page_number)    
                        context['page_obj'] = page_obj

                        return render(request, 'viewcriminal.html', context)
                    else:
                        context['error'] = 'An error occurred while fetching criminals list'
                else:
                    context['error'] = 'An error occurred while fetching criminals list'
            else:
                context['error'] = 'No station selected'
        except Exception as e:
            print("An error occurred:", e)
            context['error'] = 'An error occurred while processing the request'
    else:
        context['error'] = 'Invalid request method'
    
    return render(request, 'viewcriminal.html', context)


