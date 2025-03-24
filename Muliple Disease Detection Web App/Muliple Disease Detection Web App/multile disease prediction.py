# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:28:07 2024

@author: sufyan
"""
import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu




# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #A31D1D;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        .stApp {
            background-color: #ECDCBF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#loading the saved model

breast_cancer_model= pickle.load(open('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/breast_cancer_model.sav','rb'))

diabetes_model= pickle.load(open('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/dibaetic_model.sav','rb'))

heart_disease_model= pickle.load(open('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/heart_disease_model.sav','rb'))

parkinsons_model=  pickle.load(open('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/parkinsons_model.sav','rb'))

def model_prediction(test_image):
    model = tf.keras.models.load_model("C:/Users/LENOVO/Downloads/new_ds/new_ds/")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#sidebar for navigate

with st.sidebar:
    
    selected = option_menu('HUMAN DISEASE DETECTION SYSTEM',
                           
                         ['Home',
                          "Colon's Disease Detection",
                          'Diabetes Detection',
                          'Breast Cancer Detection',
                          'Heart Disease Detection',
                          'Parkinsons Disease Detection',
                          'About'],
                         
                         icons=['house-door-fill','record-circle','activity','basket2-fill','heart','person','file-earmark-person fill'],
                         # from https://icons.getbootstrap.com/
                          default_index=0)
    
    # deafault 0 means when you open App 'Diabetes Prediction' is selected by default
    # if default was 1 then 'Heart Disease Prediction' will be selected by default
    # default 3 for 'Parkinsons Disease Prediction'

#Home Page
if(selected=='Home'):
    
    #page title
    st.title('HUMAN MULTIPLE DISEASE DETECTION SYSTEM')
    st.image('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/healthcare.jpg')
    st.markdown("""
    # Welcome to the Human Multiple Disease Detection System! 🔍
                
    Your health matters, and early detection can make all the difference. Many diseases progress silently, and timely detection can lead to more effective treatment. Our system is designed to assist in identifying multiple human diseases efficiently using **state-of-the-art machine learning techniques**. With just a few simple steps, you can analyze medical data and gain valuable insights into potential health risks.

    This AI-powered system utilizes deep learning and advanced medical data analysis techniques to help users detect diseases early, providing a **fast, reliable, and user-friendly experience**.

    ---
    
    ## How It Works ⚙️
    1. **Provide Health Information:** Navigate to the **Any Disease Recognition** page and input the relevant test data for the suspected disease.
    2. **Advanced AI-Powered Analysis:** Our machine learning model processes the data using deep learning techniques to detect patterns and identify potential diseases.
    3. **Instant Results & Recommendations:** View a detailed breakdown of the results, along with potential next steps for medical consultation and treatment.

    ---
    
    ## Why Choose Us? 🌟
    - **🔬 High Accuracy:** Our system leverages advanced machine learning models trained on large medical datasets to ensure precise disease detection.
    - **🖥️ User-Friendly Interface:** Designed for ease of use, making it accessible for healthcare professionals and general users alike.
    - **⚡ Fast and Efficient:** Receive results in seconds, enabling prompt decision-making and proactive health management.
    - **🩺 Multi-Disease Detection:** Detect a range of diseases with a single system, improving diagnostic efficiency and reliability.
    - **🔒 Secure & Private:** We prioritize user data privacy and security, ensuring that all uploaded medical data is processed securely and not stored.

    ---
    
    ## Get Started 🚀
    Click on the **Any Disease Recognition** page in the sidebar to upload your test results or input relevant data. Experience the power of AI-driven disease detection and take control of your health today!
    
    Supported diseases:
    - ✅ Diabetes Detection
    - ✅ Breast Cancer Detection
    - ✅ Heart Disease Detection
    - ✅ Parkinson’s Disease Detection
    - ✅ Colon’s Disease Detection
    - ✅ More diseases coming soon!
    
    ---
    
    ## About Us 📖
    Want to know more about how this system works? Visit the **About** page to explore details about our machine learning models, medical datasets, and research methodologies powering this innovative disease detection platform.
    
    **Your health is our priority. Let’s work together to ensure a healthier future! 💙🌍**
    """)

#About Page
if(selected=="About"):
    st.title("About")
    
    st.markdown("""
                ## 🏥 About This Web Application
                
                This Web Application is created by **Sufyan Khan** to assist in disease prediction using machine learning models trained on various medical datasets. 
                The application is designed to provide **accurate** 🧠 and **efficient** ⏳ predictions for different diseases based on patient data, enhancing early diagnosis and decision-making.

                ### 📊 About the Dataset
                The datasets used in this project have been **preprocessed and standardized** 📌 to ensure high accuracy and reliability. 
                The original datasets have undergone transformations such as **normalization**, **feature scaling**, and **handling missing values** to improve model performance.
                The dataset is divided into an **80/20 ratio** ⚖️ for training and testing while maintaining the original directory structure.

                ### ⚡ Features of This Application:
                - 🖥️ **User-Friendly Interface**: Easy navigation and seamless user experience.
                - 🏥 **Multiple Disease Prediction**: Supports classification for **Diabetes, Breast Cancer, Parkinson’s Disease, and Heart Disease, Colon's Disease**.
                - 🔍 **Data Standardization & Preprocessing**: Ensures improved model accuracy by handling missing values and normalizing input features.
                - 🔒 **Secure & Efficient Processing**: Data is processed securely, and results are provided instantly.
                - 🚀 **Real-Time Predictions**: Users can upload patient data and receive **immediate** predictions based on the trained model.

                ### 📂 Datasets Used:
                The following datasets have been used in training the models:

                1. **Diabetes Dataset** - [📥 Download](https://www.dropbox.com/scl/fi/0uiujtei423te1q4kvrny/diabetes.csv?rlkey=20xvytca6xbio4vsowi2hdj8e&e=1&dl=0)
                2. **Breast Cancer Dataset** - [📥 Download](https://drive.google.com/file/d/1wDjDuqDPAJd1cQEICcu19J9vrjFAWJ1H/view)
                3. **Parkinson's Disease Dataset** - [📥 Download](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)
                4. **Heart Disease Dataset** - [📥 Download](https://drive.google.com/file/d/1CEql-OEexf9p02M5vCC1RDLXibHYE9Xz/view)
                5. **Colon's Disease Dataset** - [📥 Download](https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning)
   
                ### 🔮 Future Enhancements:
                - 🔗 Adding more **disease datasets** to expand prediction capabilities.
                - 🧠 Implementing a **Deep Learning model** to improve accuracy.
                - 📝 Enabling **real-time patient data input** via forms for better usability.

                This application aims to **leverage machine learning for improving healthcare predictions** and assisting medical professionals in decision-making. ❤️‍🩹
                """)



#Prediction Page
if(selected=="Colon's Disease Detection"):

    st.title("Colon's Disease Detection")
    st.image('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/colon.jpg')
    
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        

#Breast Cancer Prediction Page
if(selected=='Breast Cancer Detection'):
    
    #page title
    st.title('BREAST CANCER DETECTION')
    st.image('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/Breastcancer1.jpg')
    
    #getting the input data from the user (All independent variables)
    
    
    # columns for input field 3 columns in 1 row
    col1,col2,col3=st.columns(3)
     
    with col1:
        meanradius=st.text_input('Mean radius')
    with col2:
        meantexture=st.text_input('Mean texture')
    with col3:
        meanperimeter=st.text_input('Mean perimeter')
    with col1:
        meanarea=st.text_input(' Mean area')
    with col2:
        meansmoothness=st.text_input(' Mean smoothness')
    with col3:
        meancompactness=st.text_input('Mean compactness')
    with col1:
        meanconcavity=st.text_input('Mean concavity')
    with col2:
        meanconcavepoints=st.text_input('Mean concave points')
    with col3:
        meansymmetry=st.text_input('Mean symmetry')
    with col1:
        meanfractaldimension=st.text_input('Mean fractal dimension')
    with col2:
        radiuserror=st.text_input('Radius error')
    with col3:
        textureerror=st.text_input('Texture error')
    with col1:
        perimetererror=st.text_input('Perimeter error')
    with col2:
        areaerror=st.text_input('Area error')
    with col3:
        smoothnesserror=st.text_input('Smoothness error')
    with col1:
        compactnesserror=st.text_input('Compactness error')
    with col2:
        concavityerror=st.text_input('Concavity error')
    with col3:
        concavepointserror=st.text_input('Concave points error')
    with col1:
        symmetryerror=st.text_input('Symmetry error')
    with col2:
        fractaldimensionerror=st.text_input('Fractal dimension error')
    with col3:
        worstradius=st.text_input('Worst radius')
    with col1:
        worsttexture=st.text_input('Worst texture')
    with col2:
        worstperimeter=st.text_input('Worst perimeter')
    with col3:
        worstarea=st.text_input('Worst area')
    with col1:
        worstsmoothness=st.text_input('Worst smoothness')
    with col2:
        worstcompactness=st.text_input('Worst compactness')
    with col3:
        worstconcavity=st.text_input('Worst concavity')
    with col1:
        worstconcavepoints=st.text_input(' Worst concave points')
    with col2:
        worstsymmetry=st.text_input('Worst symmetry')
    with col3:
        worstfractaldimension=st.text_input('Worst fractal dimension')
    
   
    
    # Code for Prediction
    
    breast_cancer_diagnosis=''
    #this will store the result
    
    #creating a button for Prediction
    
    if st.button('Breast Cancer Test Result'):
        breast_cancer_prediction=breast_cancer_model.predict([[meanradius,meantexture,meanperimeter,meanarea,meansmoothness,meancompactness,meanconcavity,meanconcavepoints,meansymmetry,meanfractaldimension,radiuserror,textureerror,perimetererror,areaerror,smoothnesserror,compactnesserror,concavityerror,concavepointserror,symmetryerror,fractaldimensionerror,worstradius,worsttexture,worstperimeter,worstarea,worstsmoothness,worstcompactness,worstconcavity,worstconcavepoints,worstsymmetry,worstfractaldimension]])
        # instead of reshaping the array we use 2d array('[[]]') to store the above input in his app which is different fom 'Diabetes Web App'
    
        if(breast_cancer_prediction[0]==0):
            breast_cancer_diagnosis='The Breast Cancer is Malignant'
            st.balloons()
            st.snow()
        
        else:
            breast_cancer_diagnosis='The Breast cancer is Benign'
            
    st.success(breast_cancer_diagnosis)
    # to print the diagnosis
    

#Diabetes Prediction Page
if(selected=='Diabetes Detection'):
    
    #page title
    st.title('DIABETES DETECTION')
    st.image('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/diabetes_unsplash.jpg')
    
    #getting the input data from the user (All independent variables)
    
    Age = st.slider('Age',min_value=0,max_value=100,value=30,step=1)
    
    # columns for input field 3 columns in 1 row
    col1,col2,col3=st.columns(3)
     
    with col1:
        Pregnancies=st.text_input('No. of Pregnancies')
        
    with col2:
        Glucose=st.text_input('Glucose level')
    
    with col3:
        BloodPressure=st.text_input('BloodPressure level')
    
    with col1:
        SkinThickness=st.text_input('Skin Thickness Value')
    
    with col2:
        Insulin=st.text_input('Insulin Level')
    
    with col3:
        BMI=st.text_input('BMI Value')
    
    with col2:
        DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value')
    
   
    
    # Code for Prediction
    
    diab_diagnosis=''
    #this will store the result
    
    #creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction=diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        # instead of reshaping the array we use 2d array('[[]]') to store the above input in his app which is different fom 'Diabetes Web App'
    
        if(diab_prediction[0]==0):
            diab_diagnosis='The Person is not Diabetic'
            st.balloons()
            st.snow()
        
        else:
            diab_diagnosis='The Person is Diabetic'
            
    st.success(diab_diagnosis)
    # to print the diagnosis
    
    
#Heart Disease Prediction Page

    # to print the diagnosis
if selected == 'Heart Disease Detection':

    # page title
    st.title('HEART DISEASE DETECTION')
    st.image('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/heart-disease-stock-photo-021423.jpg')
    
    age = st.slider('Age',min_value=0,max_value=100,value=30,step=2)

    col1, col2, col3 = st.columns(3)

    #with col1:
    

    with col1:
        sex = st.text_input('Sex')

    with col2:
        cp = st.text_input('Chest Pain types')

    with col3:
        trestbps = st.text_input('Resting Blood Pressure')

    with col1:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col3:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col1:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col2:
        exang = st.text_input('Exercise Induced Angina')

    with col3:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col1:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col2:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col3:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 0:
            heart_diagnosis = 'The person does not have any heart disease'
            st.balloons()
            st.snow()
        else:
            heart_diagnosis = 'The person is having heart disease'

    st.success(heart_diagnosis)    
    
#Parkinsons Disease Prediction Page
if(selected=='Parkinsons Disease Detection'):
    
    #page title
    st.title('PARKINSON DISEASE')
    st.image('C:/Users/LENOVO/Downloads/Muliple Disease Detection Web App/Muliple Disease Detection Web App/th.jpg',width=700)
    #getting the input data from the user (All independent variables)
    
    # columns for input field 3 columns in 1 row
    col1,col2,col3=st.columns(3)
    
    with col1:
        Fo=st.text_input('MDVP:Fo(Hz)')
    with col2:
        Fhi=st.text_input('MDVP:Fhi(Hz)')
    with col3:
        Flo=st.text_input('MDVP:Flo(Hz)')
    with col1:
        Jitter_percent=st.text_input('MDVP:Jitter(%)')
    with col2:
        Jitter_Abs=st.text_input('MDVP:Jitter(Abs)')
    with col3:
        MDVP_RAP=st.text_input('MDVP:RAP')
    with col1:
        MDVP_PPQ=st.text_input('MDVP:PPQ')
    with col2:
        Jitter_DDP=st.text_input('Jitter:DDP')
    with col3:
        MDVP_Shimmer=st.text_input('MDVP:Shimmer')
    with col1:
        MDVP_Shimmer_dB=st.text_input('MDVP:Shimmer(dB)')
    with col2:
        Shimmer_APQ3=st.text_input('Shimmer:APQ3')
    with col3:
        Shimmer_APQ5=st.text_input('Shimmer:APQ5')
    with col1:
        MDVP_APQ=st.text_input('MDVP:APQ')
    with col2:
        Shimmer_DDA=st.text_input('Shimmer:DDA')
    with col3:
        NHR=st.text_input('NHR')
    with col1:
        HNR=st.text_input('HNR')
    with col2:
        RPDE=st.text_input('RPDE')
    with col3:
        DFA=st.text_input('DFA')
    with col1:
        spread1=st.text_input('spread1')
    with col2:
        spread2=st.text_input('spread2')
    with col3:
        D2=st.text_input('D2')
    with col2:
        PPE=st.text_input('PPE')
    
    
    # Code for Prediction
    
    park_diagnosis=''
    #this will store the result
    
    #creating a button for Prediction
    
    if st.button('Parkinsons Test Result'):
        park_prediction=parkinsons_model.predict([[Fo,Fhi,Flo,Jitter_percent,Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        # instead of reshaping the array we use 2d array('[[]]') to store the above input in his app which is different fom 'Diabetes Web App'
    
        if(park_prediction[0]==0):
            park_diagnosis='The Person does not has Parkinson'
            st.balloons()
            st.snow()
        else:
            park_diagnosis='The Person has Parkinson'
            
    st.success(park_diagnosis)
    # to print the diagnosis
    
    
    

#"C:\Program Files\Python310\python.exe" -m streamlit run "C:\Users\LENOVO\Downloads\Muliple Disease Detection Web App\Muliple Disease Detection Web App\multile disease prediction.py"