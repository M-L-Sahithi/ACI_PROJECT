#======================== IMPORT PACKAGES ===========================
import streamlit as st
import base64
import seaborn as sns


#========================  BACKGROUND IMAGE ===========================


st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:50px;font-family:Caveat, sans-serif;">{"Fuzzy logic in sentiment analysis"}</h1>', unsafe_allow_html=True)
st.write("-------------------------------------------")



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.png')






# ----------------------------------------------

import pickle

with open('model.pickle', 'rb') as f:
    hyb = pickle.load(f)
 
with open('senti.pickle', 'rb') as f:
    sentiment = pickle.load(f)  
    

 
with open('vector.pickle', 'rb') as f:
    vector = pickle.load(f)  




user_input = st.text_input("Enter Message  !!!")


aa = st.button("PREDICT")



if aa:
    
    count_data = vector.transform([user_input])
    pred_rf = hyb.predict(count_data)
    
    if pred_rf==0:
        
        st.markdown(f'<h1 style="color:#000000 ;font-size:24px;text-align:center;font-family:convat;;">{"Predicted Emotion = IRRELAVANT"}</h1>', unsafe_allow_html=True)

    elif pred_rf==1:
        
        st.markdown(f'<h1 style="color:#000000 ;font-size:24px;text-align:center;font-family:convat;;">{"Predicted Emotion = NEGATIVE"}</h1>', unsafe_allow_html=True)
    

    elif pred_rf==2:
        
        st.markdown(f'<h1 style="color:#000000 ;font-size:24px;text-align:center;font-family:convat;;">{"Predicted Emotion = NEUTRAL"}</h1>', unsafe_allow_html=True)
        

    elif pred_rf==3:
        
        st.markdown(f'<h1 style="color:#000000 ;font-size:24px;text-align:center;font-family:convat;;">{"Predicted Emotion = POSITIVE"}</h1>', unsafe_allow_html=True)
            
    

        
    # vulnerabilities = list(sentiment)

    # # Sort the array in alphabetical order
    # vulnerabilities.sort()
    
    # # Function to retrieve the vulnerability based on user input
    # def retrieve_vulnerability(user_input):
    #     if 0 <= user_input < len(vulnerabilities):
    #         return vulnerabilities[user_input]
    #     else:
    #         return "Invalid input. Please enter a number between 0 and {}.".format(len(vulnerabilities) - 1)
    
    # # Example usage
    # user_input = int(pred_rf)
    
    # st.write(user_input)
    
    # result = retrieve_vulnerability(user_input)

    # st.write("----------------------------------------------------------------------")
    
    # aa = "Predicted Emotion " + " is " + str(result).upper()
    
    # st.markdown(f'<h1 style="color:#000000 ;font-size:24px;text-align:center;font-family:convat;;">{aa}</h1>', unsafe_allow_html=True)
    



