# Core pkgs
import streamlit as st
import altair as alt

# EDA pkgs
import pandas as pd
import numpy as np

# Utils
import joblib
import emoji

# Functions
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_02_jan_2021.pkl","rb"))

# Function to read the emotion
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"\N{angry face}","disgust":"\N{unamused face}",
                       "fear":"Afraid","happy":"\U0001F602",
                       "joy":"\U0001F60D","neutral":"\U0001F910",
                       "sadness":"\N{loudly crying face}","shame":"Ashamed",
                       "surprise":"Wow!"}

def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
            
            if submit_text:
                col1,col2 = st.columns(2)
            
            # Apply functions here
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
            
                with col1:
                    st.success("Original Text")
                    st.write(raw_text)
                    
                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction,emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))
                
                with col2:
                    st.success("Prediction Probability")
                    #st.write(probability)
                    proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                    #st.write(proba_df.T)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions","probability"]
                    
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                    st.altair_chart(fig, use_container_width=True)
                
    elif choice == "Monitor":
        st.subheader("Monitor App")
        
    else:
        st.subheader("About")
        
if __name__ == '__main__':
    main()