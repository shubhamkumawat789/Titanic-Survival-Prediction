# main.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the background image function
from background_image import set_background

# Load the trained model
@st.cache_resource
def load_model():
    with open('models/model_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Set the background image (make sure you have a 'background.jpg' or 'background.png' file)
    try:
        # Try different image formats
        for img_file in ['background.jpg', 'background.png', 'titanic_bg.jpg', 'titanic_bg.png', 'bg_image.jpg']:
            try:
                background_style = set_background(img_file)
                st.markdown(background_style, unsafe_allow_html=True)
                break
            except:
                continue
        else:
            # If no image found, use a solid color
            st.markdown("""
            <style>
            .stApp {{
                background-color: #f0f2f6;
            }}
            </style>
            """, unsafe_allow_html=True)
    except:
        pass
    
    st.title("🚢 Titanic Survival Prediction")
    st.write("Predict whether a passenger survived the Titanic disaster")
    st.markdown("---")
    
    # Load the model
    try:
        model = load_model()
        st.success("✅ Model loaded successfully!")
    except:
        st.error("❌ Model file not found! Please ensure 'models/model_rf.pkl' exists.")
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Passenger Information")
        
        # Passenger Class
        pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                            help="1 = First Class, 2 = Second Class, 3 = Third Class")
        
        # Sex
        sex = st.selectbox("Sex", ["male", "female"])
        
        # Age
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        
        # Number of Siblings/Spouses Aboard
        sibsp = st.number_input("Number of Siblings/Spouses Aboard", 
                              min_value=0, max_value=10, value=0,
                              help="Siblings = brother, sister, stepbrother, stepsister\nSpouse = husband, wife")
        
        # Number of Parents/Children Aboard
        parch = st.number_input("Number of Parents/Children Aboard", 
                              min_value=0, max_value=10, value=0,
                              help="Parent = mother, father\nChild = son, daughter, stepchild")
    
    with col2:
        st.subheader("✈️ Travel Information")
        
        # Fare
        fare = st.number_input("Fare (in £)", min_value=0.0, max_value=600.0, value=32.0, step=0.5,
                             help="Ticket fare amount")
        
        # Class (categorical version of pclass)
        class_mapping = {1: "First", 2: "Second", 3: "Third"}
        passenger_class = class_mapping[pclass]
        
        # Who (man, woman, child)
        if age < 16:
            who = "child"
        elif sex == "male":
            who = "man"
        else:
            who = "woman"
        
        # Adult Male
        adult_male = (sex == "male" and age >= 18)
        
        # Alone
        alone = (sibsp == 0 and parch == 0)
        
        # Display calculated fields in a nicer way
        st.markdown("**Automatically Calculated Fields:**")
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.info(f"**Class:** {passenger_class}")
            st.info(f"**Category:** {who}")
        with info_col2:
            st.info(f"**Adult Male:** {'Yes' if adult_male else 'No'}")
            st.info(f"**Alone:** {'Yes' if alone else 'No'}")
    
    # Create the input dataframe with EXACTLY the same features as training
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'class': [passenger_class],
        'who': [who],
        'adult_male': [adult_male],
        'alone': [alone]
    })
    
    # Display the input data in an expander
    with st.expander("📊 View Input Data Summary"):
        st.dataframe(input_data)
    
    # Make prediction when button is clicked
    if st.button("🚀 Predict Survival", type="primary", use_container_width=True):
        try:
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Display results
            st.markdown("---")
            st.subheader("📈 Prediction Results")
            
            # Create result columns
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction[0] == 1:
                    st.success("## ✅ Survived")
                    st.balloons()
                else:
                    st.error("## ❌ Did Not Survive")
            
            with result_col2:
                survived_prob = prediction_proba[0][1] * 100
                st.metric("Survival Probability", f"{survived_prob:.2f}%",
                         delta=f"{survived_prob-50:.1f}%" if survived_prob > 50 else f"{survived_prob-50:.1f}%")
            
            # Probability bar with custom styling
            st.markdown("**Confidence Level:**")
            st.progress(int(survived_prob))
            
            # Detailed probabilities in tabs
            tab1, tab2 = st.tabs(["📊 Probability Distribution", "📋 Interpretation"])
            
            with tab1:
                # Detailed probabilities
                prob_df = pd.DataFrame({
                    'Outcome': ['Did Not Survive', 'Survived'],
                    'Probability': [prediction_proba[0][0] * 100, prediction_proba[0][1] * 100]
                })
                
                # Create a bar chart
                st.bar_chart(prob_df.set_index('Outcome'), use_container_width=True)
                
                # Show probability table
                st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}))
            
            with tab2:
                # Interpretation
                if prediction[0] == 1:
                    st.success(f"""
                    ### 🎯 Survival Prediction
                    
                    Based on the provided information, the model predicts with **{survived_prob:.1f}% confidence** that this passenger would have survived the Titanic disaster.
                    
                    **Factors that may have contributed to survival:**
                    - Higher class passengers had better access to lifeboats
                    - Women and children were prioritized during evacuation
                    - Younger passengers had better survival rates
                    """)
                else:
                    st.error(f"""
                    ### 🎯 Non-Survival Prediction
                    
                    Based on the provided information, the model predicts with **{(100-survived_prob):.1f}% confidence** that this passenger would not have survived the Titanic disaster.
                    
                    **Possible reasons:**
                    - Lower class passengers had limited access to lifeboats
                    - Adult males had lower priority during evacuation
                    - Location of cabin affected evacuation time
                    """)
            
            # Feature importance (if available)
            if hasattr(model, 'best_estimator_'):
                with st.expander("🔍 View Feature Importance"):
                    try:
                        # Get feature importances
                        importances = model.best_estimator_['classifier'].feature_importances_
                        
                        # Feature names (matching the preprocessing pipeline)
                        feature_names = ['pclass', 'age', 'sibsp', 'parch', 'fare', 
                                       'sex_male', 'class_Second', 'class_Third', 
                                       'who_man', 'who_woman']
                        
                        # Create feature importance dataframe
                        feat_imp_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        # Display top 5 features
                        st.write("**Top 5 Most Important Features:**")
                        st.dataframe(feat_imp_df.head(5).style.bar(subset=['Importance'], color='lightblue'))
                        
                        # Bar chart of feature importances
                        st.bar_chart(feat_imp_df.set_index('Feature').head(10))
                        
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("💡 Please check that all input values are valid.")
    
    # Add sidebar with information
    with st.sidebar:
        st.title("ℹ️ About")
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", 
                caption="RMS Titanic", use_column_width=True)
        
        st.markdown("""
        **This app predicts Titanic passenger survival** using a Random Forest model trained on the classic Titanic dataset.
        
        ### 🎯 **Features Used:**
        - Passenger Class (pclass)
        - Sex
        - Age
        - Siblings/Spouses
        - Parents/Children
        - Fare
        - Class (categorical)
        - Who (man/woman/child)
        - Adult Male
        - Alone
        
        ### ⚠️ **Disclaimer:**
        This is a machine learning model for educational purposes. Actual historical survival depended on many complex factors.
        
        **Note:** The model uses the exact same preprocessing pipeline as during training.
        """)
        
        st.markdown("---")
        st.write("**⚙️ Model Details:**")
        st.write("- Algorithm: Random Forest Classifier")
        st.write("- Accuracy: ~82%")
        st.write("- Training Data: 891 passengers")
        st.write("- Features: 10 engineered features")
        
        st.markdown("---")
        
        # Add a download button for sample data
        st.write("**📥 Sample Data:**")
        if st.button("Download Sample CSV"):
            sample_data = pd.DataFrame({
                'pclass': [3, 1, 2],
                'sex': ['male', 'female', 'female'],
                'age': [22.0, 38.0, 26.0],
                'sibsp': [1, 1, 0],
                'parch': [0, 0, 0],
                'fare': [7.25, 71.28, 7.92]
            })
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download Sample Data",
                data=csv,
                file_name="titanic_sample_data.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        st.caption("Built with using Streamlit & Scikit-learn")
        

if __name__ == "__main__":
    main()