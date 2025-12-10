# 🚢 Titanic Survival Prediction 
The Titanic Survival Prediction project aims to predict whether a passenger survived or not, using machine learning techniques. 

## 📌 Project Overview
The project uses the famous Titanic dataset and implements various algorithms such as Logistic Regression, Random Forest, and Voting Classifier Ensemble to train a predictive model. 
- Random Forest Classifier: For capturing complex non-linear relationships
- Logistic Regression: For interpretable linear decision boundaries
- Voting Classifier Ensemble: Combines both models for improved accuracy and robustness

# 📊 Dataset Description
The Titanic dataset contains information about 891 passengers on board the RMS Titanic with the following features:

| Column        | Type          | Description                                                                 |
|---------------|---------------|-----------------------------------------------------------------------------|
| survived      | int (binary)  | Survival status (0 = Did not survive, 1 = Survived)                        |
| pclass        | int (1-3)     | Passenger class (1 = First, 2 = Second, 3 = Third)                         |
| sex           | string        | Gender of passenger (male/female)                                           |
| age           | float         | Age in years (contains missing values)                                      |
| sibsp         | int           | Number of siblings/spouses aboard                                           |
| parch         | int           | Number of parents/children aboard                                           |
| fare          | float         | Passenger fare (in £)                                                       |
| embarked      | string        | Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)              |
| class         | string        | Passenger class (First/Second/Third) - categorical version of pclass        |
| who           | string        | Age/gender category (man/woman/child) - derived from age & sex             |
| adult_male    | bool          | Whether passenger is an adult male (True/False)                              |
| deck          | string        | Cabin deck (A-G) - contains many missing values                             |
| embark_town   | string        | Town of embarkation (Cherbourg/Queenstown/Southampton)                      |
| alive         | string        | Survival status (yes/no) - alternative to 'survived'                        |
| alone         | bool          | Whether passenger was traveling alone (True/False)                          |



# Process Breakdown
## Step 1: Data Loading

The Titanic dataset is loaded using Seaborn's built-in sns.load_dataset('titanic'). The dataset contains multiple features like:

- pclass (Passenger class)
- sex (Gender)
- age (Age of passenger)
- sibsp (Number of siblings/spouses aboard)
- parch (Number of parents/children aboard)
- fare (Ticket fare)
- embarked (Port of embarkation)
- Other derived features such as class, who, adult_male, and alone.

## Step 2: Data Preprocessing

Handling Missing Values: The SimpleImputer is used to handle missing values.
- Numerical features (age, fare, etc.) are imputed with the median value.
- Categorical features (sex, embarked, etc.) are imputed with the most frequent value.

One-Hot Encoding: Categorical variables like sex, pclass, and embarked are one-hot encoded using OneHotEncoder, which turns each category into binary columns. This is necessary because models can’t work directly with categorical text values.

Scaling Numerical Features: Numerical features (age, sibsp, fare, etc.) are scaled using StandardScaler to standardize their range.

## Step 3: ⚙️ Model Training

We train three models for prediction:

- Random Forest Classifier: An ensemble method that works well for classification tasks.
- Logistic Regression: A simple linear model.
- Voting Classifier: A combination of Random Forest and Logistic Regression that takes the majority vote from both models.

The models are trained using the training data (X_train, y_train), and the hyperparameters for the Random Forest model are tuned using GridSearchCV.

## Step 4: Model Evaluation

After training the models, we evaluate them using:

- Classification Report: Shows metrics like precision, recall, and F1-score.
- Confusion Matrix: A matrix that shows the true vs predicted values, helping to evaluate the classification performance visually.
- Feature Importance: We extract and visualize the most important features that contribute to predicting survival.

## Step 5: Model Saving

The trained models are saved using Pickle:

_ Random Forest Model: model_rf.pkl
- Logistic Regression Model: model_lr.pkl
- Voting Classifier Model: model_voting.pkl

This allows you to load and use the models in future predictions without retraining.

## Step 6: Deployment

The trained models are deployed using Streamlit, allowing users to input values for features like sex, age, sibsp, fare, and get predictions on whether the passenger survived or not.

To deploy:

- We first train the models and save them.
- We then create a Streamlit app (main.py) where users can input passenger data and get real-time survival predictions.

## Step 7: Visualization

- The confusion matrix is plotted using Seaborn to visualize the model’s performance.
- Feature Importance for both Random Forest and Logistic Regression models is visualized as bar charts.

## Step 8: Saving and Loading Models for Prediction

- After training, the models are saved in the models/ directory as .pkl files.
- During prediction, these models are loaded into the Streamlit app to provide real-time predictions based on user input.
- Testing the App

Once deployed, the app allows users to input the following features:

- Sex (male/female)
- Pclass (1st, 2nd, 3rd class)
- Age (numeric value)
- Siblings/Spouses aboard (numeric value)
- Parents/Children aboard (numeric value)
- Fare (numeric value)
- Embarked (C, Q, S)

Upon entering the values, it will predict whether the passenger survived based on the model.

## Saving the Models
Each trained model is saved into the models directory as a Pickle file (.pkl). This allows easy model retrieval for prediction during deployment.

- Logistic Regression Model: model_lr.pkl
- Random Forest Model: model_rf.pkl
- Voting Classifier Model: model_voting.pkl

# 📂 Project Structure
```
Titanic-Survival-Prediction/
│
├── app/                      # Main application directory
│ ├── main.py                 # Streamlit application entry point
│ └── background_image.py     # Background image handling utility
│
├── data/                     # Dataset directory
│ ├── titanic_dataset.csv     # Complete Titanic dataset
│ └── titanic_sample_data.csv # Sample data for testing
│
├── models/                   # Trained ML models
│ ├── model_rf.pkl            # Random Forest model
│ ├── model_lr.pkl            # Logistic Regression model
│ └── model_voting.pkl        # Voting Classifier model
│
├── notebooks/                # Jupyter notebooks for EDA & modeling
│ └── Titanic Survival Prediction.ipynb
│
├── Dockerfile                # Docker configuration for containerization
├── Procfile                  # Deployment configuration for Heroku/Railway
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file

```







