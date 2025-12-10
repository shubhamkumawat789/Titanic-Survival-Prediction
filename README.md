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


# 🧠 Feature Engineering

To improve model performance, several new features were created:

- HouseAge – current year minus year built
- YearsSinceRemod – years since last remodeling
- TotalLivingArea – total usable basement space
- BsmtFinRatio – percentage of basement that is finished
- IsRemodeled – 1 if renovated, otherwise 0
- LotAreaCategory – buckets: Small / Medium / Large / XL

These engineered features help the model capture patterns that raw features alone cannot.

# ⚙️ Model Training

The project uses a RandomForestRegressor inside a full preprocessing pipeline.
Key steps:
- One-hot encoding for categorical features
- Scaling for numerical features
- Train-test split
- 5-fold cross-validation
- Hyperparameter tuning using RandomizedSearchCV
- Final training on full dataset
  
The trained pipeline is saved as:
house_price_pipe.pkl

# 📈 Model Results

The model performs well with:
- Low MAPE (Mean Absolute Percentage Error)
- Strong R² score
- Meaningful feature importance rankings

# 📂 Project Structure
```
Titanic-Survival-Prediction/
│
├── app/ # Main application directory
│ ├── main.py # Streamlit application entry point
│ └── background_image.py # Background image handling utility
│
├── data/ # Dataset directory
│ ├── titanic_dataset.csv # Complete Titanic dataset
│ └── titanic_sample_data.csv # Sample data for testing
│
├── models/ # Trained ML models
│ ├── model_rf.pkl # Random Forest model
│ ├── model_lr.pkl # Logistic Regression model
│ └── model_voting.pkl # Voting Classifier model
│
├── notebooks/ # Jupyter notebooks for EDA & modeling
│ └── Titanic Survival Prediction.ipynb
│
├── Dockerfile # Docker configuration for containerization
├── Procfile # Deployment configuration for Heroku/Railway
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # License file

```



