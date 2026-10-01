# Voyage Analytics: Integrating MLOps in Travel

**Productionization of ML Systems**

Voyage Analytics is a travel-focused machine learning portfolio project that combines three different ML use cases into one end-to-end solution:

1. **Gender Classification Model** – classifies a user's gender from user profile information.
2. **Recommendation Model** – recommends hotels using both content-based and collaborative-filtering techniques.
3. **Flight Price Regression Model** – predicts flight prices using multiple regression algorithms and model evaluation techniques.

The main purpose of the project is not only to build machine learning models, but also to demonstrate the steps involved in taking ML work toward a reusable, production-oriented setup, including preprocessing, model evaluation, model serialization, and application integration.

---

## 1. Project Overview

The project works with travel-related datasets such as flights, hotels, and users. Each sub-project focuses on a different machine learning problem:

| Project | ML Problem | Main Output |
|---|---|---|
| Gender Classification | Classification | Predicted gender class |
| Hotel Recommendation | Recommendation | Ranked hotel recommendations |
| Flight Price Prediction | Regression | Predicted flight price |

Together, the three projects demonstrate classification, recommendation systems, and regression within a travel analytics context.

---

# 2. Project 1 — Gender Classification Model

### Objective

Build a classification model to categorize a user's gender using information from the user dataset.

### Data Used

The user data contains fields such as:

- `code`
- `company`
- `name`
- `gender`
- `age`

The preprocessing stage filters the target to the `male` and `female` categories.

### Key Steps

**Data ingestion and preprocessing**

The dataset is checked for missing values, duplicates, data types, and basic statistics before model building.

**Categorical encoding**

`LabelEncoder` is used to convert categorical values such as company and gender into numerical form.

**Text feature representation**

The `name` field is treated as text. A Sentence Transformer is used to convert names into numerical embeddings.

**Dimensionality reduction**

PCA is applied to the text embeddings to reduce their dimensionality. The notebook uses **23 PCA components** for this feature representation.

**Train-test split and scaling**

The data is split into training and testing sets using an 80/20 split, and `StandardScaler` is applied to the features.

### Models Evaluated

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

### Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

The notebook reports the following example test accuracies:

| Model | Reported Accuracy |
|---|---:|
| Logistic Regression | 96.1% |
| Decision Tree | 58.3% |
| Random Forest | 95.0% |
| Gradient Boosting | 97.2% |

The notebook defines **Logistic Regression as the benchmark model** and performs hyperparameter tuning on it before saving the tuned model.

### Saved Artifacts

The project saves the trained components using Pickle:

- `tuned_logistic_regression_model.pkl`
- `scaler.pkl`
- `pca.pkl`

Saving the preprocessing components along with the model allows the same transformations to be applied when new data is processed later.

---

# 3. Project 2 — Hotel Recommendation Model

### Objective

Build a recommendation engine that provides hotel suggestions based on hotel information and historical user interactions.

This project uses two recommendation approaches:

1. **Content-Based Recommendation**
2. **Collaborative Filtering**

---

## 3.1 Content-Based Recommendation

The content-based approach recommends hotels using the information associated with the hotel itself.

### Key Steps

**Hotel information preparation**

The hotel name and location are combined into a single field called `Hotel_Info`.

**TF-IDF representation**

`TfidfVectorizer` converts the hotel text information into numerical vectors.

**Cosine similarity**

Cosine similarity is calculated between hotel vectors to measure how similar the hotel records are to one another.

**Recommendation function**

The `get_hotel_recommendations()` function filters hotels using inputs such as:

- Hotel information
- Number of days
- Maximum price

The filtered results are then ranked using similarity scores.

---

## 3.2 Collaborative Filtering

The collaborative-filtering approach uses the historical interaction pattern between users and hotels rather than relying only on hotel descriptions.

### Key Steps

**User interaction filtering**

The project counts hotel interactions for each user and keeps users with at least two hotel interactions in the active filtering code. This helps reduce the impact of users with very little history.

**Hotel encoding**

Hotel names are converted to numeric IDs using `LabelEncoder`.

**User-hotel interaction data**

The project groups user and hotel interactions and aggregates the price information to create a user-item interaction dataset.

**User-item matrix**

A pivot table is created with users as rows and hotels as columns. Missing interactions are filled with zero.

**SVD matrix factorization**

Singular Value Decomposition (SVD) is applied to the user-item matrix. The notebook uses **8 latent factors** to represent hidden preference patterns.

The factorized matrices are then used to reconstruct predicted user-hotel interactions.

**Recommendation generation**

The `CFRecommender` class sorts predicted interaction scores, removes hotels already seen by the user, and returns the highest-scoring recommendations.

### Evaluation

The collaborative-filtering model is evaluated using Top-N recommendation metrics, particularly:

- Recall@2
- Recall@3

The notebook reports global results of approximately:

- **Recall@2: 89.4%**
- **Recall@3: 95.1%**

### Model Persistence

The collaborative-filtering recommender is serialized with Pickle so that the trained recommendation logic can be reused without rebuilding the model from scratch.

---

# 4. Project 3 — Flight Price Regression Model

### Objective

Predict the price of a flight using travel and flight-related information.

This is a regression problem because the target variable, flight price, is a continuous numerical value.

### Main Features

The project uses information including:

- Boarding city
- Destination
- Flight type
- Agency
- Travel time
- Travel distance
- Date-related features

### Key Steps

**Data ingestion and exploration**

The project loads flight, hotel, and user datasets and performs basic data-quality checks such as row counts, data types, missing values, duplicate values, and summary statistics.

**Date feature engineering**

The date field is converted to datetime and additional features are extracted, including:

- Day
- Month
- Year
- Weekday
- Week number

**Feature engineering**

A derived `flight_speed` feature is created using distance divided by travel time.

**Categorical encoding**

Categorical variables such as boarding city, destination, flight type, and agency are converted into numerical features using one-hot encoding.

**Feature selection**

The project applies an ANOVA F-test based feature-selection step and defines a final ordered feature set for modeling. Correlation and VIF analysis are also included to inspect relationships between predictors and possible multicollinearity.

**Train-test split and scaling**

The data is split into training and testing sets using an 80/20 split, followed by feature scaling with `StandardScaler`.

### Models Evaluated

- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### Hyperparameter Tuning

`GridSearchCV` is used with cross-validation to test different hyperparameter combinations for the regression models.

### Evaluation Metrics

The project uses metrics such as:

- MAE
- MSE
- RMSE
- R-squared
- Adjusted R-squared

The benchmark section also includes actual-vs-predicted and residual plots for additional model checking.

### Deployment

After model evaluation, the selected model and preprocessing scaler are saved using Pickle. A Flask application is then used to accept flight details, prepare the input in the same format used during training, apply the saved preprocessing, and return a predicted flight price.

---

# 5. End-to-End ML Workflow

Across the three projects, the overall workflow can be summarized as:

```text
Data Collection / Ingestion
        ↓
Data Understanding & Quality Checks
        ↓
Data Preprocessing
        ↓
Feature Engineering / Representation
        ↓
Train-Test Split
        ↓
Model Development
        ↓
Model Evaluation
        ↓
Hyperparameter Tuning
        ↓
Model Selection
        ↓
Model Serialization
        ↓
Application / Deployment Integration
```

The exact steps vary depending on the ML problem. For example, the recommendation project uses TF-IDF, cosine similarity and SVD, while the classification and regression projects use supervised learning models.

---

# 6. Technologies & Libraries

### Programming

- Python
- Jupyter Notebook / Google Colab

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- SciPy

### Natural Language / Text Representation

- Sentence Transformers
- TF-IDF

### Model & Feature Processing

- Label Encoding
- Standard Scaling
- PCA
- GridSearchCV
- SVD / Matrix Factorization

### Deployment / Persistence

- Flask
- Streamlit (for the recommendation project interface)
- Pickle

---

# 7. Suggested Repository Structure

```text
Voyage-Analytics-Integrating-MLOps-in-Travel/
│
├── README.md
│
├── Gender Classification Model/
│   ├── Project_of_Gender_Classification_Model.ipynb
│   ├── tuned_logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── pca.pkl
│
├── Recommendation Model/
│   ├── Project_of_Recommendation_model.ipynb
│   └── collaborative_filtering_model.pkl
│
└── Flight Price Regression Model/
    ├── Project_of_Flight_Price_Regression_Model.ipynb
    ├── trained_model.pkl
    └── scaler.pkl
```

The exact filenames can be adjusted to match the files included in the repository.

---

# 8. Notes

- The notebooks were developed in a Google Colab environment and use Google Drive paths for dataset access.
- Dataset files are not included in this repository unless explicitly added.
- Model artifacts should be reviewed before committing them to a public repository.
- API keys, authentication tokens, and other secrets must never be committed to GitHub.
- Some notebook cells are exploratory or diagnostic and are included to show the development process.

---

# 9. Project Outcome

Voyage Analytics brings together three different travel analytics use cases in a single portfolio project:

- **Classification:** user gender classification
- **Recommendation:** personalized hotel recommendation using content and interaction data
- **Regression:** flight price prediction

The project demonstrates the progression from raw travel data to machine learning models and reusable model artifacts, with deployment-oriented steps included where applicable.

---

## Author

**Mukesh Regar**

Master's in Data Science and Artificial Intelligence

Bachelor's in Commerce
