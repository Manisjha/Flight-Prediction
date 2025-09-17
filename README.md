# ✈️ Flight Price Prediction using EDA & Feature Engineering

This project is a hands-on implementation of **Exploratory Data Analysis (EDA)** and **Feature Engineering** techniques on a real-world flight fare dataset. Inspired by [Krish Naik's Live Day 3 session](https://www.youtube.com/watch?v=v5dqavbyE-I), the goal is to preprocess raw flight data and build a robust foundation for predictive modeling.

## 📌 Project Objectives

- Clean and preprocess flight fare data
- Engineer meaningful features from raw columns
- Encode categorical variables for model readiness
- Prepare the dataset for machine learning workflows

---

## 📂 Dataset Overview

The dataset contains flight details such as:

- Airline name
- Date of journey
- Source and destination cities
- Departure and arrival times
- Duration of flight
- Number of stops
- Price of the ticket (target variable)

📥 Dataset Source: [Krish Naik’s GitHub](https://github.com/krishnaik06/5-Days-EDA-FeatureEngineering)

---

## 🧪 Technologies Used

- Python 3.8+
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 🔍 Exploratory Data Analysis (EDA)

Key steps performed:

- **Missing Value Treatment**: Checked and imputed missing values.
- **Datetime Conversion**: Extracted day, month, and hour from `Date_of_Journey`, `Dep_Time`, and `Arrival_Time`.
- **Duration Parsing**: Converted `Duration` from string format (e.g., "2h 50m") to total minutes using string manipulation and `eval()`.

```python
df['Duration'] = df['Duration'].str.replace("h", "*60").str.replace("m", "*1").str.replace(" ", "+").apply(eval)
```

- **Route Analysis**: Split `Route` into multiple legs to analyze stopovers.
- **Outlier Detection**: Used boxplots and distribution plots to identify skewed features.

---

## 🛠️ Feature Engineering

- **Categorical Encoding**:
  - Applied **Label Encoding** for ordinal features.
  - Used **One-Hot Encoding** for nominal features like `Airline`, `Source`, and `Destination`.

- **New Features Created**:
  - `Journey_Day`, `Journey_Month`
  - `Dep_Hour`, `Dep_Minute`
  - `Arrival_Hour`, `Arrival_Minute`
  - `Duration_Minutes`
  - `Total_Stops` (converted from text to numeric)

- **Feature Selection**:
  - Dropped redundant columns like `Route`, `Additional_Info`, and original time columns after transformation.

---

## 📈 Model Preparation (Optional)

While the focus is on EDA and feature engineering, the dataset is now ready for:

- Regression models (Linear, Random Forest, XGBoost)
- Hyperparameter tuning
- Cross-validation

You can save the preprocessed dataset using `pickle` for future model training:

```python
import pickle
pickle.dump(df, open('flight_data_cleaned.pkl', 'wb'))
```

---

## 📊 Visualizations

- Distribution plots for price and duration
- Boxplots to detect outliers
- Count plots for categorical features
- Correlation heatmap (optional)

---

## 📁 Project Structure

```
flight-price-prediction/
│
├── data/
│   └── flight_data.csv
│
├── notebooks/
│   └── eda_feature_engineering.ipynb
│
├── src/
│   └── preprocessing.py
│
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/eda_feature_engineering.ipynb
   ```

---

## 🧠 Learnings

- Hands-on experience with real-world data preprocessing
- Importance of feature engineering in improving model performance
- Practical use of encoding techniques and time-based transformations

---

## 🙌 Acknowledgments

Special thanks to [Krish Naik](https://www.youtube.com/@KrishNaik) for his insightful live sessions and open-source contributions.

---
