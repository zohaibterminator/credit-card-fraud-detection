# Fraud Detection using PaySim Simulation Data

This project was developed as the final assignment of a 1-month Data Science Workshop. It involves applying data analysis and machine learning techniques to detect fraudulent financial transactions using a simulated dataset from [PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).

## 📊 Dataset Overview

The dataset simulates mobile money transactions based on real-world financial data. It contains over 6 million transaction records with the following key fields:

- `step`: Time step unit (1 unit = 1 hour).
- `type`: Type of transaction (e.g., CASH-IN, CASH-OUT, DEBIT, etc.).
- `amount`: Amount of the transaction.
- `nameOrig`, `nameDest`: Customer IDs.
- `oldbalanceOrg`, `newbalanceOrig`: Sender's balance before and after transaction.
- `oldbalanceDest`, `newbalanceDest`: Recipient's balance before and after transaction.
- `isFraud`: Indicates whether the transaction is fraudulent.
- `isFlaggedFraud`: Indicates whether the transaction was flagged as suspicious.

## 🧠 Project Objectives

- Perform exploratory data analysis (EDA) to understand transaction patterns.
- Preprocess the data by handling imbalances and irrelevant fields.
- Build machine learning models to predict fraudulent transactions.
- Evaluate model performance using precision, recall, F1-score, and ROC-AUC.

## 🔍 Key Steps

1. **Exploratory Data Analysis (EDA):**
   - Checked data imbalance (`isFraud` is only ~0.1% of total).
   - Analyzed fraud trends by transaction type and amount.
   - Identified that most fraud occurs in `CASH_OUT` and `TRANSFER` types.

2. **Preprocessing:**
   - Removed identifier columns (`nameOrig`, `nameDest`) to prevent leakage.
   - One-hot encoded the `type` column.
   - Addressed class imbalance using under-sampling of the majority class.

3. **Modeling:**
   - Tested ML models including:
     - Logistic Regression
     - Multinomial Naive-Bayes
     - Random Forest
   - Tuned hyperparameters using GridSearchCV.

4. **Evaluation:**
   - Chose evaluation metrics suitable for imbalanced datasets.
   - Random Forest achieved the best results with high recall and F1-Score.

## ✅ Results

- **Best Model:** Random Forest
- **Accuracy:** ~0.87
- **Precision:** ~0.86
- **Recall:** ~0.87
- **F1-Score:** ~0.87
- Successfully identified fraud with good accuracy with reasonable amount of false negatives.

## 📁 Files

- `FraudDetection.ipynb`: Main Jupyter notebook with EDA, preprocessing, modeling, and evaluation.
- `README.md`: Project overview and documentation.
- `LICENCE`: MIT License for Open-Source projects.

## 🛠 Technologies Used

- Python 3.10
- Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn (for visualization)
- Jupyter Notebook

## 📌 Future Improvements

- Use SMOTE for synthetic oversampling instead of under-sampling.
- Deploy model as a web service for real-time fraud detection.
- Experiment with deep learning methods (e.g., LSTM for sequence analysis).

## 📚 References

- [Kaggle PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

---
