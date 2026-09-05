    # 📊 Sales Prediction Model with AI-Powered Insights

An end-to-end machine learning application that predicts future sales and uses an LLM to provide natural-language business insights and recommendations.

## 🚀 Project Overview

This project combines Machine Learning and Generative AI to help businesses analyze historical sales data and make better decisions.

The application:

- Analyzes historical sales data
- Predicts future sales using Machine Learning
- Identifies high-performing products
- Provides prediction results through an interactive Streamlit dashboard
- Uses an LLM to explain prediction results in natural language
- Answers user questions based on the prediction results
- Provides business recommendations and insights

## 🏗️ Architecture

User
↓
Streamlit Application
↓
User Question
↓
Machine Learning Model
↓
Sales Prediction Results
↓
LLM / Gemini
↓
Natural Language Explanation & Recommendation

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Google Gemini API
- Matplotlib
- Jupyter Notebook
- Git & GitHub

## 📂 Project Structure

```text
Model_Prediction/
│
├── data/
│   └── sales.csv
│
├── models/
│   └── trained ML models
│
├── notebook/
│   └── eda.ipynb
│
├── outputs/
│   └── prediction results
│
├── src/
│   ├── train.py
│   ├── multi_train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md       