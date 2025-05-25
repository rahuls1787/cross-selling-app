

```markdown
# 🛒 Cross Selling Prediction App

A Streamlit web application that predicts the likelihood of customers purchasing additional insurance products based on input features. This app integrates a trained ensemble machine learning model and OpenAI for explanation generation.

## 🚀 Live Demo

👉 [Try the app on Streamlit](https://cross-selling-app-dvnuuju7otzr6i6ax2kxnj.streamlit.app/)

---

## 📁 Project Structure

```

Project\_STI/
├── app.py                  # Main Streamlit app
├── dataset\_insu.csv        # Dataset (for reference/training)
├── ensemble\_model.pkl      # Trained ML model
├── ensemble\_model.json     # Optional: model in JSON format
├── preprocessor.pkl        # Data preprocessing pipeline
├── requirements.txt        # Python dependencies
└── .streamlit/
└── secrets.toml        # API key config (excluded from Git)

````

---

## ✨ Features

- Upload customer data or enter manually
- Predict cross-sell probability using trained model
- Generate explanation using OpenAI GPT
- Clean, interactive UI with Streamlit

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/rahuls1787/cross-selling-app.git
cd cross-selling-app
````

2. **Create Virtual Environment and Install Dependencies**

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

3. **Set OpenAI API Key**

Create a file `.streamlit/secrets.toml` and add:

```toml
[api_keys]
openai_api_key = "your_openai_key_here"
```

✅ This file is ignored in Git using `.gitignore`.

4. **Run the App Locally**

```bash
streamlit run app.py
```

---

## 🧠 Tech Stack

* Python
* Streamlit
* scikit-learn
* OpenAI GPT API
* Pandas, NumPy

---

## 📊 Model Overview

* Preprocessing: StandardScaler, LabelEncoder, OneHotEncoder
* Model: Ensemble (e.g., VotingClassifier or custom stack)
* Output: Probability of cross-sell & explanation using GPT

---

## 📦 Deployment

Deployed on [Streamlit Cloud](https://streamlit.io/cloud).
Push changes to GitHub → auto-reflects in the app.

---

## 🛡️ License

This project is part of the Hack2Future Hackathon 2025.

---

## 🙋‍♂️ Author

**Rahul Sarawale**
📌 [GitHub](https://github.com/rahuls1787)

---

````
