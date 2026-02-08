
# 🩺 AI-Based Disease Prediction System

An intelligent web application that predicts possible diseases based on user-entered symptoms and provides **easy-to-understand health awareness explanations** along with **visual references**, using Machine Learning and Generative AI.
⚠️ *This project is strictly for educational and awareness purposes and does NOT provide medical diagnosis.*

---

## 🚀 Project Features

* Predicts disease based on symptoms using a trained ML model
* Converts user symptoms into a machine-understandable feature vector
* Generates **simple, point-wise explanations** using a lightweight LLM
* Dynamically fetches **relevant medical images** for awareness
* Avoids medical diagnosis and technical jargon (user-friendly)
* Interactive web interface built using Streamlit

---

### 💻 Programming & Frameworks

* *Python 3*
* *Streamlit* – Web UI
* *Scikit-learn* – Machine Learning
* *Joblib* – Model serialization
* *Requests* – API calls

###  AI & ML

* Supervised ML Model (trained on symptom-based dataset)
* Pipeline with SimpleImputer
* Ollama (phi model) – Lightweight LLM for explanations

### External APIs

SerpAPI (Google Images) – Fetch symptom-related images dynamically

---

## 📂 Project Structure

```
├── ml_app.py                # Main Streamlit application
├── disease_model.joblib     # Trained ML model
├── README.md                # Project documentation
```

---

##  Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-disease-prediction.git
cd ai-disease-prediction
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Required Packages

```bash
pip install streamlit scikit-learn joblib ollama requests
```

---

## 🔑 API & Model Setup

### 🔹 SerpAPI

* Create an account at **[https://serpapi.com](https://serpapi.com)**
* Copy your API key
* Paste it inside `ml_app.py`:

```python
SERP_API_KEY = "YOUR_SERPAPI_KEY"
```

### 🔹 Ollama

Install Ollama and pull the lightweight model:

```bash
ollama pull phi
```

---

## ▶️ How to Run the Application

```bash
streamlit run ml_app.py
```
(      (or)
```bash
python -m streamlit run ml_app.py
```
Then open in browser:

```
http://localhost:8501
```

---

## 🧪 Example Usage

**Input Symptoms:**

```
itching, skin_rash, redness
```

**Output:**

* Predicted disease
* Point-wise explanation
* Severity level
* Home-care tips
* When to consult a doctor
* Related medical images (side-by-side)

---

## ⚠️ Disclaimer

> This application is built for **educational and awareness purposes only**.
> It does **NOT** provide medical advice, diagnosis, or treatment.
> Always consult a certified medical professional for health concerns.

---

## 🌟 Future Improvements

* 🔢 Confidence score for predictions
* 🧬 Top-3 disease suggestions
* 📊 Symptom visualization
* 📱 Mobile-responsive UI
* 🧠 Image-based disease detection using CNNs


![WhatsApp Image 2026-02-08 at 6 18 59 PM](https://github.com/user-attachments/assets/196fd259-32f0-4aaf-a1c1-b81a4dee00b3)

![WhatsApp Image 2026-02-08 at 6 18 59 PM (1)](https://github.com/user-attachments/assets/50419910-806f-4fcb-bcb6-a84bf310140a)

![WhatsApp Image 2026-02-08 at 6 18 59 PM (2)](https://github.com/user-attachments/assets/0d76cc2e-a037-4b9f-907e-51681332363d)

