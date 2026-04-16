# 🛡️ EnsemblePhish — AI-Powered Phishing URL Detection System

EnsemblePhish is a **professional cybersecurity project** that detects phishing URLs using a **hybrid approach of rule-based analysis and machine learning ensemble models**.
It is designed to provide **accurate, explainable, and real-time detection** through an interactive web interface built with Streamlit.

---

## 🚀 Features

* 🔗 **Real-time URL Detection**
* 📋 **Bulk URL Scanning**
* 📄 **PDF Link Extraction**
* 🧠 **Machine Learning Ensemble (RF + SVM + MLP)**
* ⚙️ **Rule-Based Security Engine**
* 📊 **Custom Dataset Upload & Model Retraining**
* 🎯 **High Accuracy (~96%+)**
* 💡 **Explainable Results (Why a URL is phishing/safe)**

---

## 🧠 How It Works

The system follows a **multi-layered architecture**:

1. **Whitelist Check**
   Trusted domains are instantly marked as safe.

2. **Feature Extraction**
   Extracts key URL features like:

   * IP address usage
   * URL length
   * HTTPS status
   * Special symbols (@, -)
   * Subdomains

3. **Rule-Based Scoring**
   Assigns a phishing risk score (0–100) based on suspicious patterns.

4. **Machine Learning Ensemble**

   * Random Forest 🌲
   * Support Vector Machine ⚡
   * Neural Network 🧠
     Majority voting improves prediction accuracy.

5. **Final Decision**
   Combines:

   * 60% ML prediction
   * 40% rule score

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **ML Models:** Scikit-learn
* **Libraries:** NumPy, Pandas
* **Optional:** PyMuPDF (for PDF link extraction)

---

## 📂 Project Structure

```
EnsemblePhish/
│── app.py
│── dataset.csv
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EnsemblePhish.git
cd EnsemblePhish
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 📊 Dataset Information

* 📌 Total URLs: **11,055**
* 📌 Features: **30**
* 📌 Labels:

  * `-1` → Phishing
  * `1` → Legitimate

---

---

## 🔥 Key Highlights

* Hybrid **Rule + ML approach**
* **Explainable AI output**
* Clean **cybersecurity-themed UI**
* Supports **real-world phishing detection use cases**

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**.
It should not be used as the sole security measure in production systems.

---

## 👨‍💻 Contributors
- Om Makwana
- Krishna Hadiya

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 🧠 Contribute improvements

---
