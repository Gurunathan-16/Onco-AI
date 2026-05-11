# Onco AI – Multi-Cancer Detection System

## Overview

Onco AI is an AI-powered web application designed to assist in early cancer detection using deep learning models.
The system allows users to upload medical images and receive AI-based predictions for different cancer types.

The project is built using:

* Python
* Flask
* TensorFlow / Keras
* HTML/CSS
* SQLite

---

# Features

* User Authentication System

  * Login
  * Registration
  * Session Management

* AI-Based Cancer Detection

  * Breast Cancer Detection
  * Oral Cancer Detection

* Deep Learning Integration

  * TensorFlow/Keras models
  * Image preprocessing
  * Real-time prediction

* Prediction History

* Secure Database Storage

* Responsive Web Interface

---

# Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript

## Backend

* Flask
* Flask-Login
* SQLAlchemy

## AI/ML

* TensorFlow
* Keras
* NumPy
* OpenCV
* Pillow

## Database

* SQLite

---

# Project Structure

```plaintext
Onco-AI/
│
├── app_flask.py
├── config.py
├── models.py
├── requirements.txt
│
├── models/
│   ├── breast_model.keras
│   └── oral_model.keras
│
├── utils/
│   └── model_loader.py
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── prediction.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
└── database/
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd Onco-AI
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Run Application

```bash
python app_flask.py
```

---

# 5. Open Browser

```plaintext
http://127.0.0.1:5000
```

---

# AI Model Information

## Breast Cancer Model

* Framework: TensorFlow/Keras
* File: `breast_model.keras`

## Oral Cancer Model

* Framework: TensorFlow/Keras
* File: `oral_model.keras`

---

# Future Enhancements

* Multi-cancer support
* PDF report generation
* Cloud deployment
* Doctor dashboard
* AI explainability
* Confidence visualization
* Medical report export
* Mobile application

---

# Screenshots

Add screenshots of:

* Login Page
* Registration Page
* Dashboard
* Prediction Results

---

# Security Features

* Password hashing
* User authentication
* Session protection
* Secure file uploads

---

# Author

**Gurunathan R**
M.Sc Computer Science – AI/ML Enthusiast

---

# License

This project is developed for educational and research purposes.

---

# Disclaimer

This application is intended for research and educational use only.
It should not be considered a replacement for professional medical diagnosis.
