# 🌊 SwasthVision – Waterborne Disease Risk Prediction Model  
### AI-powered early risk detection for waterborne diseases across Indian states

SwasthVision is a machine-learning–based predictive model that estimates the **risk level of waterborne diseases** (such as cholera, diarrhea, dysentery, and typhoid) across Indian states using environmental and public-health indicators.

This project supports **early outbreak detection**, **public health planning**, and **data-driven decision-making**.

---

## 🚀 Features

### ✅ Machine Learning Model
- Predicts **Low / Medium / High disease risk**
- Trained on Indian datasets (rainfall, water contamination, sanitation index, temperature)
- Uses Random Forest / XGBoost for robust prediction
- Evaluated using accuracy, F1-score & confusion matrix

### ✅ Data Engineering
- Cleaned & preprocessed real-world environmental datasets
- Feature scaling, label encoding, and missing value treatment
- Converts raw environmental parameters into meaningful features

### ✅ Use Cases
- Government health departments  
- NGOs & rural health programs  
- Environmental safety monitoring  
- Research & academic analysis  

---

---

## 🧠 Technologies Used

| Category | Tools |
|---------|-------|
| Programming | Python |
| ML Frameworks | Scikit-Learn, Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| Deployment (optional) | Flask |
| Model Packaging | Pickle |

---

## 🔍 How the Model Works

### **1️⃣ Data Preprocessing**
- Removal of null values  
- Normalization of rainfall & temperature data  
- Encoding sanitation & contamination indicators  
- Preparing final clean dataset for training  

### **2️⃣ Model Training**
The model is trained on:
- Rainfall patterns  
- Temperature trends  
- Water contamination indicators  
- Sanitation index  
- Past disease outbreak patterns  

Algorithms used:
- Random Forest Classifier  
- XGBoost (optional)

### **3️⃣ Prediction Output Example**
State: Bihar
Predicted Risk Level: HIGH
Reason: Excess rainfall + poor sanitation + contaminated water sources.

🔮 Future Improvements

Add a live dashboard with real-time predictions

Integrate IMD rainfall API + water quality API

District-level disease risk forecasting

Build mobile-friendly public health app
