# 📈 Stock Price Predictor (Capstone 4 - ML Deployment)

This project is an end-to-end machine learning pipeline that predicts the next day's stock closing price based on the previous day's closing value. It's built as part of my Capstone 4 project focusing on MLOps, model packaging, and web deployment.

---

## 🚀 Project Overview

- 📊 **Domain**: Finance / Time Series Analysis  
- 🧠 **Model**: Scikit-Learn Linear Regression with StandardScaler  
- 🛠 **Tech Stack**: Python, yfinance, scikit-learn, joblib, Gradio  
- ☁️ **Deployment**: Hugging Face Spaces (Gradio UI)

---

## 📂 Project Structure

```
finance-time-series-mlops/
│
├── app.py                # Gradio web app interface
├── model.joblib          # Trained ML pipeline
├── requirements.txt      # Dependencies for deployment
└── README.md             # Project summary
```

---

## 📦 Features

- Retrieves historical stock data using `yfinance`
- Trains a regression model to forecast the next day's price
- Saves the model using `joblib`
- Deploys a web UI using `Gradio` for real-time prediction

---

## 🧪 Sample Input

**Enter**: Last closing price (e.g., `170.50`)  
**Output**: `Predicted next close: $172.10`

---

## 🛠 Installation (For Local Use)

```bash
pip install -r requirements.txt
python app.py
```

---

## 🌐 Live Demo

📍 
✅ Launches a shareable web UI using Gradio.

---

## 🤖 Model Details

- **Pipeline**: `StandardScaler + LinearRegression`
- **Target**: Next day’s closing price
- **Feature**: Previous day's closing price (`Lag1`)
- **Data Source**: [Yahoo Finance](https://finance.yahoo.com)

---

## 👩‍💻 Author

**Vineetha V**  
MS in ECE - Machine Learning and Data Science  
University of Southern California  
🔗 [GitHub](https://github.com/vineetha00)

---

## 📝 License

This project is open-sourced under the MIT License.
