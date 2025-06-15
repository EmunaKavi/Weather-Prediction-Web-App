# 🌦️ Weather Prediction Web App using LSTM

A real-time weather forecasting application built as part of my internship at **Novanectar Services Pvt. Ltd.**

This app predicts the **next hour's Temperature, Humidity, and Pressure** using an LSTM (Long Short-Term Memory) neural network trained on historical weather data.

---

## 🚀 Features

- 🔮 **Predicts weather parameters:**
  - 🌡️ Temperature (°C)
  - 💧 Humidity (0–1)
  - 🌬️ Pressure (millibars)
- 🧠 Based on 24-hour historical patterns
- 💻 User-friendly web interface with Flask
- 🔄 Real-time inputs with immediate prediction

---

## 🧰 Tech Stack

| Component        | Technology           |
|------------------|----------------------|
| ML Model         | LSTM (Keras/TensorFlow) |
| Web Framework    | Flask (Python)       |
| Data Handling    | Pandas, NumPy        |
| Visualization    | Matplotlib (for testing) |
| Normalization    | MinMaxScaler (scikit-learn) |
| Model I/O        | Joblib               |
| Frontend         | HTML + CSS (basic UI) |

---

## 📁 Project Structure

```
weather-lstm-app/
├── app.py                    # Flask web server
├── model_train.py            # Script to train LSTM model
├── model_test.py             # Script to test and plot model results
├── weather_lstm_model.h5     # Trained LSTM model
├── scaler.save               # Saved MinMaxScaler
├── weatherHistory.csv        # Kaggle dataset used for training
├── templates/
│   └── index.html            # Web UI template
└── requirements.txt          # Project dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/weather-lstm-app.git
cd weather-lstm-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**
```txt
flask
numpy
pandas
matplotlib
scikit-learn
joblib
tensorflow
```

### 3. Train the Model
```bash
python model_train.py
```

### 4. Run the Web Application
```bash
python app.py
```

Then visit: **http://127.0.0.1:5000/** in your browser.

---

## 📊 Dataset Information

- **Source:** [Kaggle Weather History Dataset](https://www.kaggle.com)
- **Features used:**
  - Temperature (°C)
  - Humidity
  - Pressure (millibars)
- **Training approach:** 24-hour sliding window for LSTM input

---

## 🎯 Sample Prediction

### Input:
```
Temperature: 22.5 °C
Humidity: 0.78
Pressure: 1012.4 mb
```

### Output:
```
Next Hour Temperature: 22.63 °C
Next Hour Humidity: 0.76
Next Hour Pressure: 1012.70 mb
```

---

## 🏗️ Model Architecture

- **Model Type:** LSTM (Long Short-Term Memory)
- **Input Shape:** (24, 3) - 24 hours of Temperature, Humidity, Pressure
- **Output:** 3 weather parameters for the next hour
- **Preprocessing:** MinMaxScaler normalization
- **Framework:** TensorFlow/Keras

---

## 🚀 Future Enhancements

- [ ] Add more weather parameters (wind speed, visibility)
- [ ] Implement multi-day forecasting
- [ ] Add weather visualization charts
- [ ] Deploy to cloud platform
- [ ] Mobile-responsive UI improvements

---

## 🙌 Acknowledgements

This project was completed as part of my internship at **Novanectar Services Pvt. Ltd.**

Big thanks to the team for their support and guidance throughout this project!

---

## 📬 Contact

**Gokul Krishnan**
- 📧 Email: your.email@example.com
- 🔗 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)
- 🔗 GitHub: [Your GitHub Profile](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Made with ❤️ during my internship at Novanectar Services Pvt. Ltd.*
