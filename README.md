
# 🚲 Bike Rental Demand Prediction

This project predicts **hourly bike rental demand** using the UCI Bike Sharing Dataset. It compares a standard Feedforward Neural Network (MLP) with a Long Short-Term Memory (LSTM) network to determine if modeling temporal sequences improves forecasting accuracy.

---

## 📂 Project Structure

```bash
Bike_Rental_Prediction/
│
├── assets/
│   └── model_results/              # EDA and model result plots (SVG)
│
├── data/
│   ├── hour.csv                    # Main dataset used
│   ├── day.csv
│   └── Readme.txt
│
├── logs/
│   ├── mlp_hist.pkl                # Training history (MLP)
│   └── lstm_hist.pkl               # Training history (LSTM)
│
├── models/
│   ├── mlp.py                      # MLP architecture
│   └── lstm.py                     # LSTM architecture
│
├── tools/
│   ├── preprocessor.pkl            # Saved sklearn preprocessing pipeline
│   └── y_scaler.pkl                # Target scaler
│
├── utils/
│   ├── dataset.py                  # Dataset + preprocessing + sequence logic
│   └── train.py                    # Training utilities
│
├── weights/
│   ├── weights_mlp.pt              # Trained MLP weights
│   └── weights_lstm.pt             # Trained LSTM weights
│
├── config.py                       # Hyperparameters and settings
├── main.py                         # Training entry point
├── test.ipynb                      # Testing and inference
├── EDA.ipynb                       # Exploratory Data Analysis
├── environment.yml                 # Conda environment
├── .gitignore
└── README.md

```

---

## 📊 Dataset

* **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
* **Target Variable**: `cnt` (Total hourly rental count)
* **Features**: Weather (temp, humidity, windspeed), Seasonality (season, month, hour), and Type of Day (holiday, workingday).

---

## 🛠️ Methodology

* **Temporal Preservation**: Data is split chronologically (no shuffling) to simulate real-world forecasting.
* **Cyclic Encoding**: Features like `hour` and `month` are transformed into  and  components to represent their periodic nature.
* **Sequential Learning**: The LSTM uses a **24-hour sliding window** to capture daily rhythms and trends.

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate bike

```

### 2. Training

Run the main script to preprocess data and train both models:

```bash
python main.py

```

### 3. Inference

Explore the results and visualize model performance:

```bash
jupyter notebook test.ipynb

```

---

## 📈 Results at a Glance

| Feature | MLP (Baseline) | LSTM (Time-Series) |
| --- | --- | --- |
| **Architecture** | Dense Layers | Recurrent Units |
| **Input Type** | Single Instance | 24-Hour Sequence |
| **Performance** | Good for general trends | **Superior** for peak hours |

*Detailed plots for loss curves and prediction comparisons are stored in `assets/model_results/`.*

---

## 👨‍💻 Author

**Ashirbad Parida**
