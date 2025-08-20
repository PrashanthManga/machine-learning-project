# 📈 Stock Price Prediction using Machine Learning

This project demonstrates a **Machine Learning workflow** for predicting stock prices using historical data.  
It includes data preprocessing, model training, evaluation, and making predictions with a trained model.

---

## 🚀 Project Structure

- **main.py** – Python script for training and running the ML model.  
- **ML.ipynb** – Jupyter notebook for experimentation, visualization, and model testing.  
- **stock_predictions_model.keras** – Pre-trained Keras model for stock price prediction.  
- **ml_workflow.png** – Workflow diagram for the project.  

---

## ⚙️ Workflow

1. **Stock Data Collection** – Gather historical data (e.g., from Yahoo Finance APIs).  
2. **Data Preprocessing** – Clean, scale, and split data into training/testing sets.  
3. **Feature Engineering** – Create time windows and input sequences for training.  
4. **Model Training** – Train a deep learning model (LSTM/ANN) using Keras/TensorFlow.  
5. **Model Evaluation** – Assess model accuracy and loss metrics.  
6. **Prediction** – Forecast future stock prices using the trained model.  

---

## 🏗️ Architecture

```mermaid
flowchart TD
    %% --- User & App ---
    U[User Input (Ticker, Date Range, Horizon)] --> S[Streamlit App]

    %% --- Data Ingestion & Prep ---
    S --> YF[yfinance - Download OHLCV]
    YF --> PD[pandas + numpy - Clean & Feature Engineer]
    PD --> SC[Scaling - MinMax]
    SC --> SPLIT[Train/Test Split]

    %% --- Model Training ---
    subgraph TRAIN [Model Training]
      SPLIT -->|X_train, y_train| TF[Keras/TensorFlow (LSTM/MLP)]
      TF --> M[(model.keras)]
    end

    %% --- Inference & Visualization ---
    subgraph INFER [Inference & Visualization]
      S --> YF2[yfinance - Latest Data]
      YF2 --> PD2[pandas + numpy - same transforms]
      PD2 --> SC2[Apply saved scaler]
      SC2 --> M
      M --> PRED[Predicted Prices]
      PRED --> VIZ[Matplotlib Charts]
      VIZ --> S
      S --> OUT[Forecast plot, metrics, download]
    end
```
---

## 🛠️ Tech Stack

- **Python** (NumPy, Pandas, Matplotlib, Scikit-learn)  
- **TensorFlow / Keras** for deep learning  
- **Jupyter Notebook** for experimentation  
- **GitHub** for version control  

---



## 🏗️ Architecture

```mermaid
flowchart TD
  U[User Input ticker and date range] --> S[Streamlit App]

  S --> YF[yfinance download OHLCV]
  YF --> PD[pandas and numpy clean and feature engineer]
  PD --> SC[scaling MinMax]
  SC --> SPLIT[train test split]

  subgraph TRAIN [Model Training]
    SPLIT -->|X train and y train| TF[Keras TensorFlow model]
    TF --> M[(saved model keras)]
  end

  subgraph INFER [Inference and Visualization]
    S --> YF2[yfinance latest data]
    YF2 --> PD2[pandas and numpy same transforms]
    PD2 --> SC2[apply saved scaler]
    SC2 --> M
    M --> PRED[predicted prices]
    PRED --> VIZ[Matplotlib charts]
    VIZ --> S
    S --> OUT[forecast plot metrics and download]
  end
```
