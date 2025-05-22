# Sales Forecasting Using Classical and Deep Learning Models

## 📌 Project Overview
This project addresses the task of forecasting future sales for a bakery using historical sales data. Implemented as part of an assessment for GEMBO, the project includes a comprehensive analysis using multiple forecasting models such as Linear Regression, ARIMA, Auto ARIMA, and LSTM (Long Short-Term Memory). The goal is to accurately predict sales for a user-defined number of days.

## 🧠 Models Implemented
- **Linear Regression:** Baseline supervised model using time encoded as ordinal data.
- **ARIMA (1,1,1):** Classical time series forecasting method.
- **Auto ARIMA:** Automatically selected best (p,d,q) values for ARIMA.
- **LSTM (Basic and Tuned):** Deep learning models for sequential forecasting. Tuned version includes multiple LSTM layers, dense layers, and early stopping.

## 📈 Evaluation Metrics
Each model is evaluated using the following metrics:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R^2 Score (Coefficient of Determination)**
- **MAPE (Mean Absolute Percentage Error)**

## ✅ Best Performing Model
- The tuned **LSTM** model outperformed all others, delivering the most accurate forecasts, especially when capturing long-term dependencies.

## 📊 Dataset
- **Rows:** 173
- **Columns:**
  - `DATE` (converted to datetime)
  - `SALES` (numerical sales values)
- Cleaned and preprocessed for null values, formatting, and feature engineering (e.g., Date Ordinal, Month, Year).

## 🛠️ Tools and Libraries
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Statsmodels, pmdarima
- TensorFlow, Keras

## 📂 Project Structure
```
.
├── GEMBO_Assessment.ipynb         # Jupyter Notebook with all code and outputs
├── dataset.csv                    # Dataset used for training
├── Sales_Forecasting_Test_Cases.docx  # QA Test Case document
├── README.md                      # This README file
```

## 🚀 How to Run
1. Upload the notebook and dataset to Google Colab or run locally.
2. Run all cells in order. You can modify the `n_days` variable to forecast future sales.
3. View plots and evaluation metrics in the output.

## 🧪 Validation & Testing
- QA covered in `Sales_Forecasting_Test_Cases.docx`
- Includes data integrity checks, model accuracy benchmarks, and forecast sanity checks.

## 📌 Future Improvements
- Introduce external features (e.g., holiday calendar, promotions)
- Implement validation splits for time-series cross-validation
- Explore Prophet and Transformer models for further accuracy

## 👤 Author
**Ankush Kukde**  
GEMBO Assessment Submission
