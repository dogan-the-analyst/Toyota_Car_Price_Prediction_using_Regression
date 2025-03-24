**Toyota Car Price Prediction Using Regression Models**

### **Project Overview**
This project aims to develop a machine learning model to predict the prices of used Toyota cars based on various attributes. By leveraging regression techniques, we analyze key factors influencing car prices, such as mileage, age, fuel type, and transmission. The ultimate goal is to build an accurate predictive model that can assist dealerships and buyers in making informed pricing decisions.

### **Dataset**
- **Source**: [Used Car Dataset - Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data)
- **File Used**: `toyota.csv`
- **Features**: The dataset includes car specifications such as:
  - Model
  - Year
  - Transmission Type
  - Mileage
  - Fuel Type
  - Tax
  - Engine Size
  - Price (Target Variable)

### **Data Preprocessing & Cleaning**
- **Removed Duplicates**: 39 duplicate rows were identified and removed.
- **Checked for Missing Values**: The dataset was assessed for null values and was found to have no missing data.
- **Feature Engineering**:
  - Converted categorical variables (Transmission, Fuel Type) into numerical values using encoding techniques.
  - Normalized numerical features using MinMaxScaler for improved model performance.
  - Created new derived features for better model accuracy.

### **Modeling Approach**
We trained multiple regression models to predict car prices, evaluating their performance to select the best one:
1. **Linear Regression** – Baseline model for comparison.
2. **Random Forest Regressor** – Captures non-linear relationships.
3. **Neural Networks (TensorFlow/Keras)** – Explores deep learning potential.

Each model was trained using **train-test split** and hyperparameter tuning to optimize results.

### **Evaluation Metrics**
The models were evaluated using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared Score** (R²)

The Random Forest Regressor outperformed other models, providing the most accurate predictions.

### **Insights & Business Impact**
✔ **Age and mileage are key price determinants** – Older, high-mileage cars tend to have significantly lower prices.
✔ **Fuel type & transmission affect pricing** – Hybrid and automatic cars generally have higher resale values.
✔ **Machine learning can assist dealerships** – Predictive models help set competitive prices, improving sales strategies and profitability.

### **Technologies Used**
- **Python** (Pandas, NumPy, Seaborn, Matplotlib)
- **Machine Learning** (Scikit-learn, TensorFlow/Keras)
- **Data Preprocessing** (MinMaxScaler, Train-Test Split)
- **Jupyter Notebook** (For analysis and visualization)

### **Conclusion**
By leveraging machine learning, this project provides a data-driven approach to pricing used Toyota cars. The findings can help dealerships optimize their pricing strategy and assist consumers in making informed purchasing decisions. Future improvements could include additional data sources and more advanced deep learning models to enhance prediction accuracy.

