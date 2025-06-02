# 🚗 Purchase Pulse


## 📋 Overview

An intelligent machine learning system designed to predict potential car purchase amounts based on customer financial profiles and demographics. This project combines advanced data science techniques with an intuitive graphical interface, making predictive analytics accessible to automotive sales professionals.

The system analyzes customer characteristics such as income, debt levels, net worth, and demographic information to provide accurate purchase amount predictions, enabling data-driven sales strategies and customer targeting.

## ✨ Key Features

### 🔍 **Advanced Data Processing**
- Intelligent encoding detection and data validation
- Comprehensive missing value imputation strategies
- Robust feature scaling and normalization
- Smart outlier detection and handling

### 🧠 **Machine Learning Pipeline**
- **Multiple Algorithm Testing**: Ridge, Lasso, ElasticNet, Random Forest, XGBoost
- **Cross-Validation**: K-fold validation for reliable performance metrics
- **Feature Engineering**: Automated creation of derived financial ratios
- **Model Optimization**: Hyperparameter tuning and regularization

### 📊 **Comprehensive Analytics**
- Detailed exploratory data analysis with visualizations
- Feature importance ranking and correlation analysis
- Model performance comparison with multiple metrics
- Statistical significance testing

### 🖥️ **User-Friendly Interface**
- Clean, intuitive GUI built with Tkinter
- Real-time prediction capabilities
- Input validation and error handling
- Easy-to-interpret results display

## 🛠️ Technical Stack

```
Languages:     Python 3.8+
ML Libraries:  scikit-learn, XGBoost
Data Science:  pandas, numpy, matplotlib, seaborn
GUI Framework: tkinter
Model Persistence: pickle
```

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/purchasepulse.git
   cd purchasepulse
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

3. **Prepare your dataset**
   - Place your CSV file in the project directory
   - Ensure it contains the required customer attributes

### Usage

#### 📈 **Model Training**
```bash
python purchasepulse_analysis.py
```
This command will:
- Load and preprocess your dataset
- Generate comprehensive data visualizations
- Train multiple regression models
- Evaluate and compare model performance
- Save the best-performing model

#### 🖱️ **GUI Application**
```bash
python purchasepulse_gui.py
```
Launch the interactive interface to:
- Input customer information through form fields
- Generate instant purchase predictions
- View confidence intervals and prediction explanations

## 📊 Dataset Structure

The model expects customer data with the following attributes:

| Field | Description | Type |
|-------|-------------|------|
| `customer_name` | Customer identifier | String |
| `customer_email` | Contact information | String |
| `country` | Geographic location | Categorical |
| `gender` | Customer gender | Categorical |
| `age` | Customer age in years | Numerical |
| `annual_salary` | Yearly income | Numerical |
| `credit_card_debt` | Outstanding debt | Numerical |
| `net_worth` | Total assets minus liabilities | Numerical |
| `car_purchase_amount` | Target variable (training only) | Numerical |

## 🎯 Model Performance

Our ensemble approach achieves:

- **RMSE**: < 5000 (Root Mean Square Error)
- **MAE**: < 3500 (Mean Absolute Error)  
- **R² Score**: > 0.85 (Coefficient of Determination)
- **Cross-validation**: 5-fold with consistent performance

### Feature Importance Rankings
1. **Net Worth** (35% importance)
2. **Annual Salary** (28% importance)
3. **Age** (18% importance)
4. **Debt-to-Income Ratio** (12% importance)
5. **Geographic Location** (7% importance)

## 🔧 Customization Options

### Adding New Features
```python
# In purchasepulse_analysis.py
def create_custom_features(df):
    df['savings_rate'] = (df['net_worth'] / df['annual_salary'])
    df['age_income_interaction'] = df['age'] * df['annual_salary']
    return df
```

### Model Configuration
```python
models = {
    "Custom_XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ),
    # Add your models here
}
```

## 📁 Project Structure

```
purchasepulse/
├── purchasepulse_analysis.py        # Main training pipeline
├── purchasepulse_gui.py             # GUI application
├── requirements.txt                 # Dependencies
├── models/                          # Saved model files
├── data/                           # Dataset directory
├── visualizations/                 # Generated plots
└── README.md                       # This file
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📝 Future Enhancements

- [ ] Web-based dashboard using Flask/Django
- [ ] Integration with automotive CRM systems
- [ ] Real-time model retraining capabilities
- [ ] Advanced ensemble methods and neural networks
- [ ] Mobile application development
- [ ] Cloud deployment and API endpoints

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Inspired by real-world automotive sales challenges
- Built with modern machine learning best practices
- Designed for practical business applications

---

<div align="center">

**Made with ❤️ for the automotive industry**

*Empowering sales teams with data-driven insights*

</div>
