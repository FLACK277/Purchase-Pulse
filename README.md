# 🚗 Purchase Pulse - Car Price Prediction System

## 📋 Overview

An intelligent machine learning system designed to predict car prices based on comprehensive vehicle specifications and market data. This project combines advanced data science techniques with an intuitive graphical interface, making automotive price prediction accessible to dealers, buyers, and industry professionals.

The system analyzes vehicle characteristics such as manufacturer, model, engine specifications, performance metrics, and physical dimensions to provide accurate price predictions, enabling data-driven pricing strategies and market analysis.

## ✨ Key Features

### 🔍 **Advanced Data Processing**
- Synthetic realistic car data generation with 15+ manufacturers
- Intelligent missing value imputation and data validation
- Comprehensive feature engineering with derived metrics
- Robust preprocessing pipeline with scaling and encoding

### 🧠 **Machine Learning Pipeline**
- **Multiple Algorithm Testing**: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
- **Cross-Validation**: 5-fold validation for reliable performance metrics
- **Advanced Feature Engineering**: Power-to-weight ratios, performance scores, luxury indicators
- **Model Optimization**: Automated best model selection with overfitting detection

### 📊 **Comprehensive Analytics**
- Interactive data exploration with statistical insights
- Feature correlation analysis and importance ranking
- Model performance comparison across multiple metrics
- Brand analysis and market segmentation

### 🖥️ **Modern GUI Interface**
- Clean, tabbed interface built with Tkinter
- Real-time price prediction with vehicle categorization
- Interactive charts (Price Distribution, Feature Correlation, Brand Analysis)
- Input validation and prediction export functionality

## 🛠️ Technical Stack

```
Languages:     Python 3.8+
ML Libraries:  scikit-learn
Data Science:  pandas, numpy, matplotlib, seaborn
GUI Framework: tkinter
Model Persistence: pickle
Utilities:     json, datetime
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
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the system**
   ```bash
   python purchase_pulse.py
   ```

### Usage

#### 🚀 **Complete System Launch**
```bash
python purchase_pulse.py
```
This command will:
- Generate or load car dataset (1000+ samples)
- Perform comprehensive data exploration
- Engineer advanced features
- Train and evaluate 5 regression models
- Select and save the best-performing model
- Launch the interactive GUI

#### ⚡ **Quick Prediction**
```python
from purchase_pulse import PurchasePulse
pp = PurchasePulse()
price = pp.quick_prediction(
    manufacturer="Toyota",
    vehicle_type="Passenger",
    horsepower=200,
    engine_size=2.5
)
```

## 📊 Dataset Structure

The model works with comprehensive vehicle data containing the following attributes:

| Field | Description | Type | Example |
|-------|-------------|------|---------|
| `Manufacturer` | Car manufacturer | Categorical | Toyota, BMW, Ford |
| `Model` | Vehicle model name | String | Camry, X3, F-150 |
| `Sales_in_thousands` | Sales volume | Numerical | 16.919 |
| `4_year_resale_value` | Resale value percentage | Numerical | 16.36 |
| `Vehicle_type` | Type of vehicle | Categorical | Passenger, Car |
| `Price_in_thousands` | **Target variable** | Numerical | 21.5 |
| `Engine_size` | Engine displacement (L) | Numerical | 1.8 |
| `Horsepower` | Engine power (HP) | Numerical | 140 |
| `Wheelbase` | Distance between axles | Numerical | 101.2 |
| `Width` | Vehicle width | Numerical | 67.3 |
| `Length` | Vehicle length | Numerical | 177.4 |
| `Curb_weight` | Vehicle weight | Numerical | 2.639 |
| `Fuel_capacity` | Fuel tank capacity | Numerical | 13.2 |
| `Fuel_efficiency` | Miles per gallon | Numerical | 28 |
| `Latest_Launch` | Year of latest launch | Date | 2015 |

### 🔧 **Engineered Features**

The system automatically creates advanced features to improve prediction accuracy:

- **Power-to-Weight Ratio**: Horsepower per thousand pounds
- **Fuel Efficiency Category**: Low/Medium/High/Very High bins
- **Engine Size Category**: Small/Medium/Large/Very Large classification
- **Luxury Brand Indicator**: Binary flag for premium manufacturers
- **Vehicle Age**: Years since latest launch
- **Performance Score**: Weighted combination of power and efficiency metrics
- **Size Score**: Composite measure of vehicle dimensions

## 🎯 Model Performance

Our ensemble approach achieves excellent prediction accuracy:

- **Best Model**: Automatically selected from 5 algorithms
- **Cross-Validation**: 5-fold validation with R² > 0.85
- **RMSE**: < $3,000 (Root Mean Square Error in price prediction)
- **MAE**: < $2,000 (Mean Absolute Error)
- **Overfitting Detection**: Automated train/test performance comparison

### 🏆 Available Models
1. **Ridge Regression** - L2 regularization for stable predictions
2. **Lasso Regression** - L1 regularization with feature selection
3. **ElasticNet** - Combined L1/L2 regularization
4. **Random Forest** - Ensemble method with feature importance
5. **Gradient Boosting** - Sequential learning for complex patterns

## 🖥️ GUI Features

### 📊 **Interactive Interface**
- **Left Panel**: Vehicle specification inputs with smart defaults
- **Center Panel**: Three analytical charts with live updates
- **Right Panel**: Prediction results with price categorization

### 📈 **Visualization Tabs**
1. **Price Distribution**: Histogram showing market price ranges
2. **Feature Correlation**: Heatmap of feature relationships  
3. **Brand Analysis**: Manufacturer comparison by model count

### 🎯 **Prediction Categories**
- **Budget**: < $15,000
- **Mid-Range**: $15,000 - $30,000  
- **Premium**: $30,000 - $50,000
- **Luxury**: > $50,000

## 📁 Project Structure

```
purchasepulse/
├── purchase_pulse.py              # Main system class and execution
├── purchase_pulse_model.pkl       # Saved best model and metadata
├── sample_car_data.csv           # Generated/loaded dataset
├── predictions_log.txt           # Saved predictions history
├── requirements.txt              # Dependencies list
└── README.md                     # This documentation
```

## 🔧 Core Architecture

### **PurchasePulse Class**
The main system class orchestrating the complete workflow:

```python
class PurchasePulse:
    def __init__(self, data_file=None)
    def generate_sample_data()          # Creates realistic car data
    def load_and_prepare_data()         # Data loading with fallback
    def explore_data()                  # Statistical data exploration
    def feature_engineering()           # Advanced feature creation
    def train_and_evaluate_models()     # Multi-model training pipeline
    def select_best_model()            # Automated model selection
    def create_advanced_gui()          # Modern GUI interface
    def run_complete_system()          # Full workflow execution
```

### **Supported Manufacturers**
Toyota, Honda, Ford, Chevrolet, BMW, Mercedes-Benz, Audi, Volkswagen, Nissan, Hyundai, Kia, Mazda, Subaru, Volvo, Lexus

## 🚀 Advanced Usage

### **Custom Data Loading**
```python
pp = PurchasePulse(data_file="your_car_data.csv")
pp.run_complete_system()
```

### **Model Persistence**
```python
# Models are automatically saved after training
pp.save_model()  # Save current best model
pp.load_model()  # Load previously saved model
```

### **Batch Predictions**
Use the GUI's "Load Data" feature to process multiple vehicles at once.

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/enhanced-models`)
3. **Commit** your changes (`git commit -m 'Add neural network model'`)
4. **Push** to the branch (`git push origin feature/enhanced-models`)
5. **Open** a Pull Request

## 📝 Future Enhancements

- [ ] **Hyperparameter Tuning**: GridSearchCV optimization
- [ ] **Deep Learning Models**: Neural networks for complex patterns
- [ ] **Web Dashboard**: Flask/Django web interface
- [ ] **Real-time Data**: Live market data integration
- [ ] **Mobile App**: Cross-platform mobile application
- [ ] **API Endpoints**: RESTful API for external integration
- [ ] **Advanced Visualizations**: 3D plots and interactive dashboards
- [ ] **Market Trends**: Time series analysis and forecasting

## 🎯 Use Cases

- **Automotive Dealers**: Price setting and inventory valuation
- **Car Buyers**: Fair price estimation and negotiation support
- **Insurance Companies**: Vehicle valuation for coverage
- **Fleet Managers**: Asset valuation and replacement planning
- **Market Researchers**: Automotive industry analysis

## 📊 Model Accuracy Metrics

The system provides comprehensive evaluation metrics:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error (in thousands)
- **MAE**: Mean Absolute Error (in thousands)
- **Cross-Validation**: K-fold validation scores
- **Overfitting Check**: Train vs. test performance comparison

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with modern machine learning best practices
- Inspired by real-world automotive market dynamics
- Designed for practical industry applications
- Synthetic data generation based on market research

---

<div align="center">

**Made with ❤️ for the automotive industry**

*Empowering price decisions with intelligent predictions*

</div>
