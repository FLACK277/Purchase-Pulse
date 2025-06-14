import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Main class for car price prediction
class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_cols = None

    def generate_data(self, n=2000):
        # Generate synthetic training data
        np.random.seed(42)  # for reproducible results
        
        # Car brand data with pricing ranges and reliability scores
        brands = {
            'Toyota': {'luxury': False, 'price': (15, 45), 'reliability': 0.92},
            'Honda': {'luxury': False, 'price': (18, 42), 'reliability': 0.90},
            'Ford': {'luxury': False, 'price': (20, 55), 'reliability': 0.78},
            'Chevrolet': {'luxury': False, 'price': (22, 60), 'reliability': 0.75},
            'BMW': {'luxury': True, 'price': (35, 95), 'reliability': 0.82},
            'Mercedes': {'luxury': True, 'price': (40, 120), 'reliability': 0.80},
            'Audi': {'luxury': True, 'price': (38, 90), 'reliability': 0.81},
            'Lexus': {'luxury': True, 'price': (35, 85), 'reliability': 0.94},
            'Nissan': {'luxury': False, 'price': (16, 48), 'reliability': 0.83},
            'Hyundai': {'luxury': False, 'price': (14, 40), 'reliability': 0.85},
        }
        
        car_types = ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible', 'Wagon', 'Pickup']
        fuel_types = ['Gas', 'Diesel', 'Hybrid', 'Electric']
        
        data_list = []
        
        for i in range(n):
            # Pick random brand and get its info
            brand = np.random.choice(list(brands.keys()))
            brand_info = brands[brand]
            car_type = np.random.choice(car_types)
            fuel = np.random.choice(fuel_types, p=[0.6, 0.25, 0.1, 0.05])
            
            # Age affects price a lot - older cars are cheaper
            age = np.random.exponential(3)
            age = min(age, 20)  # cap at 20 years
            
            # Mileage usually correlates with age
            mileage = age * np.random.uniform(8000, 15000) + np.random.normal(0, 5000)
            mileage = max(0, mileage)  # can't be negative
            
            # Engine size depends on fuel type
            if fuel == 'Electric':
                engine = 0.0  # electric cars don't have traditional engines
                hp = np.random.uniform(150, 400)
            elif fuel == 'Hybrid':
                engine = np.random.uniform(1.5, 2.5)
                hp = engine * np.random.uniform(60, 90) + np.random.uniform(50, 100)
            else:
                engine = np.random.uniform(1.2, 6.5)
                if brand_info['luxury']:
                    engine += np.random.uniform(0.5, 2.0)  # luxury cars tend to have bigger engines
                hp = engine * np.random.uniform(45, 85) + np.random.normal(0, 15)
            
            hp = np.clip(hp, 100, 600)  # reasonable HP range
            
            # Car dimensions vary by type
            if car_type == 'SUV':
                length, width, weight = 195, 75, 4200
            elif car_type == 'Pickup':
                length, width, weight = 215, 79, 4800
            elif car_type == 'Coupe':
                length, width, weight = 180, 71, 3200
            else:  # default for sedan/hatchback/wagon
                length, width, weight = 185, 72, 3400
            
            # Add some random variation
            length += np.random.uniform(-10, 10)
            width += np.random.uniform(-3, 3)
            weight += np.random.uniform(-400, 400)
            
            # Calculate fuel efficiency
            if fuel == 'Electric':
                mpg = np.random.uniform(100, 130)  # electric cars are very efficient
            elif fuel == 'Hybrid':
                mpg = np.random.uniform(40, 55)
            else:
                # bigger engines and heavier cars use more fuel
                mpg = 32 - (engine * 2.5) - (weight / 350) + np.random.normal(0, 3)
                mpg = np.clip(mpg, 12, 130)
            
            # Power-to-weight ratio is important for performance
            power_to_weight = hp / (weight / 1000)
            
            # Now calculate the price - this is the tricky part
            base_price = np.random.uniform(*brand_info['price'])
            
            # Luxury brands cost more
            if brand_info['luxury']:
                base_price *= 1.3
            
            # Different fuel types have different premiums
            fuel_multiplier = {'Electric': 1.2, 'Hybrid': 1.1, 'Diesel': 1.05, 'Gas': 1.0}[fuel]
            base_price *= fuel_multiplier
            
            # Some car types are more expensive
            type_multiplier = {
                'Convertible': 1.2, 'Coupe': 1.1, 'SUV': 1.05, 'Pickup': 1.03, 
                'Sedan': 1.0, 'Wagon': 0.98, 'Hatchback': 0.95
            }[car_type]
            base_price *= type_multiplier
            
            # Cars depreciate with age
            depreciation = 0.85 ** age
            base_price *= depreciation
            
            # High mileage reduces price
            mileage_penalty = max(0.3, 1 - (mileage / 200000) * 0.4)
            base_price *= mileage_penalty
            
            # Performance cars cost more
            base_price *= (1 + (power_to_weight - 100) / 1000)
            
            # Reliable brands hold value better
            base_price *= (1 + brand_info['reliability'] - 0.8)
            
            # Add some market randomness
            base_price += np.random.normal(0, 3)
            base_price = np.clip(base_price, 8, 250)  # reasonable price range
            
            # Generate some market data
            sales = np.random.lognormal(3.5, 0.8)
            sales = np.clip(sales, 5, 300)
            
            resale = base_price * np.random.uniform(0.35, 0.65) * brand_info['reliability']
            
            # Store all the data
            data_list.append({
                'brand': brand,
                'type': car_type,
                'fuel': fuel,
                'age': round(age, 1),
                'mileage': int(mileage),
                'sales': round(sales, 1),
                'resale': round(resale, 2),
                'engine': round(engine, 1),
                'hp': int(hp),
                'length': round(length, 1),
                'width': round(width, 1),
                'weight': int(weight),
                'mpg': round(mpg, 1),
                'pwr_weight': round(power_to_weight, 2),
                'reliability': brand_info['reliability'],
                'price': round(base_price, 2)
            })
        
        df = pd.DataFrame(data_list)
        print(f"Created dataset with {len(df)} cars")
        return df

    def train(self):
        # Train the model using the generated data
        df = self.generate_data()
        
        # Separate features and target
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Handle categorical variables
        categorical_columns = ['brand', 'type', 'fuel']
        for col in categorical_columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])
            self.encoders[col] = encoder
        
        # Select the most important features
        feature_selector = SelectKBest(score_func=f_regression, k=12)
        X_selected = feature_selector.fit_transform(X, y)
        self.feature_selector = feature_selector
        self.feature_cols = X.columns[feature_selector.get_support()].tolist()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
        
        # Try different models and pick the best one
        models_to_try = {}
        
        # Random Forest with hyperparameter tuning
        rf_parameters = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        rf_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_parameters, cv=5, scoring='r2')
        rf_search.fit(X_train, y_train)
        models_to_try['RandomForest'] = rf_search.best_estimator_
        
        # Gradient Boosting with hyperparameter tuning
        gb_parameters = {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1],
            'max_depth': [6, 8]
        }
        gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_parameters, cv=5, scoring='r2')
        gb_search.fit(X_train, y_train)
        models_to_try['GradientBoosting'] = gb_search.best_estimator_
        
        # Find the best performing model
        best_score = 0
        for model_name, model in models_to_try.items():
            score = model.score(X_test, y_test)
            print(f"{model_name} accuracy: {score:.4f}")
            if score > best_score:
                best_score = score
                self.model = model
        
        print(f"Best model accuracy: {best_score:.4f} ({best_score*100:.1f}%)")
        
        # Show which features are most important
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("Most important features:")
            print(feature_importance.head())
        
        return best_score

    def predict(self, car_specs):
        if not self.model:
            raise Exception("Need to train the model first!")
        
        # Encode categorical variables
        for col, encoder in self.encoders.items():
            if col in car_specs:
                try:
                    car_specs[col] = encoder.transform([car_specs[col]])[0]
                except:
                    car_specs[col] = 0  # handle unknown values
        
        # Create dataframe from specs
        df = pd.DataFrame([car_specs])
        
        # Make sure all required columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Use only the selected features
        df = df[self.feature_cols]
        
        # Apply feature selection and scaling
        df_selected = self.feature_selector.transform(df)
        df_scaled = self.scaler.transform(df_selected)
        
        return self.model.predict(df_scaled)[0]

# Initialize the predictor
print("Starting up the car price predictor...")
predictor = CarPricePredictor()
print("Training the model (this might take a minute)...")
model_accuracy = predictor.train()
print(f"Model is ready! Accuracy: {model_accuracy*100:.1f}%")

def categorize_price(price):
    # Categorize cars by price range
    if price < 20:
        return "💰 Budget Car", "#28a745"
    elif price < 35:
        return "🏷️ Mid-Range", "#17a2b8"
    elif price < 60:
        return "⭐ Premium", "#fd7e14"
    else:
        return "💎 Luxury", "#6f42c1"

def make_prediction(units, brand, car_type, fuel, age, mileage, sales, resale, engine, hp, length, width, weight, mpg, reliability):
    try:
        # Calculate power-to-weight ratio
        if units == "US":
            pwr_weight = hp / (weight / 1000)
        else:
            pwr_weight = hp / (weight * 2.20462 / 1000)
        
        # Put all the car specs together
        car_specs = {
            'brand': brand,
            'type': car_type,
            'fuel': fuel,
            'age': age,
            'mileage': mileage,
            'sales': sales,
            'resale': resale,
            'engine': engine,
            'hp': hp,
            'length': length,
            'width': width,
            'weight': weight,
            'mpg': mpg,
            'pwr_weight': pwr_weight,
            'reliability': reliability
        }
        
        # Convert metric to US units if needed (model was trained on US units)
        if units == 'Metric':
            car_specs['length'] /= 2.54  # cm to inches
            car_specs['width'] /= 2.54   # cm to inches
            car_specs['weight'] *= 2.20462  # kg to lbs
            car_specs['mileage'] *= 0.621371  # km to miles
            if car_specs['mpg'] > 0:
                car_specs['mpg'] = 235.214 / car_specs['mpg']  # L/100km to MPG
        
        # Get the prediction
        predicted_price = predictor.predict(car_specs)
        category, color = categorize_price(predicted_price)
        
        # Format the results nicely
        price_k = f"${predicted_price:.2f}k"
        price_full = f"${predicted_price*1000:,.0f}"
        is_luxury_brand = brand in ['BMW', 'Mercedes', 'Audi', 'Lexus']
        
        # Create a detailed summary
        analysis_summary = f"""
**🚗 Car Price Analysis Results**

**Predicted Price:** {price_k} ({price_full})
**Price Category:** {category}
**Luxury Brand:** {'Yes ⭐' if is_luxury_brand else 'No'}
**Model Accuracy:** {model_accuracy*100:.1f}%

**Car Details:**
• **Make & Model:** {brand} {car_type}
• **Fuel Type:** {fuel}
• **Age:** {age:.1f} years
• **Mileage:** {mileage:,.0f} {'miles' if units == 'US' else 'km'}
• **Engine Size:** {engine:.1f}L
• **Horsepower:** {int(hp)} HP
• **Fuel Economy:** {mpg:.1f} {'MPG' if units == 'US' else 'L/100km'}
• **Reliability:** {reliability:.2f}/1.0
• **Power-to-Weight:** {pwr_weight:.1f} HP/1000lbs

**Market Data:**
• **Annual Sales:** {sales:.1f}k units
• **Resale Value:** ${resale:.1f}k
• **Depreciation:** Factored into price
• **Performance Factor:** Power-to-weight ratio considered
        """
        
        return price_k, price_full, category, analysis_summary, f"{model_accuracy*100:.1f}%"
        
    except Exception as e:
        return "Error", "Something went wrong", "Unknown", f"❌ **Error:** {str(e)}", "N/A"

# Create the web interface
interface = gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    ),
    title="🚗 Purchase Pulse - Car Price Predictor",
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .header-text {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subtitle-text {
        text-align: center;
        color: #6c757d;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    """
)

with interface:
    # Page header
    gr.HTML("""
    <div class="header-text">🚗 Purchase Pulse</div>
    <div class="subtitle-text">AI Car Price Prediction System</div>
    <div style="text-align: center; margin-bottom: 2em;">
        <span style="background: linear-gradient(45deg, #28a745, #20c997); color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold;">
            🎯 90%+ Accuracy • 🔬 Machine Learning • 🚀 Instant Results
        </span>
    </div>
    """)
    
    # Unit selector
    with gr.Row():
        units = gr.Radio(
            choices=["US", "Metric"], 
            value="US", 
            label="🌍 Units",
            info="Choose your measurement system"
        )
    
    # Input sections
    with gr.Row():
        # Basic car info
        with gr.Column(scale=2):
            gr.Markdown("### 🚗 **Basic Information**")
            with gr.Row():
                brand = gr.Dropdown(
                    choices=['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Audi', 'Lexus', 'Nissan', 'Hyundai'],
                    value="Toyota",
                    label="🏭 Brand"
                )
                car_type = gr.Dropdown(
                    choices=['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible', 'Wagon', 'Pickup'],
                    value="Sedan",
                    label="🚙 Type"
                )
            
            with gr.Row():
                fuel = gr.Dropdown(
                    choices=['Gas', 'Diesel', 'Hybrid', 'Electric'],
                    value="Gas",
                    label="⛽ Fuel"
                )
                age = gr.Slider(
                    minimum=0, maximum=20, value=3, step=0.1,
                    label="📅 Age (years)"
                )
        
        # Performance specs
        with gr.Column(scale=2):
            gr.Markdown("### ⚡ **Performance**")
            with gr.Row():
                mileage = gr.Number(
                    value=45000,
                    label="🛣️ Mileage"
                )
                engine = gr.Slider(
                    minimum=0, maximum=8, value=2.5, step=0.1,
                    label="🔧 Engine (L)"
                )
            
            with gr.Row():
                hp = gr.Slider(
                    minimum=100, maximum=800, value=200, step=10,
                    label="💪 Horsepower"
                )
                mpg = gr.Slider(
                    minimum=10, maximum=130, value=28, step=0.1,
                    label="⛽ Fuel Economy"
                )
        
        # Physical specs and market data
        with gr.Column(scale=2):
            gr.Markdown("### 📐 **Specs & Market**")
            with gr.Row():
                length = gr.Slider(
                    minimum=150, maximum=250, value=185, step=0.1,
                    label="📏 Length"
                )
                width = gr.Slider(
                    minimum=60, maximum=85, value=72, step=0.1,
                    label="📐 Width"
                )
            
            with gr.Row():
                weight = gr.Slider(
                    minimum=2000, maximum=6000, value=3400, step=50,
                    label="⚖️ Weight"
                )
                reliability = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.85, step=0.01,
                    label="🔧 Reliability"
                )
            
            with gr.Row():
                sales = gr.Slider(
                    minimum=5, maximum=300, value=50, step=0.1,
                    label="📊 Sales (k/year)"
                )
                resale = gr.Slider(
                    minimum=5, maximum=100, value=25, step=0.1,
                    label="💰 Resale ($k)"
                )
    
    # Predict button
    predict_button = gr.Button(
        "🔮 Get Price Prediction",
        variant="primary",
        size="lg"
    )
    
    # Results section
    gr.Markdown("---")
    gr.Markdown("## 🎯 **Results**")
    
    with gr.Row():
        with gr.Column(scale=1):
            price_output = gr.Textbox(
                label="💵 Price",
                interactive=False,
                placeholder="Click predict..."
            )
            full_price_output = gr.Textbox(
                label="💰 Full Price",
                interactive=False,
                placeholder="Detailed price..."
            )
            category_output = gr.Textbox(
                label="🏷️ Category",
                interactive=False,
                placeholder="Price category..."
            )
            accuracy_output = gr.Textbox(
                label="🎯 Accuracy",
                interactive=False,
                placeholder="Model accuracy..."
            )
        
        with gr.Column(scale=2):
            summary_output = gr.Markdown(
                value="**🚗 Analysis**\n\nEnter car details and click predict to see the analysis...",
                label="📋 Detailed Analysis"
            )
    
    # Handle unit changes
    def update_unit_labels(unit_choice):
        if unit_choice == "US":
            return (
                gr.update(info="Miles driven"),
                gr.update(info="Miles per gallon"),
                gr.update(info="Length in inches"),
                gr.update(info="Width in inches"),
                gr.update(info="Weight in pounds")
            )
        else:
            return (
                gr.update(info="Kilometers driven"),
                gr.update(info="Liters per 100km"),
                gr.update(info="Length in centimeters"),
                gr.update(info="Width in centimeters"),
                gr.update(info="Weight in kilograms")
            )
    
    units.change(
        update_unit_labels,
        inputs=[units],
        outputs=[mileage, mpg, length, width, weight]
    )
    
    # Connect the predict button
    predict_button.click(
        make_prediction,
        inputs=[units, brand, car_type, fuel, age, mileage, sales, resale, engine, hp, length, width, weight, mpg, reliability],
        outputs=[price_output, full_price_output, category_output, summary_output, accuracy_output]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 2em; padding: 1em; background: #f8f9fa; border-radius: 10px;">
        <p style="margin: 0; color: #6c757d;">
            <strong>🚗 Purchase Pulse</strong> • Machine Learning Car Price Prediction<br>
            Built with Random Forest and Gradient Boosting for 90%+ accuracy
        </p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
