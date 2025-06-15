import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_cols = None
        self.original_columns = None
        self.is_trained = False

    def generate_data(self, n=2000):
        """Generate sophisticated synthetic training data with ALL features"""
        np.random.seed(42)
        
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
        
        types = ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible', 'Wagon', 'Pickup']
        fuels = ['Gas', 'Diesel', 'Hybrid', 'Electric']
        
        data = []
        for i in range(n):
            brand = np.random.choice(list(brands.keys()))
            brand_info = brands[brand]
            vtype = np.random.choice(types)
            fuel = np.random.choice(fuels, p=[0.6, 0.25, 0.1, 0.05])
            
            age = np.random.exponential(3)
            age = min(age, 20)
            
            mileage = age * np.random.uniform(8000, 15000) + np.random.normal(0, 5000)
            mileage = max(0, mileage)
            
            if fuel == 'Electric':
                engine = 0.0
                hp = np.random.uniform(150, 400)
            elif fuel == 'Hybrid':
                engine = np.random.uniform(1.5, 2.5)
                hp = engine * np.random.uniform(60, 90) + np.random.uniform(50, 100)
            else:
                engine = np.random.uniform(1.2, 6.5)
                if brand_info['luxury']:
                    engine += np.random.uniform(0.5, 2.0)
                hp = engine * np.random.uniform(45, 85) + np.random.normal(0, 15)
            
            hp = np.clip(hp, 100, 600)
            
            if vtype == 'SUV':
                length, width, weight = 195, 75, 4200
            elif vtype == 'Pickup':
                length, width, weight = 215, 79, 4800
            elif vtype == 'Coupe':
                length, width, weight = 180, 71, 3200
            else:
                length, width, weight = 185, 72, 3400
            
            length += np.random.uniform(-10, 10)
            width += np.random.uniform(-3, 3)
            weight += np.random.uniform(-400, 400)
            
            if fuel == 'Electric':
                mpg = np.random.uniform(100, 130)
            elif fuel == 'Hybrid':
                mpg = np.random.uniform(40, 55)
            else:
                mpg = 32 - (engine * 2.5) - (weight / 350) + np.random.normal(0, 3)
                mpg = np.clip(mpg, 12, 130)
            
            pwr_weight = hp / (weight / 1000)
            
            base_price = np.random.uniform(*brand_info['price'])
            
            if brand_info['luxury']:
                base_price *= 1.3
            
            fuel_mult = {'Electric': 1.2, 'Hybrid': 1.1, 'Diesel': 1.05, 'Gas': 1.0}[fuel]
            base_price *= fuel_mult
            
            type_mult = {'Convertible': 1.2, 'Coupe': 1.1, 'SUV': 1.05, 'Pickup': 1.03,
                        'Sedan': 1.0, 'Wagon': 0.98, 'Hatchback': 0.95}[vtype]
            base_price *= type_mult
            
            depreciation = 0.85 ** age
            base_price *= depreciation
            
            mileage_factor = max(0.3, 1 - (mileage / 200000) * 0.4)
            base_price *= mileage_factor
            
            base_price *= (1 + (pwr_weight - 100) / 1000)
            base_price *= (1 + brand_info['reliability'] - 0.8)
            base_price += np.random.normal(0, 3)
            base_price = np.clip(base_price, 8, 250)
            
            sales = np.random.lognormal(3.5, 0.8)
            sales = np.clip(sales, 5, 300)
            
            resale = base_price * np.random.uniform(0.35, 0.65) * brand_info['reliability']
            
            data.append({
                'brand': brand, 'type': vtype, 'fuel': fuel, 'age': round(age, 1),
                'mileage': int(mileage), 'sales': round(sales, 1), 'resale': round(resale, 2),
                'engine': round(engine, 1), 'hp': int(hp), 'length': round(length, 1),
                'width': round(width, 1), 'weight': int(weight), 'mpg': round(mpg, 1),
                'pwr_weight': round(pwr_weight, 2), 'reliability': brand_info['reliability'],
                'price': round(base_price, 2)
            })
        
        return pd.DataFrame(data)

    def train(self):
        print("Generating training dataset...")
        df = self.generate_data()
        X = df.drop('price', axis=1)
        y = df['price']
        
        self.original_columns = X.columns.tolist()
        print(f"Training with features: {self.original_columns}")
        
        categorical_columns = ['brand', 'type', 'fuel']
        for col in categorical_columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])
            self.encoders[col] = encoder
        
        feature_selector = SelectKBest(score_func=f_regression, k=12)
        X_selected = feature_selector.fit_transform(X, y)
        self.feature_selector = feature_selector
        self.feature_cols = X.columns[feature_selector.get_support()].tolist()
        
        print(f"Selected features: {self.feature_cols}")
        
        X_scaled = self.scaler.fit_transform(X_selected)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
        
        print("Training ensemble models with hyperparameter optimization...")
        
        rf_parameters = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
        rf_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_parameters, cv=5, scoring='r2')
        rf_search.fit(X_train, y_train)
        
        gb_parameters = {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1],
            'max_depth': [6, 8]
        }
        gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_parameters, cv=5, scoring='r2')
        gb_search.fit(X_train, y_train)
        
        models = {'RandomForest': rf_search.best_estimator_, 'GradientBoosting': gb_search.best_estimator_}
        best_score = 0
        
        for name, model in models.items():
            score = model.score(X_test, y_test)
            print(f"{name}: {score:.4f}")
            if score > best_score:
                best_score = score
                self.model = model
        
        self.is_trained = True
        print(f"Best accuracy: {best_score:.4f} ({best_score*100:.1f}%)")
        return best_score

    def predict(self, specs):
        if not self.is_trained:
            raise Exception("Model not trained")
        
        input_specs = specs.copy()
        
        for col, encoder in self.encoders.items():
            if col in input_specs:
                try:
                    input_specs[col] = encoder.transform([input_specs[col]])[0]
                except:
                    input_specs[col] = 0
        
        df = pd.DataFrame([input_specs])
        
        for col in self.original_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.original_columns]
        df_selected = self.feature_selector.transform(df)
        df_scaled = self.scaler.transform(df_selected)
        
        return self.model.predict(df_scaled)[0]

# Initialize and train model
print("🚗 Purchase Pulse - Car Price Prediction System")
print("Initializing advanced ensemble ML model...")
predictor = CarPricePredictor()
accuracy = predictor.train()

# ONLY ADDITION: Validation function
def check_for_weird_stuff(brand, vtype, fuel, age, mileage, sales, resale, engine, hp, length, width, weight, mpg, reliability):
    """Check for obviously unrealistic inputs with funny messages"""
    errors = []
    
    # Gas/Diesel cars MUST have engines
    if fuel in ['Gas', 'Diesel'] and engine == 0:
        errors.append("Gas car with no engine? That's like a pizza with no cheese! Add some engine size!")
    
    # Electric cars can't have engines
    if fuel == 'Electric' and engine > 0:
        errors.append("Electric car with an engine? That's like putting a sail on a Tesla!")
    
    # Hybrid cars need engines too
    if fuel == 'Hybrid' and engine == 0:
        errors.append("Hybrid with no engine? Even hybrids need some gas power, buddy!")
    
    # Age vs mileage reality check
    if age > 0 and mileage / age > 50000:
        errors.append(f"🚗 Whoa! {mileage:,.0f} miles on a {age:.1f}-year-old car? That's {mileage/age:,.0f} miles/year! Was this an Uber? 😅")
    
    # Brand new car but high mileage
    if age == 0 and mileage > 100:
        errors.append(f"🎉 Brand new car with {mileage} miles? Someone took a really long test drive! 🏁")
    
    # Horsepower extremes
    if hp > 800:
        errors.append(f"🚀 {hp} HP? Calm down, Dom Toretto! This ain't Fast & Furious! 💨")
    elif hp < 50:
        errors.append(f"🐌 {hp} HP? My grandma's wheelchair has more power! 👵")
    
    # Weight extremes
    if weight < 1500:
        errors.append(f"⚖️ {weight} lbs? That's lighter than a golf cart! You sure this isn't a bicycle? 🚲")
    elif weight > 8000:
        errors.append(f"🐘 {weight} lbs? Bro, this ain't a tank! Even Hummers don't weigh that much! 🛻")
    
    return errors

def predict_car_price(brand, vtype, fuel, age, mileage, sales, resale, engine, hp, length, width, weight, mpg, reliability):
    try:
        # ONLY NEW PART: Check for weird inputs first
        weird_stuff = check_for_weird_stuff(brand, vtype, fuel, age, mileage, sales, resale, engine, hp, length, width, weight, mpg, reliability)
        
        if weird_stuff:
            error_msg = "## 🚨 Hold up! Something seems off here...\n\n"
            error_msg += "\n".join([f"• {err}" for err in weird_stuff])
            error_msg += "\n\n**Help:** Adjust the values to be more realistic and try again! 😊"
            return "🤔 Hmm...", "Check inputs", "Validation Failed", error_msg
        
        # Calculate power-to-weight ratio
        pwr_weight = hp / (weight / 1000)
        
        # Create complete feature vector with ALL required features
        specs = {
            'brand': brand, 'type': vtype, 'fuel': fuel, 'age': age,
            'mileage': mileage, 'sales': sales, 'resale': resale,
            'engine': engine, 'hp': hp, 'length': length, 'width': width,
            'weight': weight, 'mpg': mpg, 'pwr_weight': pwr_weight,
            'reliability': reliability
        }
        
        # Get prediction
        price = predictor.predict(specs)
        
        if price < 20:
            category = "Budget (< $20k)"
        elif price < 35:
            category = "Mid-Range ($20k-$35k)"
        elif price < 60:
            category = "Premium ($35k-$60k)"
        else:
            category = "Luxury (> $60k)"
        
        price_formatted = f"${price:.2f}k"
        price_full = f"${price*1000:,.0f}"
        is_luxury = brand in ['BMW', 'Mercedes', 'Audi', 'Lexus']
        
        summary = f"""**Prediction Results:**
- **Price:** {price_formatted} ({price_full})
- **Category:** {category}
- **Luxury Brand:** {'Yes' if is_luxury else 'No'}
- **Model Accuracy:** {accuracy*100:.1f}%

**Vehicle Analysis:**
- {brand} {vtype} ({fuel})
- Age: {age:.1f} years, Mileage: {mileage:,.0f}
- Engine: {engine:.1f}L, Power: {int(hp)} HP
- Efficiency: {mpg:.1f} MPG
- Reliability: {reliability:.2f}/1.0
- Power-to-Weight: {pwr_weight:.1f} HP/1000lbs
- Market Data: {sales:.1f}k sales, ${resale:.1f}k resale"""
        
        return price_formatted, price_full, category, summary
        
    except Exception as e:
        error_msg = f"**Error:** {str(e)}\n\nPlease check input values and try again."
        return "Error", "N/A", "Unknown", error_msg

# YOUR ORIGINAL UI - COMPLETELY UNCHANGED
with gr.Blocks(title="Purchase Pulse - Car Price Predictor") as demo:
    
    gr.Markdown("""
    # 🚗 Purchase Pulse - Car Price Predictor
    
    **Advanced Machine Learning System for Accurate Vehicle Valuation**
    
    This system uses ensemble methods (Random Forest + Gradient Boosting) with sophisticated feature engineering to predict car prices with 90%+ accuracy.
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Vehicle Information")
            brand = gr.Dropdown(
                choices=['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Audi', 'Lexus', 'Nissan', 'Hyundai'],
                value="Toyota", label="Brand"
            )
            vtype = gr.Dropdown(
                choices=['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible', 'Wagon', 'Pickup'],
                value="Sedan", label="Vehicle Type"
            )
            fuel = gr.Dropdown(
                choices=['Gas', 'Diesel', 'Hybrid', 'Electric'],
                value="Gas", label="Fuel Type"
            )
            age = gr.Slider(0, 20, value=3, step=0.1, label="Age (years)")
            mileage = gr.Number(value=45000, label="Mileage (miles)")
        
        with gr.Column():
            gr.Markdown("### Performance Specifications")
            engine = gr.Slider(0, 8, value=2.5, step=0.1, label="Engine Size (L)")
            hp = gr.Slider(100, 800, value=200, step=10, label="Horsepower")
            mpg = gr.Slider(10, 130, value=28, step=0.1, label="Fuel Economy (MPG)")
            length = gr.Slider(150, 250, value=185, step=0.1, label="Length (inches)")
            width = gr.Slider(60, 85, value=72, step=0.1, label="Width (inches)")
        
        with gr.Column():
            gr.Markdown("### Market & Physical Data")
            weight = gr.Slider(2000, 6000, value=3400, step=50, label="Weight (lbs)")
            reliability = gr.Slider(0.5, 1.0, value=0.85, step=0.01, label="Reliability Score")
            sales = gr.Slider(5, 300, value=50, step=0.1, label="Annual Sales (thousands)")
            resale = gr.Slider(5, 100, value=25, step=0.1, label="Resale Value ($k)")
    
    predict_btn = gr.Button("Predict Car Price", variant="primary", size="lg")
    
    gr.Markdown("### Prediction Results")
    
    with gr.Row():
        with gr.Column(scale=1):
            price_output = gr.Textbox(label="Predicted Price", interactive=False)
            price_full_output = gr.Textbox(label="Full Price", interactive=False)
            category_output = gr.Textbox(label="Market Category", interactive=False)
        
        with gr.Column(scale=2):
            analysis_output = gr.Markdown(label="Detailed Analysis")
    
    predict_btn.click(
        predict_car_price,
        inputs=[brand, vtype, fuel, age, mileage, sales, resale, engine, hp, length, width, weight, mpg, reliability],
        outputs=[price_output, price_full_output, category_output, analysis_output]
    )
    
    gr.Markdown("""
    ---
    **Technical Details:** Ensemble ML • 2,000 synthetic samples • 15+ features • 90%+ accuracy • GridSearchCV optimization
    """)

if __name__ == "__main__":
    demo.launch()
