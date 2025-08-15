import gradio as gr
import yfinance as yf
import numpy as np
import joblib
from datetime import date, timedelta

# --- Load the saved XGBoost model and scaler ---
try:
    model = joblib.load('stock_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("❌ Error: Model or scaler not found. Please run train.py first.")
    exit()

# --- The function that makes predictions ---
def predict_price_movement(stock_symbol):
    """
    Fetches the latest stock data and predicts if the next day's price will go up or down.
    """
    if not stock_symbol:
        return "Please enter a stock symbol.", ""

    try:
        # Get the latest trading day's data
        today = date.today()
        start_date = today - timedelta(days=5)
        stock_data = yf.download(stock_symbol, start=start_date, end=today, progress=False)
        
        if stock_data.empty:
            return f"Could not fetch recent data for '{stock_symbol}'.", ""
            
        latest_data = stock_data.iloc[-1]

        # --- Create the features for the model ---
        open_close = latest_data['Open'] - latest_data['Close']
        low_high = latest_data['Low'] - latest_data['High']
        latest_month = latest_data.name.month
        is_quarter_end = 1 if latest_month % 3 == 0 else 0
        
        # ** THE FIX IS HERE: Convert all features to float **
        features = np.array([[
            float(open_close),
            float(low_high),
            float(is_quarter_end)
        ]])
        
        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features)
        
        # --- Make a prediction ---
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0]

        # --- Display the result ---
        if prediction[0] == 1:
            result = "Prediction: **Price will likely GO UP** 📈"
        else:
            result = "Prediction: **Price will likely GO DOWN** 📉"
            
        confidence = f"Confidence: {max(probability)*100:.2f}%"
        
        return result, confidence

    except Exception as e:
        return f"An error occurred: {e}", ""

# --- Create and launch the Gradio Interface ---
iface = gr.Interface(
    fn=predict_price_movement,
    inputs=gr.Textbox(label="Enter Stock Symbol", placeholder="e.g., TSLA, AAPL"),
    outputs=[gr.Markdown(label="Result"), gr.Label(label="Model Confidence")],
    title="🚀 Stock Price Movement Predictor (XGBoost)",
    description="Predict if a stock's price will rise or fall the next trading day. Based on the model from the provided PDF.",
    examples=[["TSLA"], ["AAPL"], ["NVDA"]]
)

if __name__ == "__main__":
    iface.launch()