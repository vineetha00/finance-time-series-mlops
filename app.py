import gradio as gr
import joblib

# Load model
model = joblib.load("model.joblib")

# Predict function
def predict_price(last_close):
    pred = model.predict([[last_close]])
    return f"Predicted next close: ${round(float(pred[0]), 2)}"

# Gradio UI
interface = gr.Interface(
    fn=predict_price,
    inputs=gr.Number(label="Last Close Price"),
    outputs="text",
    title="📈 Stock Price Predictor",
    description="Enter the last stock close price to get a next-day prediction.",
)

if __name__ == "__main__":
    interface.launch()
