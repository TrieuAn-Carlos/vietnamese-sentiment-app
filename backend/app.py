from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI()

# CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model_path = "../phobert_sentiment_model_final"
if not os.path.exists(model_path):
    print("Model not found locally, downloading from Hugging Face...")
    # For Railway deployment, download model from Hugging Face
    model_name = "your-username/your-model-name"  # Replace with your Hugging Face model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    #  Prediction logic here
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
    
    labels = ["Negative", "Neutral", "Positive"]
    result = labels[predicted_class.item()]
    confidence = probabilities[0][predicted_class.item()].item()
    
    return {
        "sentiment": result,
        "confidence": round(confidence * 100, 1),
        "probabilities": {
            "negative": round(probabilities[0][0].item() * 100, 1),
            "neutral": round(probabilities[0][1].item() * 100, 1),
            "positive": round(probabilities[0][2].item() * 100, 1)
        }
    }

@app.get("/")
def read_root():
    return {"message": "Vietnamese Sentiment Analysis API"}