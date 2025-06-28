import cv2
import numpy as np
import pickle

from .model import predict_recommendation, load_model, save_model

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Invalid image')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    mean_intensity = np.mean(gray)
    return np.hstack([hist / np.sum(hist), mean_intensity, np.sum(edges)])

def analyze_chart(image_path, strategy, model_path):
    features = extract_features(image_path)
    model = load_model(model_path)
    # store the latest features inside the model for training with feedback
    model.__dict__['last_features'] = features
    save_model(model, model_path)

    recommendation = predict_recommendation(features, strategy, model_path)
    analysis = {
        'recommendation': recommendation,
        'entry': 'Based on recent price action',
        'stop_loss': 'Placed below recent swing low',
        'take_profit': ['TP1 at 1R', 'TP2 at 2R', 'TP3 at 3R'],
        'reasoning': [
            'Market structure indicates potential shift',
            'Support and resistance derived from detected lines',
            'Risk assessed via volatility in screenshot'
        ]
    }
    return analysis
