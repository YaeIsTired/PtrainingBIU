import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURES_KEY = 'features'
LABEL_KEY = 'label'


def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = LogisticRegression(max_iter=100)
    return model


def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def predict_recommendation(features, strategy, model_path):
    model = load_model(model_path)
    model.__dict__['last_features'] = features
    save_model(model, model_path)
    if hasattr(model, 'classes_'):
        prediction = model.predict([features])[0]
        return prediction
    else:
        # default recommendation when model not trained
        return 'buy' if strategy == 'scalping' else 'hold'


def update_feedback(recommendation, correct, model_path):
    model = load_model(model_path)
    features = model.__dict__.get('last_features')
    if features is None:
        return
    label = 1 if correct else 0
    X = model.__dict__.get(FEATURES_KEY)
    y = model.__dict__.get(LABEL_KEY)
    if X is None:
        X = np.array([features])
        y = np.array([label])
    else:
        X = np.vstack([X, features])
        y = np.hstack([y, label])
    model.fit(X, y)
    model.__dict__[FEATURES_KEY] = X
    model.__dict__[LABEL_KEY] = y
    save_model(model, model_path)
