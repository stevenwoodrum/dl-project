import os
#os.environ.setdefault("HF_HUB_OFFLINE", "1") # hf offline
os.environ["KERAS_BACKEND"] = "torch"
import keras
import utils
import numpy as np

MODEL_FILE = "model.keras"
model = None


def _load_model():
    """Load model from file to memory"""
    global model
    if model is None:
        try:
            model = keras.saving.load_model(MODEL_FILE)
        except Exception as e:
            print(os.getcwd())
            raise RuntimeError(f"Failed to load model from {MODEL_FILE}: {e}")
    return model


def single_prediction(input: list, labeled: list | None = None) -> float:
    """Make a prediction on a single item"""
    subject, item = input
    x1 = utils.get_sbert().encode([item['item_content']])
    x2 = utils.get_sbert().encode([subject['normalized_name']])
    x = np.concatenate((x1, x2), axis=1)
    model = _load_model()
    prediction = model.predict(x)
    print("Predictions")
    print(prediction)
    return prediction[0][0]