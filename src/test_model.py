import os
#os.environ.setdefault("HF_HUB_OFFLINE", "1") # hf offline
os.environ["KERAS_BACKEND"] = "torch"
import keras
import utils

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


def single_prediction(subject: dict, item: dict) -> float:
    """Make a prediction on a single item"""
    subject, item = input
    model = _load_model()
    x = utils.get_sbert().encode([item['item_content']])
    prediction = model.predict(x)
    print("Predictions")
    print(prediction)
    return prediction[0][0]