import os
#os.environ.setdefault("HF_HUB_OFFLINE", "1") # hf offline
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#TODO:
# Add subject ID (condition prediction on subject model)
# Do a "leave one benchmark out" for 
# Chkpt one: strengths/limitations of leave-one-benchmark-out approach.
# Mess with activation functions, hyperparameters, optimizers etc, to add 
# content to checkpoint one report
# check Discord etc for recent reporting on application of IRT
    # - make sure we're leveraging, not plagiarizing
    # - make sure we're accounting for the latest and greatest
    # - determine if we need to take a different approach

# -----------------------------
# Global state
# -----------------------------
model = None
sbert = None
model_file = "./model.keras"

# -----------------------------
# Load SBERT
# -----------------------------
def load_sbert():
    global sbert
    if sbert is None:
        sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return sbert

# -----------------------------
# Load model from file to memory
# -----------------------------
def load_model():
    global model, model_file
    if model is None:
        try:
            model = keras.saving.load_model(model_file)
        except Exception as e:
            print(os.getcwd())
            raise RuntimeError(f"Failed to load model from {model_file}: {e}")
    return model

# -----------------------------
# Save model in memory to file
# -----------------------------
def save_model():
    global model, model_file
    try:
        model.save(model_file)
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {model_file}: {e}")

# -----------------------------
# Minimal training function
# -----------------------------
def train_model() -> keras.Model:
    global model
    print("Fetching data from HuggingFace...")
    splits = {'matharena': 'matharena/items.parquet', 'mmdocrag': 'mmdocrag/items.parquet', 'multi_swebench': 'multi_swebench/items.parquet', 'real_webagents': 'real_webagents/items.parquet', 'researchcodebench': 'researchcodebench/items.parquet', 'swe_rebench': 'swe_rebench/items.parquet'}
    items = pd.read_parquet("hf://datasets/aims-foundations/measurement-db/" + splits["matharena"])
    #print(items.head)
    splits = {'matharena': 'matharena/response.parquet', 'mmdocrag': 'mmdocrag/response.parquet', 'multi_swebench': 'multi_swebench/response.parquet', 'real_webagents': 'real_webagents/response.parquet', 'researchcodebench': 'researchcodebench/response.parquet', 'swe_rebench': 'swe_rebench/response.parquet'}
    responses = pd.read_parquet("hf://datasets/aims-foundations/measurement-db/" + splits["matharena"])
    #print(responses.head)

    merged = pd.merge(items, responses, how="left", on="item_id")
    print(merged.shape)
    merged = merged.head(500) # Experiment with only 50 samples of the MathArena responses initially

    sentences = np.array(merged['content'])
    responses = np.array(merged['response'])

    embeddings = load_sbert().encode(sentences, show_progress_bar=True).astype(np.float32)
    print(embeddings.shape)

    print("Building FNN with Keras...")
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(embeddings.shape[1],)), # input layer
            keras.layers.Dense(2, activation="relu"), # hidden layer
            keras.layers.Dense(1, activation="sigmoid") # output layer
        ]
    )

    model.compile(
        loss="mse", # mean squared error equivalent to Brier score for binary events
    )

    model.summary()

    # Embed all training texts
    X_train = np.vstack(embeddings)
    y_train = np.array(responses, dtype=np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=5814, stratify=y_train
    )

    history = model.fit(
        X_tr, 
        y_tr, 
        epochs=15,
        validation_data=(X_val, y_val)
    )

    visualize_training(history)

    save_model()

    return model

# -----------------------------
# Graph results of training/validation
# -----------------------------
def visualize_training(history: keras.callbacks.History):
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    plt.figure(figsize=(6, 5))

    # Plot Loss
    plt.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss')
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Set ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    plt.tight_layout()
    plt.show(block=False)

# -----------------------------
# Prediction function
# -----------------------------
def do_predict(subject: dict, item: dict) -> float:
    model = load_model()
    x = load_sbert().encode([item['item_content']])
    prediction = model.predict(x)
    print("Predictions")
    print(prediction)
    return prediction[0][0]
