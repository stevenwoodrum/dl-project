import os
#os.environ.setdefault("HF_HUB_OFFLINE", "1") # hf offline
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from datasets import load_dataset
import utils


# TODO:
# Add subject ID (condition prediction on subject model)
# Do a "leave one benchmark out" for 
# Chkpt one: strengths/limitations of leave-one-benchmark-out approach.
# Mess with activation functions, hyperparameters, optimizers etc, to add 
# content to checkpoint one report
# check Discord etc for recent reporting on application of IRT
    # - make sure we're leveraging, not plagiarizing
    # - make sure we're accounting for the latest and greatest
    # - determine if we need to take a different approach


MODEL_FILE = "model.keras"


def save_model(model, filename):
    """Save model from memory to file"""
    try:
        model.save(filename)
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {filename}: {e}")


def train_model() -> keras.Model:
    """Minimal training function"""

    # Load the unified Parquet dataset
    """df = pd.read_parquet(
        "hf://datasets/aims-foundations/measurement-db@refs/convert/parquet"
    )"""

    ds = load_dataset(
        "aims-foundations/measurement-db",
        split="train",
        streaming=True
    )

    # Get first N rows without downloading the whole dataset
    N = 5
    sample = [next(iter(ds)) for _ in range(N)]
    print(sample)

    df = pd.DataFrame(sample)

    # Filter rows belonging to MathArena
    items = df[df["benchmark"] == "matharena"]

    # If responses are stored in the same dataset, filter them too
    responses = df[df["benchmark"] == "matharena"]
    #print(responses.head)

    merged = pd.merge(items, responses, how="left", on="item_id")
    print(merged.shape)
    merged = merged.head(500) # Experiment with only 50 samples of the MathArena responses initially

    sentences = np.array(merged['content'])
    responses = np.array(merged['response'])

    embeddings = utils.get_sbert().encode(sentences, show_progress_bar=True).astype(np.float32)
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

    save_model(model, MODEL_FILE)

    return model


def visualize_training(history: keras.callbacks.History):
    """Graph results of training/validation"""
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
