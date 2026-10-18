import os
#os.environ.setdefault("HF_HUB_OFFLINE", "1") # hf offline
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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
RANDOM_SEED = 5814


def save_model(model, filename):
    """Save model from memory to file"""
    try:
        model.save(filename)
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {filename}: {e}")


def train_model() -> keras.Model:
    """Train model"""

    import datasets

    # TODO: Randomize samples of "responses"

    def _merge_row(example):
        """callback function that joins responses with their corresponding items"""
        key = example["item_id"]
        # Fetch matching data from the lookup table
        ref_data = items_lookup_dict.get(key)
        
        # Merge the dictionaries
        return {**example, **ref_data}

    splits = "matharena+mmdocrag+multi_swebench+real_webagents+researchcodebench+swe_rebench"

    print("Caching items by item_id")
    items_ds = datasets.load_dataset(
        "aims-foundations/measurement-db",
        name="items",
        split=splits
    )
    items_lookup_dict = {row["item_id"]: row for row in items_ds}
    print("Number of items " + str(len(items_lookup_dict)))

    print("Setting up responses stream")
    responses_stream = datasets.load_dataset(
        "aims-foundations/measurement-db",
        name="responses",
        split=splits,
        streaming=True
    )

    joined_stream = responses_stream.map(_merge_row)

    sentences = []
    responses = []

    # joined_stream is an iterator, so we can keep consuming it
    for row in joined_stream:
        # The merged row contains the item text and the response label
        sentences.append(row["content"])
        responses.append(float(row["response"]))

        # Optional: stop early during debugging
        if len(sentences) >= 500:
            break

    sbert = utils.get_sbert()
    embeddings = sbert.encode(
        sentences,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)
    #print(embeddings.shape)

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
        X_train, y_train, test_size=0.20, random_state=RANDOM_SEED, stratify=y_train
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
