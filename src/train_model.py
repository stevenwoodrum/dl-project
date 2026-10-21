import os
#os.environ.setdefault("HF_HUB_OFFLINE", "1") # hf offline
import numpy as np
import datasets
os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import utils



# TODO:
# Do a "leave one benchmark out" for 
#   Chkpt one: strengths/limitations of leave-one-benchmark-out approach.
# Mess with activation functions, hyperparameters, optimizers etc, to add 
# Add content to checkpoint one report
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
    """Train model using reservoir-sampled responses and efficient SBERT batching."""

    import random
    random.seed(RANDOM_SEED)

    benchmarks = (
        "multi_swebench+real_webagents+"
        "researchcodebench+swe_rebench"
    )

    SAMPLE_SIZE = 10000

    # ---------------------------------------------------------
    # Load lookup tables (items + subjects)
    # ---------------------------------------------------------
    print("Caching items by item_id")
    items_ds = datasets.load_dataset(
        "aims-foundations/measurement-db",
        name="items",
        split=benchmarks
    )
    items_lookup = {row["item_id"]: row for row in items_ds}
    print(f"Items cached: {len(items_lookup)}")

    print("Caching subjects by subject_id")
    subjects_ds = datasets.load_dataset(
        "aims-foundations/measurement-db",
        name="subjects",
        split=benchmarks
    )
    subjects_lookup = {row["subject_id"]: row for row in subjects_ds}
    print(f"Subjects cached: {len(subjects_lookup)}")

    # ---------------------------------------------------------
    # Streaming responses + reservoir sampling
    # ---------------------------------------------------------
    print("Streaming responses and performing reservoir sampling...")
    responses_stream = datasets.load_dataset(
        "aims-foundations/measurement-db",
        name="responses",
        split=benchmarks,
        streaming=True
    )

    def _merge_row(example):
        item = items_lookup.get(example["item_id"])
        subject = subjects_lookup.get(example["subject_id"])
        if item is None or subject is None:
            return None
        return {**example, **item, **subject}

    joined_stream = responses_stream.map(_merge_row)

    reservoir = []
    n_seen = 0

    for row in joined_stream:
        if row is None:
            continue

        # Validate fields early
        content = row.get("content")
        subject_name = row.get("normalized_name")
        response_val = row.get("response")

        if not isinstance(content, str):
            continue
        if not isinstance(subject_name, str):
            continue
        if response_val is None:
            continue

        # Reservoir sampling
        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(row)
        else:
            j = random.randint(0, n_seen)
            if j < SAMPLE_SIZE:
                reservoir[j] = row

        n_seen += 1

    print(f"Reservoir sampled {len(reservoir)} rows from ~{n_seen} streamed rows")

    # ---------------------------------------------------------
    # Extract text + labels
    # ---------------------------------------------------------
    item_texts = [r["content"] for r in reservoir]
    subject_texts = [r["normalized_name"] for r in reservoir]
    labels = np.array([float(r["response"]) for r in reservoir], dtype=np.float32)

    # ---------------------------------------------------------
    # SBERT embedding (batched)
    # ---------------------------------------------------------
    print("Embedding item and subject text with SBERT...")
    sbert = utils.get_sbert()

    item_embeddings = sbert.encode(
        item_texts,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    subject_embeddings = sbert.encode(
        subject_texts,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    # Concatenate embeddings
    X = np.concatenate([item_embeddings, subject_embeddings], axis=1)

    # ---------------------------------------------------------
    # LOBO (Leave-One-Benchmark-Out) training and evaluation
    # ---------------------------------------------------------
    benchmark_ids = np.array([r["benchmark_id"] for r in reservoir])
    logo = LeaveOneGroupOut()
    oof_predictions = np.empty(len(labels), dtype=np.float32)
    benchmark_scores = {}
    histories = list()

    for train_idx, test_idx in logo.split(X, labels, groups=benchmark_ids):
        held_out = benchmark_ids[test_idx][0]        

        # Separate training and validation data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X[train_idx], labels[train_idx],
            test_size=0.20,
            random_state=RANDOM_SEED,
        )

        # Create a fresh model for each fold
        fold_model = build_model(X.shape[1])
        # Train/validate
        history = fold_model.fit(
            X_tr,
            y_tr,
            epochs=15,
            verbose=0,
            validation_data=(X_val, y_val)
        )
        histories.append(history)

        # Test on held-out benchmark
        predictions = fold_model.predict(X[test_idx], verbose=0).reshape(-1)
        oof_predictions[test_idx] = predictions
        benchmark_scores[held_out] = mean_squared_error(
            labels[test_idx],
            predictions,
        )

    print("Overall held-out MSE:", mean_squared_error(labels, oof_predictions))
    print("MSE by benchmark:", benchmark_scores)

    # Graph results
    """ TODO: Make title of each graph reflect WHICH benchmark is being held out.
    It is okay to leave anonymous for now, just means we are testing 
    for overall generalization without observing WHICH held-out benchmarks
    test poorer."""
    visualize_training('Training and Validation Loss when holding out each benchmark',
                       histories)

    # ---------------------------------------------------------
    # Retrain final model with all benchmarks and save
    # ---------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, labels,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )
    final_model = build_model(X.shape[1])
    history = final_model.fit(
        X_tr,
        y_tr,
        epochs=15,
        verbose=0,
        validation_data=(X_val, y_val)
    )
    visualize_training("Final model with no benchmarks held out",
                       [history])
    save_model(final_model, MODEL_FILE)

    return final_model


def build_model(input_length):
    """Assemble the model to undergo training"""
    print("Building FNN with Keras...")
    model = keras.Sequential([
        keras.layers.Input(shape=(input_length,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="mse")
    return model


def visualize_training(title: str, histories: list[keras.callbacks.History]):
    """Plot each training history in its own subplot."""
    if not histories:
        raise ValueError("At least one training history is required.")

    ncols = int(np.ceil(np.sqrt(len(histories))))
    nrows = int(np.ceil(len(histories) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(title)

    for index, (ax, history) in enumerate(zip(axes.flat, histories)):
        history_dict = history.history
        epochs = range(1, len(history_dict["loss"]) + 1)

        ax.plot(epochs, history_dict["loss"], "bo-", label="Training Loss")
        if "val_loss" in history_dict:
            ax.plot(epochs, history_dict["val_loss"], "ro-", label="Validation Loss")
        ax.set_title(f"History {index + 1}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))

    for ax in axes.flat[len(histories):]:
        ax.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show(block=False)
