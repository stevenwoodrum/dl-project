"""

Approach 1: SBERT item embedding + normalized_name embedding -> Dense(64, ReLU) -> sigmoid.

No TTA. 

NN forward pass done in plain numpy, so Keras isn't needed at submission time
"""
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1") # hf offline
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent

# load everything once
_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
with np.load(HERE / "weights.npz") as _w:
    _NAME_TABLE, _W1, _B1, _W2, _B2 = [_w[k] for k in ["name_table", "W1", "b1", "W2", "b2"]]
with open(HERE / "vocab.json") as f:
    _VOCAB = json.load(f)



# turns an items text into its 384 numbers
# the cache is a dictionary from text to vector, so the the same text isnt reembedded multiple times
_item_cache = {}  


def _item_vec(text):
    if text not in _item_cache:
        _item_cache[text] = _SBERT.encode([text], convert_to_numpy=True)[0]
    return _item_cache[text]

# name to row number, cleans and looks the name up. If 0 is returned from get the trained unknown row is used (missing values or models not seen in training)
def _name_idx(name):
    return _VOCAB.get(name.lower().strip(), 0)

# function called by Codabench
# input is a list of two dictionaries the subject and the item. labeled is not used (no TTA in approach 1)
def predict(input, labeled=None):
    subject, item = input

    # gets items 384 numbers and names 16 numbers
    item_vec = _item_vec(item.get("item_content", "") or "")
    name_vec = _NAME_TABLE[_name_idx(subject.get("normalized_name", "") or "")]

    x = np.concatenate([item_vec, name_vec]) # 384 + 16 numbers
    h = np.maximum(x @ _W1 + _B1, 0.0)  # hidden layer
    z = float((h @ _W2 + _B2)[0])  # output before sigmoid
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))) # sigmoid

    # codabench expects a float between 0 and 1, clip guarentees that range 
    return float(np.clip(p, 0.0, 1.0))


# model.py
# def predict(input: list[dict], labeled: list) -> float:
#     return float(0.5)
