"""Your submission must define a single function:

    predict(input: list, labeled: list | None = None) -> float

The ingestion program calls predict() once per distinct [subject, item] input.
Module-level code runs once when the container starts. Load weights,
tokenizers, prompt templates here. Heavy training must be done OFFLINE
(e.g. publish a model to HuggingFace and load it at module init).

`input` shape
-------------
A list of exactly two dicts: ``[subject, item]``. Every value is a string;
missing fields are "".

    subject = {
        "normalized_name":        str,  # subject (model) display name
        "provider":               str,  # e.g. "openai"
        "release_date":           str,
        "access_date":            str,
        "harness":                str,  # evaluation harness used
        "reasoning_effort":       str,
        "harness_version":        str,
        "subject_features_extra": str,  # free-form extra metadata
    }

    item = {
        "item_content":  str,  # the question / prompt / task text
        "item_features": str,  # free-form item feature string; may be ""
        "interactors":   str,  # other agents/tools in the loop; may be ""
        "benchmark_id":  str,  # stable anonymous alias, also in acquired items
    }

The same item may have multiple independent outcomes for a subject. Identical
inputs share one predicted probability; every outcome still contributes to scoring.

`labeled` (optional)
--------------------
A list of ``[[subject, item], label]`` entries: each holds an input-shaped
pair plus its ground-truth label (1 = subject answered correctly, 0 =
incorrectly). These are revealed via adaptive labeling (see labeling.py).
May be None or empty.

Return value
------------
A single float in [0, 1], the predicted probability that the subject
answers the item correctly.
"""


from __future__ import annotations

# ---------------------------------------------------------------------------
# Module-level init: runs once when the container starts.
# Replace this with model loading, tokenizer setup, prompt templates, etc.
# ---------------------------------------------------------------------------
import sandbox
sandbox.load_model()


def predict(input: list, labeled: list | None = None) -> float:
    """Return the predicted probability that the subject answers correctly."""
    subject, item = input
    return sandbox.do_predict(subject, item)
