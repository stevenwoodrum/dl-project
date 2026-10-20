"""Check and package the saved combined FNN without retraining it."""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from .training import HERE, SUBMISSION_DIR, file_sha256, save_json
from .submission.features import network_logits, sigmoid


def restore_encoder():
    """The encoder is stored once in Git, inside the ready-to-upload ZIP."""
    if (SUBMISSION_DIR / "text_encoder/modules.json").is_file():
        return
    archive_path = HERE / "submission_fnn.zip"
    if not archive_path.is_file():
        raise FileNotFoundError("The local text_encoder folder and submission_fnn.zip are both missing.")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.filename.startswith("text_encoder/") and not member.is_dir():
                target = (SUBMISSION_DIR / member.filename).resolve()
                if not target.is_relative_to(SUBMISSION_DIR.resolve()):
                    raise ValueError("Unsafe encoder path in ZIP.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
    if not (SUBMISSION_DIR / "text_encoder/modules.json").is_file():
        raise FileNotFoundError("The ZIP does not contain the frozen encoder.")


def verify_checkpoint():
    import keras

    config = json.loads((SUBMISSION_DIR / "config.json").read_text())
    if config["method"]["hidden"] != [32, 16] or config["refit_on_all_public"]:
        raise ValueError("Unexpected checkpoint or training policy.")
    checkpoint_path = HERE / "results/validation/best_fnn.keras"
    if file_sha256(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("The compiled checkpoint differs from the submission provenance.")
    model = keras.models.load_model(checkpoint_path, compile=False)
    if model.count_params() != 21185:
        raise ValueError("Expected the selected 21,185-parameter FNN.")
    with np.load(SUBMISSION_DIR / "weights.npz", allow_pickle=False) as archive:
        weights = {key: archive[key].copy() for key in archive.files}
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    for index, layer in enumerate(dense):
        for key, value in zip((f"W{index}", f"b{index}"), layer.get_weights()):
            np.testing.assert_array_equal(weights[key], value)
    probes = np.random.default_rng(42).normal(size=(16, 644)).astype(np.float32)
    np.testing.assert_allclose(model.predict(probes, verbose=0).reshape(-1),
                               sigmoid(network_logits(probes, weights)), atol=1e-5)
    keras.backend.clear_session()
    return {"parameters": 21185, "checkpoint_parity_cases": len(probes),
            "checkpoint_sha256": file_sha256(checkpoint_path)}


def build(output=None):
    restore_encoder()
    checks = verify_checkpoint()
    output = Path(output).resolve() if output else HERE / "submission_fnn.zip"
    if output.is_relative_to(SUBMISSION_DIR.resolve()):
        raise ValueError("The ZIP must be outside its source submission folder.")
    required = ("model.py", "features.py", "text_encoding.py", "config.json", "weights.npz",
                "requirements.txt", "prediction_config.json", "text_encoder/modules.json",
                "text_encoder/model.safetensors")
    if any(not (SUBMISSION_DIR / name).is_file() for name in required):
        raise FileNotFoundError("The submission folder is incomplete.")
    files = [path for path in sorted(SUBMISSION_DIR.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(SUBMISSION_DIR).as_posix())
    temporary.replace(output)
    checks.update({"zip_sha256": file_sha256(output), "zip_bytes": output.stat().st_size,
                   "bundle_files": {path.relative_to(SUBMISSION_DIR).as_posix(): file_sha256(path) for path in files},
                   "note": "Local packaging checks only; no CodaBench submission was made."})
    save_json(HERE / "results/submission_checks.json", checks)
    print(f"Ready to upload: {output}")
    print(f"SHA-256: {checks['zip_sha256']}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.output)
