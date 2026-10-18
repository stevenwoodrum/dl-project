import os

sbert = None

def get_sbert():
    """Return a singleton instance of an SBERT model"""
    global sbert
    if sbert is None:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return sbert


def create_submission_zip(zip_name: str="submission", files: list=None):
    """
    Create a ZIP archive containing the given files, flattened into the root
    of the archive.

    Parameters
    ----------
    zip_name : str
        Output ZIP name *without* the .zip extension.
    files : list[str]
        List of file paths provided by the user. Each file is copied into the
        ZIP root without preserving directory structure.

    Notes
    -----
    - All files are flattened: only their basenames appear in the ZIP.
    - If multiple files share the same basename, later ones overwrite earlier
      ones.
    - Raises FileNotFoundError if any provided path does not exist.
    """
    import shutil
    import tempfile

    if not files:
        files = ["model.py", "test_model.py", "utils.py", "models.txt",
                 "model.keras"]

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"Required file not found: {f}")

            dest_path = os.path.join(tmpdir, os.path.basename(f))
            shutil.copy2(f, dest_path)

        shutil.make_archive(zip_name, "zip", tmpdir)