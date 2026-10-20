"""Frozen SBERT text windows shared by training and the submission."""

from bisect import bisect_left
import re
import numpy as np

MAX_TOKENS = 256
WINDOW_TOKENS = 254
ENCODING_VERSION = "context_code_windows_v1"


def select_windows(text, offsets, width=WINDOW_TOKENS):
    """Select content-only windows, without consulting benchmark IDs or labels."""
    count = len(offsets)
    if count <= width:
        return [(0, count)]
    starts = [offset[0] for offset in offsets]

    def token_at(character):
        return min(count, bisect_left(starts, character))

    abstract = re.search(r"\\begin\{abstract\}", text)
    document = re.search(r"\\begin\{document\}", text)
    context = token_at((abstract or document).end()) if abstract or document else 0
    end_document = re.search(r"\\end\{document\}", text)
    context_end = token_at(end_document.start()) if end_document else count
    code = re.search(r"(?im)^\s*Here is the code[^\n]*:\s*", text)
    guidelines = re.search(r"(?im)^\s*Please implement the missing code in the TODO blocks", text)
    useful_end = token_at(guidelines.start()) if code and guidelines else count
    code_start = token_at(code.end()) if code else 0
    targets = list(re.finditer(r"(?im)^\s*#\s*TODO:\s*Implement block[^\n]*", text))
    targets = [token_at(match.start()) for match in targets
               if code and code.end() <= match.start() < (guidelines.start() if guidelines else len(text))]
    if code and targets:
        proposals = [context, max(context, (context + context_end - width) // 2),
                     targets[0] - 96, targets[-1] - 96, code_start]
    else:
        proposals = np.linspace(context, max(context, useful_end - width), 4).astype(int).tolist()
    proposals.extend(np.linspace(context, max(context, useful_end - width), 4).astype(int).tolist())
    windows = []
    for proposed in proposals:
        start = min(max(0, int(proposed)), max(0, useful_end - width))
        window = (start, min(start + width, useful_end))
        if window not in windows:
            windows.append(window)
        if len(windows) == 4:
            break
    return windows

def encode_token_windows(encoder, sequences, batch_size=64):
    """Run the frozen SBERT stack on exact token windows, with no retokenization."""
    import torch

    tokenizer = encoder.tokenizer
    encoder.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            inputs = tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt")
            inputs = {key: value.to(encoder.device) for key, value in inputs.items()}
            vectors = encoder(inputs)["sentence_embedding"]
            result.append(torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().numpy())
            if len(sequences) > 1000 and start % 1024 == 0:
                print(f"Frozen SBERT windows: {min(start + batch_size, len(sequences)):,}/{len(sequences):,}", flush=True)
    return np.concatenate(result).astype(np.float32)

def bert_window(tokenizer, ids):
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("The frozen MiniLM tokenizer must define BERT CLS and SEP tokens.")
    return [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]

def encode_text(encoder, text):
    """Match the evaluated short-prompt and normalized-window representations."""
    tokenized = encoder.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                  truncation=False, verbose=False)
    ids = tokenized["input_ids"]
    if len(ids) <= WINDOW_TOKENS:
        return encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)[0].astype(np.float32)
    windows = select_windows(text, tokenized["offset_mapping"])
    sequences = [bert_window(encoder.tokenizer, ids[a:b]) for a, b in windows]
    if any(len(sequence) > MAX_TOKENS for sequence in sequences):
        raise ValueError("A selected window exceeds the encoder limit.")
    pooled = encode_token_windows(encoder, sequences).mean(axis=0)
    return (pooled / max(np.linalg.norm(pooled), 1e-12)).astype(np.float32)
