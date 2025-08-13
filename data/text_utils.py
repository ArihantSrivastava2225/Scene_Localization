import numpy as np

def tokenize_and_pad(query, vocab_size=10000, max_len=20):
    """Tokenize query and pad/truncate to match training."""
    tokens = query.lower().split()
    token_ids = [hash(w) % vocab_size for w in tokens]
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))  # pad
    else:
        token_ids = token_ids[:max_len]  # truncate
    return np.array(token_ids, dtype=np.int32)
