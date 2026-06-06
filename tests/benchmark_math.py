import sys
import os
import time
import json
import numpy as np

# Setup paths for SNN language module

from spike.tokenizer.vocab import HolographicVocab
from spike.tokenizer.core import SNNTokenizer
from spike.semantic.mapper import SemanticMapper
from spike.semantic.alibi import ALiBiEngine

def load_math_dataset(filepath, num_records=None):
    """Loads the GSM8K dataset as a continuous text stream."""
    text_corpus = ""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # Combine question and answer into the context stream
                text_corpus += f"Q: {data['question']}\nA: {data['answer']}\n\n"
                count += 1
                if num_records and count >= num_records:
                    break
            except Exception as e:
                pass
    return text_corpus, count

def run_math_benchmark():
    pass
    pass
    pass
    
    vocab = HolographicVocab()
    tokenizer = SNNTokenizer(vocab)
    mapper = SemanticMapper(d_model=256, num_hashes=3)
    
    dataset_path = r"C:\Users\aryan\Downloads\datasets\math\gsm8k_train.jsonl"
    
    pass
    start_t = time.perf_counter()
    text_stream, total_records = load_math_dataset(dataset_path, num_records=None)
    load_time = time.perf_counter() - start_t
    pass
    
    pass
    start_t = time.perf_counter()
    # Math text has lots of repeated symbols like <<, >>, =, numbers.
    # Entropy-BPE will fuse these naturally.
    tokenizer.train(text_stream, iterations=15, min_freq=5)
    train_time = time.perf_counter() - start_t
    pass
    
    pass
    start_t = time.perf_counter()
    encoded_tokens = tokenizer.encode(text_stream)
    enc_time = time.perf_counter() - start_t
    pass
    
    pass
    start_t = time.perf_counter()
    fluxbits_matrix = mapper.hash_to_fluxbits_vectorized(encoded_tokens)
    
    # ALiBi is calculated per Attention Block (e.g. chunk size 4096) to prevent 80GB dense matrix explosions
    block_size = min(len(encoded_tokens), 4096)
    penalties = ALiBiEngine.build_penalty_matrix(seq_len=block_size, n_heads=4)
    map_time = time.perf_counter() - start_t
    pass
    pass
    pass
    
    pass
    # We will decode the first 250 tokens just to prove lossless reconstruction
    start_t = time.perf_counter()
    reconstructed_text = tokenizer.decode(encoded_tokens[:250])
    dec_time = time.perf_counter() - start_t
    pass
    
    pass
    pass
    pass
    pass

    pass
    pass
    pass

if __name__ == "__main__":
    run_math_benchmark()
