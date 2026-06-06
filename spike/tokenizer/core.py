import numpy as np
from .vocab import HolographicVocab
from ..entropy.analyzer import EntropyAnalyzer

class SNNTokenizer:
    def __init__(self, vocab: HolographicVocab):
        self.vocab = vocab
        # Map of (ID_A, ID_B) -> Fused Hash ID
        self.merge_rules = {}
        # Reverse lookup just for human-readable debugging
        self.hash_to_bytes = {}
        
    def train(self, text: str, iterations: int = 10, min_freq: int = 2):
        """
        Trains the Entropy-BPE merges directly on the raw byte stream.
        """
        raw_bytes = text.encode('utf-8')
        
        # Initial state: every raw byte gets its FNV-1a hash
        tokens = []
        for b in raw_bytes:
            b_chunk = bytes([b])
            h = self.vocab.get_hash(b_chunk)
            tokens.append(h)
            self.hash_to_bytes[h] = b_chunk
            
        tokens = np.array(tokens, dtype=np.uint32)
        
        # Iteratively find entropy-optimal merges (EBPE loop)
        for _ in range(iterations):
            merges = EntropyAnalyzer.find_optimal_merges(tokens, min_freq=min_freq, entropy_threshold=0.1)
            if not merges:
                break
                
            for pair in merges:
                # Merge the underlying bytes
                b_A = self.hash_to_bytes[pair[0]]
                b_B = self.hash_to_bytes[pair[1]]
                fused_bytes = b_A + b_B
                
                # Generate exact deterministic hash for the fused bytes
                fused_hash = self.vocab.get_hash(fused_bytes)
                
                self.merge_rules[(pair[0], pair[1])] = fused_hash
                self.hash_to_bytes[fused_hash] = fused_bytes
                
            # Re-encode to continue training
            tokens = self.encode(text)

    def encode(self, text: str) -> np.ndarray:
        """
        Production-grade fast byte-level encoding. Applies trained EBPE merge rules.
        """
        raw_bytes = text.encode('utf-8')
        tokens = [self.vocab.get_hash(bytes([b])) for b in raw_bytes]
        
        # Apply merges efficiently
        while len(tokens) >= 2:
            new_tokens = []
            i = 0
            merged_any = False
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i+1])
                if pair in self.merge_rules:
                    new_tokens.append(self.merge_rules[pair])
                    i += 2
                    merged_any = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if i == len(tokens) - 1:
                new_tokens.append(tokens[-1])
                
            tokens = new_tokens
            if not merged_any:
                break
                
        return np.array(tokens, dtype=np.uint32)

    def decode(self, tokens: np.ndarray) -> str:
        """
        Decoder: Reconstructs the exact text from the dense hash tokens.
        """
        raw_bytes = bytearray()
        for t in tokens:
            # We look up the raw bytes from the hash dictionary
            raw_bytes.extend(self.hash_to_bytes[t])
            
        return raw_bytes.decode('utf-8', errors='replace')

