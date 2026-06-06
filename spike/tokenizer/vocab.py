class HolographicVocab:
    def __init__(self):
        # We don't store an enormous string dictionary. 
        # We rely on deterministic high-speed hashing.
        pass
        
    def get_hash(self, data: bytes) -> int:
        """
        Extremely fast FNV-1a 32-bit hash for raw bytes.
        Far faster than MD5 and perfect for infinite token hashing.
        """
        hash_val = 0x811c9dc5
        for b in data:
            hash_val ^= b
            hash_val = (hash_val * 0x01000193) & 0xFFFFFFFF
        return hash_val
