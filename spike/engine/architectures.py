import numpy as np
from .attention import FluxAttention
from .inference import SNNInferenceEngine

class SNNEncoder:
    """
    SNN Encoder Architecture.
    Reads the full prompt holistically using a Bidirectional Latent Scan.
    Generates a single, ultra-compressed context vector `s_enc`.
    """
    def __init__(self, d_model=256, bits_per_param=0.45):
        self.d_model = d_model
        self.bits_per_param = bits_per_param
        # Encoder uses FluxAttention in non-causal (bidirectional) mode
        self.attention = FluxAttention(d_model=d_model, n_heads=4, bits_per_param=bits_per_param, name="encoder_attn")
        self.engine = SNNInferenceEngine(self.attention)
        
    def encode(self, prompt_fluxbits, prompt_dense):
        """
        Compresses the prompt into s_enc.
        """
        batch_size, _, _ = prompt_dense.shape
        s_init = np.zeros((batch_size, self.d_model), dtype=np.float32)
        
        # is_causal=False triggers the bidirectional scan
        _, s_enc = self.attention.parallel_forward(
            x_bin_seq=prompt_fluxbits,
            x_dense_seq=prompt_dense,
            s_prev_init=s_init,
            is_causal=False
        )
        return s_enc

class SNNDecoder:
    """
    SNN Decoder Architecture.
    Generates tokens autoregressively.
    If conditioned on an Encoder, it accepts `s_enc` as its starting state.
    """
    def __init__(self, d_model=256, bits_per_param=0.45):
        self.d_model = d_model
        self.bits_per_param = bits_per_param
        # Decoder strictly uses causal mode
        self.attention = FluxAttention(d_model=d_model, n_heads=4, bits_per_param=bits_per_param, name="decoder_attn")
        self.engine = SNNInferenceEngine(self.attention)
        
    def decode_step(self, token_fluxbits, token_dense, s_prev):
        """
        A single autoregressive step.
        """
        out, s_next = self.attention.forward(token_fluxbits, token_dense, s_prev)
        return out, s_next

class SNNEncoderDecoder:
    """
    The Ultimate T5/BART replacement without KV Cache.
    Uses 'Latent State Injection' instead of Cross-Attention.
    """
    def __init__(self, d_model=256, bits_per_param=0.45):
        self.encoder = SNNEncoder(d_model, bits_per_param)
        self.decoder = SNNDecoder(d_model, bits_per_param)
        
    def generate(self, prompt_fluxbits, prompt_dense, lm_head, decoding_engine, max_new_tokens=10, temperature=1.0):
        """
        1. Encode prompt -> s_enc
        2. Inject s_enc into decoder's starting state
        3. Autoregressive generation (State -> LMHead -> DecodingEngine -> New Token)
        """
        s_enc = self.encoder.encode(prompt_fluxbits, prompt_dense)
        s_dec_curr = s_enc.copy()
        
        batch_size = prompt_dense.shape[0]
        assert batch_size == 1, "Generation currently supports batch size 1"
        
        generated_token_ids = []
        
        # We need the SOS (Start of Sequence) or last token of prompt
        # Here we just start with zero embeddings for the very first decode step
        current_token_dense = np.zeros((batch_size, self.decoder.d_model), dtype=np.float32)
        from .fluxbits import FluxCompiler
        current_token_bin = FluxCompiler.binarize(current_token_dense, self.decoder.attention.compiled_q['K_bits'], self.decoder.attention.compiled_q['d_hashes'])
        
        for _ in range(max_new_tokens):
            # 1. Step the SNN forward
            out_state, s_dec_curr = self.decoder.decode_step(current_token_bin, current_token_dense, s_dec_curr)
            
            # 2. Pipe the continuous state through the Holographic LMHead to get discrete logits
            logits = lm_head.compute_logits(out_state)
            
            # 3. Pipe the logits through the Decoding Engine Sampler to get the actual Token ID
            next_token_idx = decoding_engine.sample(logits, temperature=temperature, past_tokens=generated_token_ids)
            
            # 4. Save the exact token hash
            generated_token_ids.append(next_token_idx)
            
            # 5. Route the token back in as the input for the next step!
            # The exact footprint of the generated token becomes the new input feature.
            current_token_dense = lm_head.vocab_matrix[next_token_idx:next_token_idx+1]
            current_token_bin = FluxCompiler.binarize(current_token_dense, self.decoder.attention.compiled_q['K_bits'], self.decoder.attention.compiled_q['d_hashes'])
            
        return generated_token_ids, s_enc
