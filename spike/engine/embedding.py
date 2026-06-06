import numpy as np
from .fluxbits import FluxCompiler, AffineCalibrator

class FluxEmbedding:
    """
    0.45-bit FluxEmbedding Layer.
    Compiles token embeddings into a compressed 0.45-bit Bloom tensor.
    Uses continuous FP32 shadow weights during training and executes
    mathematically exact STE backpropagation.
    """
    def __init__(self, vocab_size, d_model, bits_per_param=0.45, name="embedding"):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bits_per_param = bits_per_param
        self.name = name
        
        # Initialize FP32 shadow weights: shape (d_model, vocab_size)
        # M = d_model, N = vocab_size
        limit = np.sqrt(1.0 / d_model)
        self.weight = np.random.uniform(-limit, limit, size=(d_model, vocab_size)).astype(np.float32)
        
        # Gradient accumulator
        self.weight_grad = np.zeros_like(self.weight, dtype=np.float32)
        
        # Compile initially
        self.recompile()

    def recompile(self, coactivation_matrix=None):
        """
        Compile FP32 shadow weights down to compressed Bloom Filter rows.
        Optionally uses Hebbian co-activation matrix to align hash collisions.
        """
        self.compiled_w = FluxCompiler.compile(
            self.weight, 
            bits_per_param=self.bits_per_param, 
            coactivation_matrix=coactivation_matrix
        )
        
        # Initialize Affine Calibrator
        self.calibrator = AffineCalibrator(
            M=self.d_model, 
            K_bits=self.compiled_w['K_bits'], 
            bloom_density=self.compiled_w['density'],
            identity=(self.compiled_w.get('backend') in ('1bf16', 'binary', 'int4', 'int8'))
        )
        
        # Pre-calibrate the calibrator with empirical sample forward popcounts
        dummy_x = np.random.normal(0, 1.0, size=(10, self.vocab_size)).astype(np.float32)
        dummy_packed = FluxCompiler.binarize(
            dummy_x, 
            self.compiled_w['K_bits'], 
            self.compiled_w['d_hashes']
        )
        dummy_raw = FluxCompiler.forward(dummy_packed, self.compiled_w)
        self.calibrator.calibrate(dummy_raw)

    def forward(self, x_ids):
        """
        Forward pass.
        x_ids: Integer array of shape (Batch, SeqLen)
        Returns calibrated FP32 continuous embedding of shape (Batch, SeqLen, d_model)
        """
        self.last_x_ids = np.ascontiguousarray(x_ids)
        Batch, SeqLen = x_ids.shape
        total_tokens = Batch * SeqLen
        
        # Flatten token ids to construct one-hot vectors
        flat_ids = x_ids.ravel()
        
        # Construct one-hot representation for exact mathematical binarization
        one_hot = np.zeros((total_tokens, self.vocab_size), dtype=np.float32)
        one_hot[np.arange(total_tokens), flat_ids] = 1.0
        
        # Binarize to Bloom packed representation: shape (Batch * SeqLen, K_bytes)
        x_packed = FluxCompiler.binarize(
            one_hot, 
            self.compiled_w['K_bits'], 
            self.compiled_w['d_hashes']
        )
        
        # Fast vectorized bitwise forward: shape (Batch * SeqLen, d_model)
        raw_popcounts = FluxCompiler.forward(x_packed, self.compiled_w)
        
        # Calibrate popcounts: shape (Batch * SeqLen, d_model)
        calibrated = self.calibrator.apply(raw_popcounts)
        
        # Reshape to final 3D tensor
        return calibrated.reshape(Batch, SeqLen, self.d_model)

    def backward(self, grad_y):
        """
        Straight-Through Estimator (STE) backward pass.
        grad_y: Gradient of shape (Batch, SeqLen, d_model)
        Returns: None (Gradients are accumulated into self.weight_grad)
        """
        Batch, SeqLen, d_model = grad_y.shape
        flat_grad = grad_y.reshape(-1, d_model)
        flat_ids = self.last_x_ids.ravel()
        
        # Scale output gradients by the calibrator's gamma scaling factor
        scaled_grad = flat_grad * self.calibrator.gamma
        
        # Accumulate gradients into corresponding columns of the shadow weight
        # weight is shape (d_model, vocab_size), so weight.T is shape (vocab_size, d_model)
        np.add.at(self.weight_grad.T, flat_ids, scaled_grad)

    def step_optimizer(self, metabolism, velocity, lr, weight_decay=1e-4, critical_period_mult=1.0):
        """
        Execute metabolic and gradient velocity update.
        Integrates with MetabolismEngine and VelocityEngine.
        """
        # 1. Update metabolism state
        metabolism.step(self.name + ".weight", self.weight, self.weight_grad)
        
        # 2. Run synaptic competition
        metabolism.compete(self.name + ".weight", self.weight)
        
        # 3. Retrieve vitality state
        vitality = metabolism.states[self.name + ".weight"].vitality
        
        # 4. Compute predicted gradient velocity step
        delta = velocity.update_and_get_delta(
            self.name + ".weight", 
            self.weight_grad, 
            lr, 
            vitality, 
            critical_period_mult
        )
        
        # 5. Apply update & weight decay to shadow weights
        self.weight *= (1.0 - lr * weight_decay)
        self.weight -= delta
        
        # 6. Recompile compiled Bloom representation
        self.recompile()
        
        # 7. Reset gradient accumulator
        self.weight_grad.fill(0.0)
