import numpy as np

class MultiScaleLatentState:
    """
    Multi-Scale Temporal Memory (KV-free Latent State).
    Replaces standard linear attention KV-cache with a bounded O(d_model) recurrent state.
    Features a learned, trainable decay rate (gamma) per dimension, partitioning
    memory into syntax, semantics, topic, and identity layers.
    """
    def __init__(self, d_model):
        self.d_model = d_model
        
        # Initialize gamma with 4 multi-scale temporal bands
        # Syntax (~1000 dims): γ ≈ 0.95 (20-token window)
        # Semantics (~1000 dims): γ ≈ 0.99 (100-token window)
        # Topic (~1000 dims): γ ≈ 0.999 (1000-token window)
        # Identity (~1000 dims): γ ≈ 0.9999 (10000+ token window)
        
        num_band = d_model // 4
        band_syntax = np.full(num_band, 0.95, dtype=np.float32)
        band_semantics = np.full(num_band, 0.99, dtype=np.float32)
        band_topic = np.full(num_band, 0.999, dtype=np.float32)
        # Handle residual dimensions in the last band
        band_identity = np.full(d_model - 3 * num_band, 0.9999, dtype=np.float32)
        
        # Merge bands to initialize the trainable gamma array
        self.gamma = np.concatenate([band_syntax, band_semantics, band_topic, band_identity])
        # Force gamma to stay within stable bounds [0.90, 0.9999]
        self.gamma = np.clip(self.gamma, 0.90, 0.9999)
        
        # Optimizer state for gamma
        self.gamma_grad = np.zeros(d_model, dtype=np.float32)
        self.gamma_momentum = np.zeros(d_model, dtype=np.float32)
        self.gamma_curvature = np.ones(d_model, dtype=np.float32)
        
        # Latent state vector (Batch, d_model)
        self.s = None

    def reset(self, batch_size):
        """Reset the recurrent latent state vector to zero."""
        self.s = np.zeros((batch_size, self.d_model), dtype=np.float32)
        return self.s

    def forward(self, update):
        """
        Execute a forward recurrent step.
        s_t = γ × s_{t-1} + (1 - γ) × update
        """
        if self.s is None or self.s.shape[0] != update.shape[0]:
            self.reset(update.shape[0])
            
        # Element-wise operations (numpy broadcasts self.gamma automatically)
        self.s = self.gamma * self.s + (1.0 - self.gamma) * update
        return self.s

    def backward(self, grad_s, prev_s, update):
        """
        Differentiable backward pass for learned gamma decay rates.
        Calculates exact gradients for the input update, previous state, and gamma.
        
        s_t = γ * s_{t-1} + (1 - γ) * update
        
        ∂s_t / ∂update = 1 - γ
        ∂s_t / ∂s_{t-1} = γ
        ∂s_t / ∂γ = s_{t-1} - update
        """
        # Gradients with respect to inputs
        grad_update = (1.0 - self.gamma) * grad_s
        grad_prev_s = self.gamma * grad_s
        
        # Gradient with respect to trainable parameter gamma (aggregated across batch)
        # ds_dgamma = s_{t-1} - update
        # We sum over the batch dimension
        grad_gamma_batch = grad_s * (prev_s - update)
        self.gamma_grad += np.sum(grad_gamma_batch, axis=0)
        
        return grad_update, grad_prev_s

    def step_optimizer(self, lr, momentum_decay=0.9, weight_decay=1e-4):
        """
        Update the trainable gamma parameter using standard AdamW-like LGC rules.
        """
        # Apply gradient clipping
        grad_norm = np.linalg.norm(self.gamma_grad)
        if grad_norm > 1.0:
            self.gamma_grad /= grad_norm
            
        # Update Adam-like states
        self.gamma_momentum = momentum_decay * self.gamma_momentum + self.gamma_grad
        self.gamma_curvature = momentum_decay * self.gamma_curvature + self.gamma_grad ** 2
        
        # Step update
        safe_curv = np.sqrt(self.gamma_curvature) + 1e-8
        delta = self.gamma_momentum / safe_curv
        
        # Weight decay and gradient step
        self.gamma *= (1.0 - lr * weight_decay)
        self.gamma -= lr * delta
        
        # Enforce mathematical bounds [0.90, 0.9999] for stability
        self.gamma = np.clip(self.gamma, 0.90, 0.9999)
        
        # Clear gradients
        self.gamma_grad.fill(0.0)

    def defragment(self):
        """Defragment latent state to avoid numerical drift or runaway values."""
        if self.s is not None:
            # Clip outlier values that may have accumulated over long sequences
            self.s = np.clip(self.s, -10.0, 10.0)
            # Remove minor floating point noise
            self.s[np.abs(self.s) < 1e-6] = 0.0
            
    def get_decay_stats(self):
        """Return the min, max, and average decay rates for debugging."""
        return {
            'min_gamma': float(np.min(self.gamma)),
            'max_gamma': float(np.max(self.gamma)),
            'avg_gamma': float(np.mean(self.gamma)),
        }
