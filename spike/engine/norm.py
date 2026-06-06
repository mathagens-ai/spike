import numpy as np

class RMSNorm:
    """
    Full-Precision Root Mean Square Normalization (RMSNorm).
    Maintains numerical stability across deep layers in FP32 space.
    """
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        
        # Trainable scale parameter (FP32), initialized to ones
        self.scale = np.ones(d_model, dtype=np.float32)
        
        # Gradient tracking
        self.scale_grad = np.zeros(d_model, dtype=np.float32)
        self.scale_momentum = np.zeros(d_model, dtype=np.float32)
        self.scale_curvature = np.ones(d_model, dtype=np.float32)

    def forward(self, x):
        """
        Execute RMSNorm forward pass.
        rms = sqrt(1/d * sum(x^2) + eps)
        x_norm = x / rms
        y = scale * x_norm
        """
        x = x.astype(np.float32)
        
        # Compute root mean square
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        rms = np.sqrt(variance + self.eps)
        
        # Normalize and scale
        x_norm = x / rms
        y = self.scale * x_norm
        
        return y, x_norm, rms

    def backward(self, grad_y, x, x_norm, rms):
        """
        Mathematically exact backward pass for RMSNorm.
        Computes gradients for the input x and the trainable scale parameter.
        """
        # Gradient with respect to scale (summed across batch)
        self.scale_grad += np.sum(grad_y * x_norm, axis=0)
        
        # Gradient with respect to normalized input
        grad_x_norm = grad_y * self.scale
        
        # Gradient with respect to RMS term
        # d_rms_inv / dx = - (x / d_model) * rms_inv^3
        # Which leads to:
        term1 = grad_x_norm / rms
        term2 = (x / (self.d_model * (rms ** 2))) * np.sum(grad_x_norm * x, axis=-1, keepdims=True)
        grad_x = term1 - (term2 / rms)
        
        return grad_x

    def step_optimizer(self, lr, momentum_decay=0.9, weight_decay=1e-4):
        """Update trainable scale weights using LGC optimizer rules."""
        # Gradient clipping
        grad_norm = np.linalg.norm(self.scale_grad)
        if grad_norm > 1.0:
            self.scale_grad /= grad_norm
            
        # Update Adam-like states
        self.scale_momentum = momentum_decay * self.scale_momentum + self.scale_grad
        self.scale_curvature = momentum_decay * self.scale_curvature + self.scale_grad ** 2
        
        # Update step
        safe_curv = np.sqrt(self.scale_curvature) + 1e-8
        delta = self.scale_momentum / safe_curv
        
        # Weight decay and gradient step
        self.scale *= (1.0 - lr * weight_decay)
        self.scale -= lr * delta
        
        # Clear gradients
        self.scale_grad.fill(0.0)
