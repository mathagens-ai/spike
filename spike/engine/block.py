import numpy as np
from .norm import RMSNorm
from .attention import FluxAttention
from .ffn import FluxFFN
from .fluxbits import FluxCompiler

class SNNBlock:
    """
    Super Neural Network (SNN) Block.
    Integrates Pre-LN RMSNorm, FluxAttention, and FluxFFN with exact residual connections.
    Maintains clean BPTT recurrent state tracking and Straight-Through Estimator (STE) gradients.
    """
    def __init__(self, d_model, n_heads, hidden_dim, bits_per_param_attn=0.45, bits_per_param_ffn=0.22, name="block"):
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.name = name
        
        self.rms_attn = RMSNorm(d_model)
        self.attn = FluxAttention(d_model, n_heads, bits_per_param=bits_per_param_attn, name=f"{name}.attn")
        self.rms_ffn = RMSNorm(d_model)
        self.ffn = FluxFFN(d_model, hidden_dim, bits_per_param=bits_per_param_ffn, name=f"{name}.ffn")
        
        # Saved state for exact backward passes
        self.last_x_dense = None
        self.last_norm_1 = None
        self.last_x_norm_1 = None
        self.last_rms_1 = None
        
        self.last_y = None
        self.last_norm_2 = None
        self.last_x_norm_2 = None
        self.last_rms_2 = None

    def forward(self, x_dense, s_prev):
        """
        Forward pass of SNNBlock.
        x_dense: Dense FP32 input of shape (Batch, d_model)
        s_prev: Recurrent latent state from previous timestep (Batch, d_model)
        Returns:
            out: Block output dense tensor (Batch, d_model)
            s_next: Updated recurrent latent state (Batch, d_model)
        """
        # Save inputs
        self.last_x_dense = np.ascontiguousarray(x_dense)
        
        # 1. Pre-LN for Attention
        norm_1, x_norm_1, rms_1 = self.rms_attn.forward(x_dense)
        self.last_norm_1 = norm_1
        self.last_x_norm_1 = x_norm_1
        self.last_rms_1 = rms_1
        
        # Binarize norm_1 for Attention projections
        norm_1_bin = FluxCompiler.binarize(
            norm_1, 
            self.attn.compiled_q['K_bits'], 
            self.attn.compiled_q['d_hashes']
        )
        
        # Attention forward pass
        attn_out, s_next = self.attn.forward(norm_1_bin, norm_1, s_prev)
        
        # Residual connection
        y = x_dense + attn_out
        self.last_y = y
        
        # 2. Pre-LN for FFN
        norm_2, x_norm_2, rms_2 = self.rms_ffn.forward(y)
        self.last_norm_2 = norm_2
        self.last_x_norm_2 = x_norm_2
        self.last_rms_2 = rms_2
        
        # Binarize norm_2 for FFN projections
        norm_2_bin = FluxCompiler.binarize(
            norm_2, 
            self.ffn.compiled_gate['K_bits'], 
            self.ffn.compiled_gate['d_hashes']
        )
        
        # FFN forward pass
        ffn_out = self.ffn.forward(norm_2_bin, norm_2)
        
        # Residual connection
        out = y + ffn_out
        
        return out, s_next

    def backward(self, grad_out, grad_s_next):
        """
        Backward pass of SNNBlock.
        grad_out: Gradient of shape (Batch, d_model) backpropagated from the block/layer above.
        grad_s_next: Gradient of shape (Batch, d_model) backpropagated from the next timestep's attention state.
        Returns:
            grad_x_dense: Gradient with respect to block input (Batch, d_model)
            grad_s_prev: Gradient with respect to previous recurrent state (Batch, d_model)
        """
        # 1. Backpropagation through second residual connection (FFN branch)
        # out = y + ffn_out
        grad_ffn_out = grad_out
        grad_y = grad_out
        
        # Backprop through FFN (uses STE)
        grad_norm_2 = self.ffn.backward(grad_ffn_out)
        
        # Backprop through RMSNorm 2
        grad_y_from_norm_2 = self.rms_ffn.backward(
            grad_norm_2, 
            self.last_y, 
            self.last_x_norm_2, 
            self.last_rms_2
        )
        
        # Total gradient flow at split point y
        grad_y_total = grad_y + grad_y_from_norm_2
        
        # 2. Backpropagation through first residual connection (Attention branch)
        # y = x_dense + attn_out
        grad_attn_out = grad_y_total
        grad_x_dense = grad_y_total
        
        # Backprop through Attention (recurrent BPTT + STE)
        grad_norm_1, grad_s_prev = self.attn.backward(grad_attn_out, grad_s_next)
        
        # Backprop through RMSNorm 1
        grad_x_dense_from_norm_1 = self.rms_attn.backward(
            grad_norm_1, 
            self.last_x_dense, 
            self.last_x_norm_1, 
            self.last_rms_1
        )
        
        # Total gradient flow back to block input
        grad_x_dense_total = grad_x_dense + grad_x_dense_from_norm_1
        
        return grad_x_dense_total, grad_s_prev

    def step_optimizer(self, metabolism, velocity, lr, weight_decay=1e-4, critical_period_mult=1.0):
        """
        Perform metabolic, velocity, and gradient step updates for all nested sub-modules.
        """
        self.rms_attn.step_optimizer(lr, weight_decay=weight_decay)
        self.attn.step_optimizer(metabolism, velocity, lr, weight_decay=weight_decay, critical_period_mult=critical_period_mult)
        self.rms_ffn.step_optimizer(lr, weight_decay=weight_decay)
        self.ffn.step_optimizer(metabolism, velocity, lr, weight_decay=weight_decay, critical_period_mult=critical_period_mult)
