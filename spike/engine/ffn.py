import numpy as np
from .fluxbits import FluxCompiler, AffineCalibrator

class FluxFFN:
    """
    0.25-bit SwiGLU Feed-Forward Network.
    Contains the bulk of the parameters in SNN.
    Uses 0.25-bit Bloom projections for gate, up, and down layers
    to achieve extremely small memory footprint.
    Fully implements original BPTT and STE gradient equations.
    """
    def __init__(self, d_model, hidden_dim, bits_per_param=0.22, name="ffn"):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.bits_per_param = bits_per_param
        self.name = name
        
        # Initialize FP32 shadow weights:
        # gate_proj and up_proj shape: (hidden_dim, d_model)
        # down_proj shape: (d_model, hidden_dim)
        limit = np.sqrt(1.0 / d_model)
        self.gate_proj_weight = np.random.uniform(-limit, limit, size=(hidden_dim, d_model)).astype(np.float32)
        self.up_proj_weight = np.random.uniform(-limit, limit, size=(hidden_dim, d_model)).astype(np.float32)
        self.down_proj_weight = np.random.uniform(-limit, limit, size=(d_model, hidden_dim)).astype(np.float32)
        
        # Gradient accumulators
        self.gate_proj_weight_grad = np.zeros_like(self.gate_proj_weight)
        self.up_proj_weight_grad = np.zeros_like(self.up_proj_weight)
        self.down_proj_weight_grad = np.zeros_like(self.down_proj_weight)
        
        # Compile initially
        self.recompile()

    def recompile(self, coactivation_matrix=None):
        """
        Compile FP32 shadow weights down to compressed Bloom Filter/Quantized projections.
        """
        self.compiled_gate = FluxCompiler.compile(self.gate_proj_weight, bits_per_param=self.bits_per_param, coactivation_matrix=coactivation_matrix)
        self.compiled_up = FluxCompiler.compile(self.up_proj_weight, bits_per_param=self.bits_per_param, coactivation_matrix=coactivation_matrix)
        self.compiled_down = FluxCompiler.compile(self.down_proj_weight, bits_per_param=self.bits_per_param, coactivation_matrix=coactivation_matrix)
        
        # Initialize Calibrators
        self.calibrator_gate = AffineCalibrator(self.hidden_dim, self.compiled_gate['K_bits'], self.compiled_gate['density'], identity=(self.compiled_gate.get('backend') in ('1bf16', 'binary', 'int4', 'int8')))
        self.calibrator_up = AffineCalibrator(self.hidden_dim, self.compiled_up['K_bits'], self.compiled_up['density'], identity=(self.compiled_up.get('backend') in ('1bf16', 'binary', 'int4', 'int8')))
        self.calibrator_down = AffineCalibrator(self.d_model, self.compiled_down['K_bits'], self.compiled_down['density'], identity=(self.compiled_down.get('backend') in ('1bf16', 'binary', 'int4', 'int8')))
        
        # Pre-calibrate using empirical dummy runs
        dummy_x = np.random.normal(0, 1.0, size=(10, self.d_model)).astype(np.float32)
        dummy_packed = FluxCompiler.binarize(dummy_x, self.compiled_gate['K_bits'], self.compiled_gate['d_hashes'])
        
        self.calibrator_gate.calibrate(FluxCompiler.forward(dummy_packed, self.compiled_gate))
        self.calibrator_up.calibrate(FluxCompiler.forward(dummy_packed, self.compiled_up))
        
        # down_proj expects hidden_bin which has K_bits matching compiled_down
        dummy_h = np.random.normal(0, 1.0, size=(10, self.hidden_dim)).astype(np.float32)
        dummy_h_packed = FluxCompiler.binarize(
            dummy_h, 
            self.compiled_down['K_bits'], 
            self.compiled_down['d_hashes']
        )
        self.calibrator_down.calibrate(FluxCompiler.forward(dummy_h_packed, self.compiled_down))

    def forward(self, x_bin, x_dense):
        """
        Forward pass of SwiGLU FluxFFN.
        x_bin: Packed binary input of shape (Batch, K_bytes)
        x_dense: Dense FP32 input of shape (Batch, d_model)
        """
        # Save inputs for backward pass
        self.last_x_bin = np.ascontiguousarray(x_bin)
        self.last_x_dense = np.ascontiguousarray(x_dense)
        
        # 1. POPCOUNT Projections from 0.25-bit Bloom Tensors
        raw_gate = FluxCompiler.forward(x_bin, self.compiled_gate)
        raw_up = FluxCompiler.forward(x_bin, self.compiled_up)
        
        gate = self.calibrator_gate.apply(raw_gate)
        up = self.calibrator_up.apply(raw_up)
        
        self.last_gate = gate
        self.last_up = up
        
        # 2. Swish Activation (SiLU) and Element-wise Multiply
        # Sigmoid with extreme float clipping for numerical stability (exact C++ parity)
        clipped_gate = np.clip(gate, -15.0, 15.0)
        sigmoid_gate = 1.0 / (1.0 + np.exp(-clipped_gate))
        silu = gate * sigmoid_gate
        
        # SwiGLU element-wise multiplication
        hidden = silu * up
        self.last_sigmoid_gate = sigmoid_gate
        self.last_silu = silu
        self.last_hidden = hidden
        
        # 3. Binarize Hidden State for Down Projection using FluxCompiler binarization
        hidden_bin = FluxCompiler.binarize(
            hidden, 
            self.compiled_down['K_bits'], 
            self.compiled_down['d_hashes']
        )
        self.last_hidden_bin = hidden_bin
        
        # 4. Final Projection back to d_model
        raw_out = FluxCompiler.forward(hidden_bin, self.compiled_down)
        out = self.calibrator_down.apply(raw_out)
        
        return out

    def backward(self, grad_out):
        """
        Straight-Through Estimator (STE) backward pass.
        grad_out: Gradient of shape (Batch, d_model) from the layer above.
        Returns:
            grad_x_dense: Gradient with respect to the input x_dense (Batch, d_model)
        """
        # 1. Backprop through Down Projection under STE
        scaled_grad_out = grad_out * self.calibrator_down.gamma
        
        # Binarize threshold mask of the hidden state
        thresh_h = np.mean(np.abs(self.last_hidden), axis=-1, keepdims=True)
        hidden_active = (np.abs(self.last_hidden) > thresh_h).astype(np.float32)
        
        # Accumulate down_proj weight gradients
        self.down_proj_weight_grad += scaled_grad_out.T @ hidden_active
        
        # Gradient flowing back to hidden state
        grad_hidden = scaled_grad_out @ self.down_proj_weight
        
        # 2. Backprop through SwiGLU Element-wise Multiply & SiLU
        # hidden = silu * up
        grad_up = grad_hidden * self.last_silu
        grad_silu = grad_hidden * self.last_up
        
        # d_silu = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        grad_gate = grad_silu * self.last_sigmoid_gate * (1.0 + self.last_gate * (1.0 - self.last_sigmoid_gate))
        
        # 3. Backprop through gate_proj and up_proj under STE
        scaled_grad_gate = grad_gate * self.calibrator_gate.gamma
        scaled_grad_up = grad_up * self.calibrator_up.gamma
        
        # Binarize threshold mask of the input token
        thresh = np.mean(np.abs(self.last_x_dense), axis=-1, keepdims=True)
        x_active = (np.abs(self.last_x_dense) > thresh).astype(np.float32)
        
        # Accumulate gate and up projection weight gradients
        self.gate_proj_weight_grad += scaled_grad_gate.T @ x_active
        self.up_proj_weight_grad += scaled_grad_up.T @ x_active
        
        # 4. Compute gradient with respect to x_dense
        grad_x_dense = (
            scaled_grad_gate @ self.gate_proj_weight +
            scaled_grad_up @ self.up_proj_weight
        )
        
        return grad_x_dense

    def step_optimizer(self, metabolism, velocity, lr, weight_decay=1e-4, critical_period_mult=1.0):
        """
        Execute metabolic and gradient velocity updates for gate, up, and down projections.
        """
        params = [
            ("gate_proj", self.gate_proj_weight, self.gate_proj_weight_grad),
            ("up_proj", self.up_proj_weight, self.up_proj_weight_grad),
            ("down_proj", self.down_proj_weight, self.down_proj_weight_grad)
        ]
        
        for p_name, p_weight, p_grad in params:
            full_name = f"{self.name}.{p_name}"
            
            # Update metabolism cell states
            metabolism.step(full_name, p_weight, p_grad)
            
            # Execute synaptic competition
            metabolism.compete(full_name, p_weight)
            
            # Retrieve vitality scale
            vitality = metabolism.states[full_name].vitality
            
            # Compute predicted gradient look-ahead update step
            delta = velocity.update_and_get_delta(
                full_name, 
                p_grad, 
                lr, 
                vitality, 
                critical_period_mult
            )
            
            # Apply weight decay and update step
            p_weight *= (1.0 - lr * weight_decay)
            p_weight -= delta
            
            # Reset gradients
            p_grad.fill(0.0)
            
        # Recompile compiled Bloom representations
        self.recompile()
