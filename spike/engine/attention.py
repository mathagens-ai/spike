import numpy as np
from .fluxbits import FluxCompiler, AffineCalibrator

class FluxAttention:
    """
    0.45-bit FluxAttention Layer (KV-Free Recursive Attention).
    Replaces standard O(N) quadratic KV cache with an O(d_model) recurrent state.
    Uses 0.45-bit Bloom projections for Q, K, V, O and FP32 for temporal gating.
    Fully implements original BPTT and STE gradient equations.
    """
    def __init__(self, d_model, n_heads, bits_per_param=0.45, name="attn"):
        self.d_model = d_model
        self.n_heads = n_heads
        self.bits_per_param = bits_per_param
        self.name = name
        
        # 1. Initialize FP32 shadow weights for Q, K, V, O projections (shape: d_model, d_model)
        limit = np.sqrt(1.0 / d_model)
        self.q_proj_weight = np.random.uniform(-limit, limit, size=(d_model, d_model)).astype(np.float32)
        self.k_proj_weight = np.random.uniform(-limit, limit, size=(d_model, d_model)).astype(np.float32)
        self.v_proj_weight = np.random.uniform(-limit, limit, size=(d_model, d_model)).astype(np.float32)
        self.o_proj_weight = np.random.uniform(-limit, limit, size=(d_model, d_model)).astype(np.float32)
        
        # Gradient accumulators for projections
        self.q_proj_weight_grad = np.zeros_like(self.q_proj_weight)
        self.k_proj_weight_grad = np.zeros_like(self.k_proj_weight)
        self.v_proj_weight_grad = np.zeros_like(self.v_proj_weight)
        self.o_proj_weight_grad = np.zeros_like(self.o_proj_weight)
        
        # 2. Initialize FP32 dense gating weights (shape: d_model, d_model)
        self.w_gate = np.random.uniform(-limit, limit, size=(d_model, d_model)).astype(np.float32)
        self.w_update = np.random.uniform(-limit, limit, size=(d_model, d_model)).astype(np.float32)
        self.w_bypass = np.random.uniform(-limit, limit, size=(d_model, d_model)).astype(np.float32)
        
        # Gradient accumulators for gates
        self.w_gate_grad = np.zeros_like(self.w_gate)
        self.w_update_grad = np.zeros_like(self.w_update)
        self.w_bypass_grad = np.zeros_like(self.w_bypass)
        
        # Initial compilation
        self.recompile()

    def recompile(self, coactivation_matrix=None):
        """
        Compile FP32 shadow weights down to compressed Bloom Filter/Quantized projections.
        """
        self.compiled_q = FluxCompiler.compile(self.q_proj_weight, bits_per_param=self.bits_per_param, coactivation_matrix=coactivation_matrix)
        self.compiled_k = FluxCompiler.compile(self.k_proj_weight, bits_per_param=self.bits_per_param, coactivation_matrix=coactivation_matrix)
        self.compiled_v = FluxCompiler.compile(self.v_proj_weight, bits_per_param=self.bits_per_param, coactivation_matrix=coactivation_matrix)
        self.compiled_o = FluxCompiler.compile(self.o_proj_weight, bits_per_param=self.bits_per_param, coactivation_matrix=coactivation_matrix)
        
        # Initialize Calibrators
        self.calibrator_q = AffineCalibrator(self.d_model, self.compiled_q['K_bits'], self.compiled_q['density'], identity=(self.compiled_q.get('backend') in ('1bf16', 'binary', 'int4', 'int8')))
        self.calibrator_k = AffineCalibrator(self.d_model, self.compiled_k['K_bits'], self.compiled_k['density'], identity=(self.compiled_k.get('backend') in ('1bf16', 'binary', 'int4', 'int8')))
        self.calibrator_v = AffineCalibrator(self.d_model, self.compiled_v['K_bits'], self.compiled_v['density'], identity=(self.compiled_v.get('backend') in ('1bf16', 'binary', 'int4', 'int8')))
        self.calibrator_o = AffineCalibrator(self.d_model, self.compiled_o['K_bits'], self.compiled_o['density'], identity=(self.compiled_o.get('backend') in ('1bf16', 'binary', 'int4', 'int8')))
        
        # Pre-calibrate using empirical dummy runs
        dummy_x = np.random.normal(0, 1.0, size=(10, self.d_model)).astype(np.float32)
        dummy_packed = FluxCompiler.binarize(dummy_x, self.compiled_q['K_bits'], self.compiled_q['d_hashes'])
        
        self.calibrator_q.calibrate(FluxCompiler.forward(dummy_packed, self.compiled_q))
        self.calibrator_k.calibrate(FluxCompiler.forward(dummy_packed, self.compiled_k))
        self.calibrator_v.calibrate(FluxCompiler.forward(dummy_packed, self.compiled_v))
        
        # O calibrator expects s_next_bin which has K_bits matching o_proj
        dummy_s = np.random.normal(0, 1.0, size=(10, self.d_model)).astype(np.float32)
        dummy_s_packed = FluxCompiler.binarize(
            dummy_s, 
            self.compiled_o['K_bits'], 
            self.compiled_o['d_hashes']
        )
        self.calibrator_o.calibrate(FluxCompiler.forward(dummy_s_packed, self.compiled_o))

    def forward(self, x_bin, x_dense, s_prev):
        """
        Forward recurrent step of FluxAttention.
        x_bin: Packed binary input of shape (Batch, K_bytes)
        x_dense: Dense FP32 input of shape (Batch, d_model)
        s_prev: Latent state from previous timestep of shape (Batch, d_model)
        """
        # Save inputs for backward pass
        self.last_x_bin = np.ascontiguousarray(x_bin)
        self.last_x_dense = np.ascontiguousarray(x_dense)
        self.last_s_prev = np.ascontiguousarray(s_prev)
        
        # 1. Project Q, K, V
        raw_q = FluxCompiler.forward(x_bin, self.compiled_q)
        raw_k = FluxCompiler.forward(x_bin, self.compiled_k)
        raw_v = FluxCompiler.forward(x_bin, self.compiled_v)
        
        q = self.calibrator_q.apply(raw_q)
        k = self.calibrator_k.apply(raw_k)
        v = self.calibrator_v.apply(raw_v)
        
        self.last_q = q
        self.last_k = k
        self.last_v = v
        
        # 2. Compute Latent State Update Gates using v as local context
        gate_logits = v @ self.w_gate.T
        update_cand = v @ self.w_update.T
        
        self.last_gate_logits = gate_logits
        self.last_update_cand = update_cand
        
        # 3. Apply Multi-Scale Recurrent Latent Update (s_t) with Residual Bypass
        # Clamp gate logits to prevent sigmoid underflow/overflow (exact C++ parity)
        clipped_logits = np.clip(gate_logits, -15.0, 15.0)
        g = 1.0 / (1.0 + np.exp(-clipped_logits))
        update = np.tanh(update_cand)
        
        # Residual Bypass Gate (Skip Connection)
        bypass_logits = v @ self.w_bypass.T
        bypass_gate = 1.0 / (1.0 + np.exp(-np.clip(bypass_logits, -15.0, 15.0)))
        
        core_s_next = g * s_prev + (1.0 - g) * update
        s_next = bypass_gate * x_dense + (1.0 - bypass_gate) * core_s_next
        
        self.last_g = g
        self.last_update = update
        self.last_bypass_gate = bypass_gate
        self.last_s_next = s_next
        
        # 4. Binarize s_next for the o_proj using FluxCompiler binarization
        s_next_bin = FluxCompiler.binarize(
            s_next, 
            self.compiled_o['K_bits'], 
            self.compiled_o['d_hashes']
        )
        self.last_s_next_bin = s_next_bin
        
        # 5. Output Projection
        raw_out = FluxCompiler.forward(s_next_bin, self.compiled_o)
        out = self.calibrator_o.apply(raw_out)
        
        return out, s_next

    def parallel_forward(self, x_bin_seq, x_dense_seq, s_prev_init, is_causal=True):
        """
        TTFT PREFILL ENGINE: Parallel Associative Scan
        is_causal=True: Decoder mode (only looks to the past).
        is_causal=False: Encoder mode (Bidirectional scan for holistic understanding).
        """
        batch_size, L, _ = x_dense_seq.shape
        
        # 1. Project all Q, K, V in parallel across the sequence
        x_bin_flat = x_bin_seq.reshape(-1, x_bin_seq.shape[-1])
        raw_q_flat = FluxCompiler.forward(x_bin_flat, self.compiled_q)
        raw_k_flat = FluxCompiler.forward(x_bin_flat, self.compiled_k)
        raw_v_flat = FluxCompiler.forward(x_bin_flat, self.compiled_v)
        
        q_seq = self.calibrator_q.apply(raw_q_flat).reshape(batch_size, L, self.d_model)
        k_seq = self.calibrator_k.apply(raw_k_flat).reshape(batch_size, L, self.d_model)
        v_seq = self.calibrator_v.apply(raw_v_flat).reshape(batch_size, L, self.d_model)
        
        # 2. Parallel Gates Calculation
        gate_logits_seq = v_seq @ self.w_gate.T
        update_cand_seq = v_seq @ self.w_update.T
        bypass_logits_seq = v_seq @ self.w_bypass.T
        
        clipped_logits_seq = np.clip(gate_logits_seq, -15.0, 15.0)
        g_seq = 1.0 / (1.0 + np.exp(-clipped_logits_seq))
        update_seq = np.tanh(update_cand_seq)
        bypass_gate_seq = 1.0 / (1.0 + np.exp(-np.clip(bypass_logits_seq, -15.0, 15.0)))
        
        # Mathematical mapping of Residual Bypass into standard Associative Scan
        # s_t = bypass * x + (1 - bypass) * [g * s_{t-1} + (1 - g) * u]
        # s_t = [(1 - bypass) * g] * s_{t-1} + [bypass * x + (1 - bypass) * (1 - g) * u]
        g_prime = (1.0 - bypass_gate_seq) * g_seq
        u_prime = bypass_gate_seq * x_dense_seq + (1.0 - bypass_gate_seq) * (1.0 - g_seq) * update_seq
        # Normalize u_prime so the C++ engine (which assumes s_t = g' * s + (1 - g') * u'') works natively
        u_sec_seq = u_prime / (1.0 - g_prime + 1e-9)
        
        # 3. C++ Accelerated Associative Scan for Recurrent State
        import sys
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if base_path not in sys.path:
            sys.path.append(base_path)
        import snn_training_cpp
        
        # Forward scan (C++ Backend)
        s_seq_forward = snn_training_cpp.fast_associative_scan(
            g_prime.astype(np.float32), 
            u_sec_seq.astype(np.float32), 
            s_prev_init.astype(np.float32), 
            False
        )
        s_curr = s_seq_forward[:, -1, :]
        
        # If Encoder Mode (Bidirectional), run backward scan and merge!
        if not is_causal:
            s_seq_bw = snn_training_cpp.fast_associative_scan(
                g_prime.astype(np.float32), 
                u_sec_seq.astype(np.float32), 
                s_prev_init.astype(np.float32), 
                True
            )
            
            # Combine holistic bidirectional outputs
            s_seq = (s_seq_forward + s_seq_bw) * 0.5
            s_curr = (s_seq_forward[:, -1, :] + s_seq_bw[:, 0, :]) * 0.5
        else:
            s_seq = s_seq_forward
            
        # 4. Parallel Output Projection (No Loop Required!)
        s_seq_flat = s_seq.reshape(-1, self.d_model)
        s_seq_bin = FluxCompiler.binarize(s_seq_flat, self.compiled_o['K_bits'], self.compiled_o['d_hashes'])
        raw_out_flat = FluxCompiler.forward(s_seq_bin, self.compiled_o)
        out_seq = self.calibrator_o.apply(raw_out_flat).reshape(batch_size, L, self.d_model)

        return out_seq, s_curr

    def backward(self, grad_out, grad_s_next):
        """
        Backpropagation through time (BPTT) and Straight-Through Estimator (STE) backward pass.
        grad_out: Gradient of shape (Batch, d_model) from the layer above.
        grad_s_next: Gradient of shape (Batch, d_model) backpropagated from the next timestep.
        Returns:
            grad_x_dense: Gradient with respect to the input x_dense (Batch, d_model)
            grad_s_prev: Gradient with respect to the previous recurrent state s_prev (Batch, d_model)
        """
        # 1. Backprop through Output Projection (o_proj) under STE
        scaled_grad_out = grad_out * self.calibrator_o.gamma
        
        # Binarize threshold mask of the recurrent state
        thresh_s = np.mean(np.abs(self.last_s_next), axis=-1, keepdims=True)
        s_next_active = (np.abs(self.last_s_next) > thresh_s).astype(np.float32)
        
        # Accumulate o_proj weight gradients
        self.o_proj_weight_grad += scaled_grad_out.T @ s_next_active
        
        # Gradient flow to s_next from output projection
        grad_s_next_from_o = scaled_grad_out @ self.o_proj_weight
        
        # Total gradient with respect to s_next
        total_grad_s_next = grad_s_next + grad_s_next_from_o
        
        # 2. Backprop through recurrent update equation with Bypass
        # core_s_next = g * s_prev + (1 - g) * update
        # s_next = bypass * x_dense + (1 - bypass) * core_s_next
        core_s_next = self.last_g * self.last_s_prev + (1.0 - self.last_g) * self.last_update
        
        grad_bypass_gate = total_grad_s_next * (self.last_x_dense - core_s_next)
        grad_x_dense_from_bypass = total_grad_s_next * self.last_bypass_gate
        grad_core_s_next = total_grad_s_next * (1.0 - self.last_bypass_gate)
        
        grad_s_prev = grad_core_s_next * self.last_g
        grad_g = grad_core_s_next * (self.last_s_prev - self.last_update)
        grad_update = grad_core_s_next * (1.0 - self.last_g)
        
        # Sigmoid gate backward (d_sigmoid = g * (1 - g))
        grad_gate_logits = grad_g * self.last_g * (1.0 - self.last_g)
        grad_bypass_logits = grad_bypass_gate * self.last_bypass_gate * (1.0 - self.last_bypass_gate)
        
        # Tanh activation backward (d_tanh = 1 - update^2)
        grad_update_cand = grad_update * (1.0 - self.last_update ** 2)
        
        # 3. Backprop through dense gate projection layers
        # gate_logits = v @ w_gate.T
        # update_cand = v @ w_update.T
        # bypass_logits = v @ w_bypass.T
        self.w_gate_grad += grad_gate_logits.T @ self.last_v
        self.w_update_grad += grad_update_cand.T @ self.last_v
        self.w_bypass_grad += grad_bypass_logits.T @ self.last_v
        
        # Gradient with respect to local context v
        grad_v = (grad_gate_logits @ self.w_gate + 
                  grad_update_cand @ self.w_update + 
                  grad_bypass_logits @ self.w_bypass)
        
        # 4. Backprop through compiled Bloom projections Q, K, V
        # Under STE, we propagate grad_v, and 0 for Q and K (since they are not in the recurrences)
        grad_q = np.zeros_like(self.last_q)
        grad_k = np.zeros_like(self.last_k)
        
        scaled_grad_q = grad_q * self.calibrator_q.gamma
        scaled_grad_k = grad_k * self.calibrator_k.gamma
        scaled_grad_v = grad_v * self.calibrator_v.gamma
        
        # Binarize threshold mask of the input token
        thresh = np.mean(np.abs(self.last_x_dense), axis=-1, keepdims=True)
        x_active = (np.abs(self.last_x_dense) > thresh).astype(np.float32)
        
        # Accumulate projection weight gradients
        self.q_proj_weight_grad += scaled_grad_q.T @ x_active
        self.k_proj_weight_grad += scaled_grad_k.T @ x_active
        self.v_proj_weight_grad += scaled_grad_v.T @ x_active
        
        # 5. Compute gradient with respect to x_dense
        grad_x_dense = (
            scaled_grad_q @ self.q_proj_weight +
            scaled_grad_k @ self.k_proj_weight +
            scaled_grad_v @ self.v_proj_weight +
            grad_x_dense_from_bypass
        )
        
        return grad_x_dense, grad_s_prev

    def step_optimizer(self, metabolism, velocity, lr, weight_decay=1e-4, critical_period_mult=1.0):
        """
        Execute metabolic and gradient velocity updates for all 6 weight tensors.
        """
        params = [
            ("q_proj", self.q_proj_weight, self.q_proj_weight_grad),
            ("k_proj", self.k_proj_weight, self.k_proj_weight_grad),
            ("v_proj", self.v_proj_weight, self.v_proj_weight_grad),
            ("o_proj", self.o_proj_weight, self.o_proj_weight_grad),
            ("w_gate", self.w_gate, self.w_gate_grad),
            ("w_update", self.w_update, self.w_update_grad),
            ("w_bypass", self.w_bypass, self.w_bypass_grad)
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
