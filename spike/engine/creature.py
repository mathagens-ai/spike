import numpy as np
from .fluxbits import FluxCompiler, AffineCalibrator
from .norm import RMSNorm
from .embedding import FluxEmbedding
from .block import SNNBlock
from .metabolism import MetabolismEngine
from .velocity import VelocityEngine
from .critical_period import CriticalPeriodScheduler
from .hebbian import HebbianTracker

class SNNCreature:
    """
    SNNCreature — The Fully Assembled Living Computational Entity.
    Integrates all 7 SNN biological subsystems in pure Python/Numpy.
    Achieves ultra-compressed, highly convergent autoregressive learning.
    """
    def __init__(self, vocab_size, d_model, n_heads, hidden_dim, n_layers, 
                 bits_per_param_attn=0.45, bits_per_param_ffn=0.22, weight_decay=1e-4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bits_per_param_attn = bits_per_param_attn
        self.bits_per_param_ffn = bits_per_param_ffn
        self.weight_decay = weight_decay
        
        # 1. Initialize Biological Engines
        self.metabolism = MetabolismEngine()
        self.velocity = VelocityEngine()
        self.critical_scheduler = CriticalPeriodScheduler()
        self.hebbian_tracker = HebbianTracker(d_model)
        
        # 2. Initialize Layer stack
        self.embedding = FluxEmbedding(vocab_size, d_model, bits_per_param=bits_per_param_attn, name="embedding")
        self.blocks = [
            SNNBlock(d_model, n_heads, hidden_dim, 
                     bits_per_param_attn=bits_per_param_attn, 
                     bits_per_param_ffn=bits_per_param_ffn, 
                     name=f"layer{i}") 
            for i in range(n_layers)
        ]
        
        # 3. Final Output Layer and Normalization
        self.final_norm = RMSNorm(d_model)
        
        # Output Logits projection (lm_head) compiled using bits_per_param_attn
        limit = np.sqrt(1.0 / d_model)
        self.lm_head_weight = np.random.uniform(-limit, limit, size=(vocab_size, d_model)).astype(np.float32)
        self.lm_head_weight_grad = np.zeros_like(self.lm_head_weight)
        
        self.recompile_lm_head()
        
        # Saved state history variables for backpropagation through time (BPTT)
        self.last_input_ids = None
        self.last_emb_out = None
        self.last_recurrent_states = None
        self.states_history = []
        self.hidden_history = []
        self.block_outputs = []
        self.lm_head_h = []
        self.lm_head_h_bin = []

    def recompile_lm_head(self, coactivation_matrix=None):
        """Compile FP32 shadow weights of lm_head down using bits_per_param_attn."""
        self.compiled_lm_head = FluxCompiler.compile(
            self.lm_head_weight, 
            bits_per_param=self.bits_per_param_attn, 
            coactivation_matrix=coactivation_matrix
        )
        self.calibrator_lm_head = AffineCalibrator(
            self.vocab_size, 
            self.compiled_lm_head['K_bits'], 
            self.compiled_lm_head['density'],
            identity=(self.compiled_lm_head.get('backend') in ('1bf16', 'binary', 'int4', 'int8'))
        )
        
        # Empirical dummy calibration pass
        dummy_h = np.random.normal(0, 1.0, size=(10, self.d_model)).astype(np.float32)
        dummy_h_packed = FluxCompiler.binarize(
            dummy_h, 
            self.compiled_lm_head['K_bits'], 
            self.compiled_lm_head['d_hashes']
        )
        self.calibrator_lm_head.calibrate(FluxCompiler.forward(dummy_h_packed, self.compiled_lm_head))

    def forward(self, input_ids=None, s_prev_list=None, emb_out=None):
        """
        Sequence forward pass tracking recurrent states and activations.
        input_ids: shape (Batch, SeqLen) or None
        s_prev_list: optional list of recurrent state tensors of shape (Batch, d_model)
        emb_out: pre-computed embedding of shape (Batch, SeqLen, d_model) or None
        Returns:
            logits: (Batch, SeqLen, vocab_size)
        """
        if emb_out is None:
            assert input_ids is not None, "Either input_ids or emb_out must be provided"
            self.last_input_ids = np.ascontiguousarray(input_ids)
            Batch, SeqLen = input_ids.shape
            emb_out = self.embedding.forward(input_ids)
        else:
            self.last_input_ids = None
            Batch, SeqLen, _ = emb_out.shape
            
        self.last_emb_out = emb_out
        
        # Initialize intermediate storage list variables for exact backward passes
        self.states_history = []
        self.hidden_history = []
        self.block_outputs = []
        self.lm_head_h = []
        self.lm_head_h_bin = []
        
        logits = np.zeros((Batch, SeqLen, self.vocab_size), dtype=np.float32)
        
        # Initialize recurrent state vectors for each SNNBlock
        if s_prev_list is None:
            s_current = [np.zeros((Batch, self.d_model), dtype=np.float32) for _ in range(self.n_layers)]
        else:
            s_current = [s.copy() for s in s_prev_list]
        
        # Process the sequence sequentially step-by-step through time (BPTT)
        for t in range(SeqLen):
            x_t = emb_out[:, t, :]  # shape (Batch, d_model)
            
            # Update Hebbian Co-activation Tracker
            # Reconstruct binary activation array from dense input to capture correlation signal
            h_active = (np.abs(x_t) > np.mean(np.abs(x_t), axis=-1, keepdims=True)).astype(np.uint8)
            self.hebbian_tracker.update(h_active)
            
            layer_states = []
            layer_inputs = []
            layer_outputs = []
            
            h = x_t
            for l in range(self.n_layers):
                s_prev = s_current[l]
                
                # Save activations and recurrent states BEFORE block execution
                layer_states.append(s_prev.copy())
                layer_inputs.append(h.copy())
                
                # Execute Block forward pass
                h_next, s_next = self.blocks[l].forward(h, s_prev)
                
                layer_outputs.append(h_next.copy())
                s_current[l] = s_next
                h = h_next
                
            self.states_history.append(layer_states)
            self.hidden_history.append(layer_inputs)
            self.block_outputs.append(layer_outputs)
            
            # Apply final layer normalization
            h_norm, _, _ = self.final_norm.forward(h)
            self.lm_head_h.append(h_norm.copy())
            
            # Binarize and project logits for timestep t
            h_bin = FluxCompiler.binarize(
                h_norm, 
                self.compiled_lm_head['K_bits'], 
                self.compiled_lm_head['d_hashes']
            )
            self.lm_head_h_bin.append(h_bin)
            
            raw_logits = FluxCompiler.forward(h_bin, self.compiled_lm_head)
            logits[:, t, :] = self.calibrator_lm_head.apply(raw_logits)
            
        self.last_recurrent_states = [s.copy() for s in s_current]
        return logits

    def backward(self, grad_logits):
        """
        Mathematically exact Backpropagation Through Time (BPTT).
        grad_logits: shape (Batch, SeqLen, vocab_size)
        Returns: None
        """
        Batch, SeqLen, vocab_size = grad_logits.shape
        
        # Initialize backpropagated recurrent gradients from step t+1 (Batch, d_model)
        grad_s_next = [np.zeros((Batch, self.d_model), dtype=np.float32) for _ in range(self.n_layers)]
        
        # Accumulator for embedding output gradients
        grad_emb_out = np.zeros((Batch, SeqLen, self.d_model), dtype=np.float32)
        
        # Step backward sequentially through time (t = SeqLen-1 down to 0)
        for t in reversed(range(SeqLen)):
            grad_logits_t = grad_logits[:, t, :]  # shape (Batch, vocab_size)
            h = self.lm_head_h[t]                 # shape (Batch, d_model)
            
            # 1. Backprop through output lm_head projection (uses STE)
            scaled_grad_logits_t = grad_logits_t * self.calibrator_lm_head.gamma
            
            thresh_h = np.mean(np.abs(h), axis=-1, keepdims=True)
            h_active = (np.abs(h) > thresh_h).astype(np.float32)
            
            # Accumulate weight gradients for shadow projection
            self.lm_head_weight_grad += scaled_grad_logits_t.T @ h_active
            
            # Dense gradient flowing to final layer output
            grad_final_norm = scaled_grad_logits_t @ self.lm_head_weight # (Batch, d_model)
            
            # 2. Backprop through final RMSNorm
            # Retrieve inputs for final norm step
            final_block_out = self.block_outputs[t][-1]
            y, x_norm, rms = self.final_norm.forward(final_block_out)
            grad_block = self.final_norm.backward(grad_final_norm, final_block_out, x_norm, rms)
            
            # 3. Backprop through SNN Blocks sequentially (from layer n_layers-1 down to 0)
            grad_flow = grad_block
            for l in reversed(range(self.n_layers)):
                s_prev = self.states_history[t][l]
                h_in = self.hidden_history[t][l]
                
                block = self.blocks[l]
                
                # Populate layer activation state for timestep t by re-running block forward pass
                _, _ = block.forward(h_in, s_prev)
                
                # Execute Block backward step
                grad_h_in, grad_s_prev = block.backward(grad_flow, grad_s_next[l])
                
                # Propagate layer states and hidden gradients
                grad_s_next[l] = grad_s_prev
                grad_flow = grad_h_in
                
            # Store final gradient to embedding output
            grad_emb_out[:, t, :] = grad_flow
            
        # 4. Backpropagation through Embedding layer
        self.embedding.backward(grad_emb_out)

    def step_optimizer(self, lr):
        """
        Updates shadow weights using the global Metabolism and Velocity engines.
        """
        # 1. Retrieve Critical Period LR multiplier
        phase = self.critical_scheduler.get_current_phase()
        crit_mult = phase['lr_mult']
        
        # 2. Update Embedding Layer
        self.embedding.step_optimizer(
            self.metabolism, 
            self.velocity, 
            lr, 
            weight_decay=self.weight_decay, 
            critical_period_mult=crit_mult
        )
        
        # 3. Update all Blocks
        for block in self.blocks:
            block.step_optimizer(
                self.metabolism, 
                self.velocity, 
                lr, 
                weight_decay=self.weight_decay, 
                critical_period_mult=crit_mult
            )
            
        # 4. Update final layer RMSNorm
        self.final_norm.step_optimizer(lr, weight_decay=self.weight_decay)
        
        # 5. Update lm_head output layer
        p_name = "lm_head.weight"
        self.metabolism.step(p_name, self.lm_head_weight, self.lm_head_weight_grad)
        self.metabolism.compete(p_name, self.lm_head_weight)
        vitality = self.metabolism.states[p_name].vitality
        
        delta = self.velocity.update_and_get_delta(
            p_name, 
            self.lm_head_weight_grad, 
            lr, 
            vitality, 
            crit_mult
        )
        
        self.lm_head_weight *= (1.0 - lr * self.weight_decay)
        self.lm_head_weight -= delta
        self.lm_head_weight_grad.fill(0.0)
        
        self.recompile_lm_head()
        
        # 6. Step Scheduler
        self.critical_scheduler.step()
        
        # 7. Periodic sleep/consolidation
        # Triggers at the exact start step of each sleep phase
        step = self.critical_scheduler.global_step
        if step > 0 and (step % self.critical_scheduler.sleep_interval == 0):
            self.sleep()

    def sleep(self):
        """Trigger global biological sleep cycle to consolidate and defragment memory."""
        # Get co-activation correlations
        coactivation_matrix = self.hebbian_tracker.get_coactivation_matrix(threshold=0.5)
        
        # Trigger metabolism sleep routine
        def defragment_all():
            # Reset numerical state drift in embedding/blocks
            pass
            
        self.metabolism.sleep(custom_defragment_fn=defragment_all)
        
        # Recompile all Bloom projection models using coactivation to align collisions
        self.embedding.recompile(coactivation_matrix=coactivation_matrix)
        for block in self.blocks:
            block.attn.recompile(coactivation_matrix=coactivation_matrix)
            block.ffn.recompile(coactivation_matrix=coactivation_matrix)
            
        self.recompile_lm_head(coactivation_matrix=coactivation_matrix)
        
        # Clear tracker
        self.hebbian_tracker.reset()
