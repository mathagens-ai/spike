import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class CreatureTrainer:
    """
    CreatureTrainer — Manages the training lifecycle of an SNNCreature.
    Computes exact, original Cross-Entropy loss and analytic token-level gradients,
    executes BPTT backward updates, and reports metabolic creature diagnostics.
    """
    def __init__(self, creature, lr=0.005, weight_decay=1e-4):
        self.creature = creature
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Diagnostics tracking
        self.loss_history = []
        self.step_times = []

    def compute_cross_entropy(self, logits, targets):
        """
        Compute mathematically exact categorical cross entropy loss and analytic gradients.
        logits: shape (Batch, SeqLen, vocab_size)
        targets: shape (Batch, SeqLen) - Integer class labels
        Returns:
            loss: scalar FP32 cross entropy loss
            grad_logits: analytic gradient with respect to logits (Batch, SeqLen, vocab_size)
        """
        Batch, SeqLen, vocab_size = logits.shape
        total_tokens = Batch * SeqLen
        
        # Flatten for vectorized computation
        flat_logits = logits.reshape(total_tokens, vocab_size).astype(np.float64)
        flat_targets = targets.ravel()
        
        # 1. Log-Sum-Exp trick for numerical stability (exact C++ parity)
        max_logits = np.max(flat_logits, axis=-1, keepdims=True)
        shifted = flat_logits - max_logits
        exp_logits = np.exp(shifted)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Compute probabilities
        probs = exp_logits / sum_exp
        
        # 2. Compute Loss
        log_probs = shifted - np.log(sum_exp)
        loss = -np.mean(log_probs[np.arange(total_tokens), flat_targets])
        
        # 3. Compute exact analytic gradient: dL/dz = (prob - 1_y) / total_tokens
        probs[np.arange(total_tokens), flat_targets] -= 1.0
        grad_logits = (probs / total_tokens).astype(np.float32)
        
        # Reshape gradient back to 3D tensor
        grad_logits = grad_logits.reshape(Batch, SeqLen, vocab_size)
        
        return float(loss), grad_logits

    def train_step(self, batch_ids):
        """
        Execute a single forward-backward-update training step on a batch of token IDs.
        batch_ids: Integer token array of shape (Batch, SeqLen + 1)
        Returns:
            loss: scalar loss value
            step_time: time in seconds
        """
        t_start = time.time()
        
        # Autoregressive split: predict next token
        x = batch_ids[:, :-1]  # shape (Batch, SeqLen)
        y = batch_ids[:, 1:]   # shape (Batch, SeqLen)
        
        # 1. Forward pass
        logits = self.creature.forward(x)
        
        # 2. Compute Loss and analytic gradients
        loss, grad_logits = self.compute_cross_entropy(logits, y)
        
        # 3. Backward Pass (BPTT + STE)
        self.creature.backward(grad_logits)
        
        # 4. Metabolic and Velocity optimizer update step
        self.creature.step_optimizer(self.lr)
        
        t_duration = time.time() - t_start
        self.loss_history.append(loss)
        self.step_times.append(t_duration)
        
        return loss, t_duration

    def train_epoch(self, dataset, batch_size=8, shuffle=True):
        """
        Train the creature for a single epoch over the provided dataset.
        dataset: Integer token array of shape (NumSamples, SeqLen + 1)
        """
        num_samples = len(dataset)
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)
            
        epoch_losses = []
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            batch_ids = dataset[batch_idx]
            
            # Execute single step
            loss, duration = self.train_step(batch_ids)
            epoch_losses.append(loss)
            
            # Log diagnostics after every step
            step = len(self.loss_history)
            if step % 10 == 0 or b == num_batches - 1:
                # Retrieve creature metabolism health metrics
                health = self.creature.metabolism.get_health_report()
                phase = self.creature.critical_scheduler.get_current_phase()
                
                logger.info(
                    f"Batch {b+1}/{num_batches} | Step {step} | "
                    f"Loss: {loss:.4f} | Perplexity: {np.exp(min(loss, 15.0)):.2f} | "
                    f"Phase: {phase['phase']} (LR mult: {phase['lr_mult']:.1f}x) | "
                    f"IPP: {health['ipp_score']:.2f}% | "
                    f"Avg Energy: {health['avg_energy']:.4f} | "
                    f"Time: {duration*1000.0:.1f}ms"
                )
                
        return np.mean(epoch_losses)
