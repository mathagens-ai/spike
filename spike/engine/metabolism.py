import numpy as np
from dataclasses import dataclass

@dataclass
class ParamCell:
    """
    Conceptual representation of a single parameter's metabolic cell.
    (Note: For speed, SNN implements these in vectorized numpy arrays).
    """
    vitality: float      # [0, 2.0] - Parameter health/life force
    energy: float        # [0, 1.0] - Metabolic fuel for learning
    hunger: float        # [0, 1.0] - Demand for gradient signal
    age: int             # Steps lived since last rebirth
    generation: int      # Number of rebirths/evolutions
    velocity: float      # Gradient momentum direction
    acceleration: float  # Gradient acceleration rate
    heritage: float      # Knowledge inherited from ancestors


class ParameterMetabolicState:
    """
    Holds the parallel, vectorized numpy arrays representing the metabolic
    state of a single weight tensor.
    """
    def __init__(self, shape):
        self.shape = shape
        self.size = np.prod(shape)
        
        # Parallel state arrays
        self.vitality = np.ones(shape, dtype=np.float32)
        self.energy = np.ones(shape, dtype=np.float32)
        self.hunger = np.zeros(shape, dtype=np.float32)
        self.age = np.zeros(shape, dtype=np.int32)
        self.generation = np.zeros(shape, dtype=np.int32)
        self.velocity = np.zeros(shape, dtype=np.float32)
        self.acceleration = np.zeros(shape, dtype=np.float32)
        
        # Smooth rebirth phase tracking
        self.cooldown = np.zeros(shape, dtype=np.int32)
        self.lockout = np.zeros(shape, dtype=np.int32)
        self.marked = np.zeros(shape, dtype=bool)


class MetabolismEngine:
    """
    Neural Metabolism Engine (IPP v2) — Replaces old IPP entirely.
    Manages parameters as living cells that burn fuel, compete, and evolve.
    """
    def __init__(self, metabolic_rate=0.001, regeneration_rate=0.0005, hunger_growth=0.01):
        self.metabolic_rate = metabolic_rate
        self.regeneration_rate = regeneration_rate
        self.hunger_growth = hunger_growth
        
        # Metabolism Constants
        self.decay_rate = 0.9997
        self.growth_rate = 0.05
        self.death_threshold = 0.05
        self.rebirth_cap = 0.05  # Max 5% reborn per layer per step
        self.cooldown_steps = 50
        self.lockout_steps = 500
        
        # Registry of managed parameter states
        self.states = {}

    def register_parameter(self, name, param_shape):
        """Register a new parameter tensor by name."""
        state = ParameterMetabolicState(param_shape)
        self.states[name] = state
        return state

    def step(self, name, param_weight, gradient):
        """
        Execute a single metabolic step for a registered parameter.
        Updates energy, hunger, age, and vitality per parameter cell.
        """
        if name not in self.states:
            self.register_parameter(name, param_weight.shape)
            
        state = self.states[name]
        grad_abs = np.abs(gradient)
        active = (grad_abs > 1e-7).astype(np.float32)
        
        # 1. Update Age
        state.age += 1
        
        # 2. Energy Consumption (learning burns fuel)
        state.energy = np.clip(state.energy - grad_abs * self.metabolic_rate, 0.0, 1.0)
        
        # 3. Hunger increase (starving parameters demand gradient signal)
        state.hunger = np.clip(state.hunger * 1.002 + (1.0 - active) * self.hunger_growth, 0.0, 1.0)
        
        # 4. Vitality Update (depends on both energy and hunger balance)
        # Hungry parameters get a learning boost, exhausted ones get dampened
        state.vitality = np.clip(
            state.vitality * self.decay_rate + active * self.growth_rate * (1.0 + state.hunger),
            0.0, 2.0
        )
        
        # 5. Energy Regeneration (regulates slowly during rest)
        state.energy = np.clip(state.energy + self.regeneration_rate * (1.0 - grad_abs), 0.0, 1.0)
        
        # 6. Smooth Rebirth Protocol
        self._execute_rebirth_protocol(name, param_weight, state)

    def _execute_rebirth_protocol(self, name, param_weight, state):
        """
        Runs the 5-phase rebirth protocol: Detection -> Cooldown -> Perturbation -> Warmup -> Lockout
        Ensures dead parameters are revived safely using the Heritage System.
        """
        # Decrement lockouts
        state.lockout = np.maximum(state.lockout - 1, 0)
        
        # Phase 1: DETECTION (vitality below threshold, not marked, and out of lockout)
        newly_dead = (state.vitality < self.death_threshold) & (~state.marked) & (state.lockout == 0)
        state.marked[newly_dead] = True
        state.cooldown[newly_dead] = self.cooldown_steps
        
        # Phase 2: COOLDOWN COUNTDOWN
        state.cooldown[state.marked] -= 1
        
        # Phase 3: PERTURBATION PREP (cooldown finished)
        ready = state.marked & (state.cooldown <= 0)
        
        if np.any(ready):
            # Enforce global rebirth cap per layer to prevent catastrophic shift
            n_ready = np.sum(ready)
            max_rebirth = int(self.rebirth_cap * state.size)
            
            if n_ready > max_rebirth:
                ready_indices = np.where(ready)
                shuffle_idx = np.random.permutation(n_ready)
                pruned_indices = (
                    ready_indices[0][shuffle_idx[:max_rebirth]],
                    ready_indices[1][shuffle_idx[:max_rebirth]]
                )
                ready = np.zeros_like(ready, dtype=bool)
                ready[pruned_indices] = True
                n_ready = max_rebirth
                
            # Phase 4: HERITAGE TRANSFER & PERTURBATION
            # Reborn parameters inherit 20% knowledge from the most successful neighbor in their row (output neuron)
            # Find the best column (highest vitality) in each row
            best_col = np.argmax(state.vitality, axis=-1)  # shape (M,)
            
            # Compute perturbation noise scale (0.1 of standard deviation)
            layer_std = np.std(param_weight) if np.std(param_weight) > 1e-5 else 0.02
            noise = np.random.normal(0, 0.1 * layer_std, size=param_weight.shape).astype(np.float32)
            
            # Apply to weights and reset parameter state for reborn indices
            M, N = param_weight.shape
            for i in range(M):
                row_ready = ready[i]
                if np.any(row_ready):
                    best_w = param_weight[i, best_col[i]]
                    # 20% Heritage + 80% (Original + Noise)
                    param_weight[i, row_ready] = 0.2 * best_w + 0.8 * (param_weight[i, row_ready] + noise[i, row_ready])
            
            # Reset cell states (Warmup + Lockout)
            state.vitality[ready] = 0.5  # Warmup state
            state.energy[ready] = 1.0    # Fully recharged
            state.hunger[ready] = 0.0
            state.age[ready] = 0
            state.generation[ready] += 1
            state.velocity[ready] = 0.0
            state.acceleration[ready] = 0.0
            state.lockout[ready] = self.lockout_steps
            state.marked[ready] = False

    def compete(self, name, param_weight, top_k_ratio=0.25):
        """
        Synaptic Competition (Lateral Inhibition) within output neuron groups.
        Strong connections get a vitality boost, weak ones decay faster.
        """
        if name not in self.states:
            return
            
        state = self.states[name]
        strength = np.abs(param_weight) * state.vitality
        M, N = param_weight.shape
        
        # Calculate Winner-take-all top-k threshold per row
        k = int(max(1, N * top_k_ratio))
        partition_idx = N - k
        
        partition = np.partition(strength, partition_idx, axis=-1)
        thresholds = partition[:, partition_idx][:, np.newaxis]
        
        winners = strength >= thresholds
        
        # Winners get vitality boost
        state.vitality[winners] = np.minimum(state.vitality[winners] * 1.01, 2.0)
        # Losers get accelerated decay
        state.vitality[~winners] = np.maximum(state.vitality[~winners] * 0.999, 0.0)

    def sleep(self, custom_defragment_fn=None):
        """
        Consolidation phase (SLEEP).
        Defragments memory states, runs IPP census, and runs a global rebirth cleanup.
        """
        for name, state in self.states.items():
            # 1. Global rebirth for all remaining marked cells ignoring lockout
            marked_count = np.sum(state.marked)
            if marked_count > 0:
                # Force reset marked parameters
                state.vitality[state.marked] = 0.5
                state.energy[state.marked] = 1.0
                state.hunger[state.marked] = 0.0
                state.cooldown[state.marked] = 0
                state.lockout[state.marked] = self.lockout_steps
                state.marked[state.marked] = False
            
            # 2. Defragment state (reset numerical accumulation drift in velocity & acceleration)
            drift_velocity = np.mean(np.abs(state.velocity))
            drift_accel = np.mean(np.abs(state.acceleration))
            state.velocity *= 0.95
            state.acceleration *= 0.95
            
        if custom_defragment_fn is not None:
            custom_defragment_fn()

    def get_health_report(self):
        """
        Calculate global creature health metrics.
        Returns IPP score, energy distribution, average hunger, and competition pressure.
        """
        total_cells = 0
        alive_cells = 0
        total_energy = 0.0
        total_hunger = 0.0
        
        for name, state in self.states.items():
            total_cells += state.size
            alive_cells += np.sum(state.vitality > 0.5)
            total_energy += np.sum(state.energy)
            total_hunger += np.sum(state.hunger)
            
        ipp_score = (alive_cells / total_cells * 100.0) if total_cells > 0 else 100.0
        avg_energy = (total_energy / total_cells) if total_cells > 0 else 1.0
        avg_hunger = (total_hunger / total_cells) if total_cells > 0 else 0.0
        
        return {
            'ipp_score': ipp_score,
            'avg_energy': avg_energy,
            'avg_hunger': avg_hunger,
            'total_cells': total_cells,
        }

class GradientVelocity:
    """
    Biological Optimizer. Acts like AdamW but modified by Vitality.
    """
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}

    def update_and_get_delta(self, name, grad, lr, vitality, critical_period_mult=1.0):
        if name not in self.m:
            self.m[name] = np.zeros_like(grad)
            self.v[name] = np.zeros_like(grad)
            self.t[name] = 0

        self.t[name] += 1
        t = self.t[name]

        self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad ** 2)

        m_hat = self.m[name] / (1.0 - self.beta1 ** t)
        v_hat = self.v[name] / (1.0 - self.beta2 ** t)

        # The biological modification: learning rate is multiplied by the parameter's vitality
        delta = lr * critical_period_mult * vitality * m_hat / (np.sqrt(v_hat) + self.eps)
        return delta
