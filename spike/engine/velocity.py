import numpy as np

class ParameterVelocityState:
    """
    Holds parallel numpy arrays representing the derivatives of gradient flow
    for a single weight tensor.
    """
    def __init__(self, shape):
        self.shape = shape
        self.velocity = np.zeros(shape, dtype=np.float32)
        self.acceleration = np.zeros(shape, dtype=np.float32)
        self.prev_velocity = np.zeros(shape, dtype=np.float32)
        self.prev_acceleration = np.zeros(shape, dtype=np.float32)


class VelocityEngine:
    """
    Gradient Velocity and Acceleration Engine.
    Not just gradient magnitude — tracks acceleration and jerk to predict where
    gradients are heading and dampens erratic step paths.
    """
    def __init__(self, beta_v=0.9, beta_a=0.95, jerk_sensitivity=0.5, alpha_pred=0.1):
        self.beta_v = beta_v
        self.beta_a = beta_a
        self.jerk_sensitivity = jerk_sensitivity
        self.alpha_pred = alpha_pred
        
        # Registry of managed parameter states
        self.states = {}

    def register_parameter(self, name, param_shape):
        """Register a new parameter tensor by name."""
        state = ParameterVelocityState(param_shape)
        self.states[name] = state
        return state

    def update_and_get_delta(self, name, gradient, lr, vitality_scale, critical_period_mult=1.0):
        """
        Compute the physics-based predicted gradient update step for a parameter.
        Returns the final weight update delta (Δw).
        """
        if name not in self.states:
            self.register_parameter(name, gradient.shape)
            
        state = self.states[name]
        
        # 1. Gradient Velocity (1st derivative)
        state.prev_velocity = state.velocity.copy()
        state.velocity = self.beta_v * state.velocity + (1.0 - self.beta_v) * gradient
        
        # 2. Gradient Acceleration (2nd derivative)
        state.prev_acceleration = state.acceleration.copy()
        vel_change = state.velocity - state.prev_velocity
        state.acceleration = self.beta_a * state.acceleration + (1.0 - self.beta_a) * vel_change
        
        # 3. Jerk (3rd derivative: rate of acceleration change)
        jerk = state.acceleration - state.prev_acceleration
        
        # 4. Predictive look-ahead gradient
        predicted_gradient = state.velocity + self.alpha_pred * state.acceleration
        
        # 5. Adaptive Damping (slows down learning if path is erratic/high jerk)
        damping = 1.0 / (1.0 + np.abs(jerk) * self.jerk_sensitivity)
        
        # 6. Effective per-parameter learning rate
        effective_lr = lr * damping * vitality_scale * critical_period_mult
        
        # 7. Final predicted step delta
        delta = effective_lr * predicted_gradient
        
        return delta

    def get_stats(self, name):
        """Compute statistics of gradient velocity and acceleration for diagnostics."""
        if name not in self.states:
            return {}
            
        state = self.states[name]
        mean_vel = np.mean(np.abs(state.velocity))
        mean_accel = np.mean(np.abs(state.acceleration))
        
        return {
            'mean_velocity': float(mean_vel),
            'mean_acceleration': float(mean_accel),
        }
