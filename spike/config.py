import dataclasses

@dataclasses.dataclass
class SNNConfig:
    vocab_size: int = 32000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    hidden_dim: int = 10922  # 8/3 * d_model
    bits_per_param_attn: float = 0.45
    bits_per_param_ffn: float = 0.22
    
    # Creature Biology / Metabolic Parameters
    metabolic_rate: float = 0.001        # Energy consumption per gradient
    regeneration_rate: float = 0.0005    # Energy recovery per idle step
    hunger_growth: float = 0.01          # Hunger increase per idle step
    competition_top_k: float = 0.25      # Top 25% win competition (lateral inhibition)
    hebbian_rate: float = 0.01           # Co-activation learning rate
    hebbian_decay: float = 0.999         # Co-activation decay
    velocity_beta: float = 0.9           # Velocity EMA parameter
    acceleration_beta: float = 0.95      # Acceleration EMA parameter
    jerk_sensitivity: float = 0.5        # Jerk damping factor
    prediction_top_k: int = 4            # Predictive forward candidates
    sleep_interval: int = 10000          # Steps between sleep phases
    sleep_duration: int = 100            # Steps of consolidation
    
    @classmethod
    def SNN_Nano(cls):
        """SNN-Nano config: ~50M parameters, ultra-fast prototyping, uses ~1.5 MB storage."""
        return cls(
            vocab_size=8000, 
            d_model=256, 
            n_layers=4, 
            n_heads=8, 
            hidden_dim=682,
            sleep_interval=1000
        )

    @classmethod
    def SNN_Small(cls):
        """SNN-Small config: ~300M parameters, uses ~9 MB storage."""
        return cls(
            vocab_size=16000, 
            d_model=1024, 
            n_layers=12, 
            n_heads=16, 
            hidden_dim=2730
        )

    @classmethod
    def SNN_Base(cls):
        """SNN-Base config: ~1B parameters, uses ~31 MB storage."""
        return cls(
            vocab_size=32000, 
            d_model=2048, 
            n_layers=24, 
            n_heads=32, 
            hidden_dim=5461
        )

    @classmethod
    def SNN_Large(cls):
        """SNN-Large config: ~7B parameters, uses ~217 MB storage."""
        return cls(
            vocab_size=32000, 
            d_model=4096, 
            n_layers=32, 
            n_heads=32, 
            hidden_dim=10922
        )

    @classmethod
    def SNN_Ultra(cls):
        """SNN-Ultra config: ~70B parameters, uses ~2.2 GB storage."""
        return cls(
            vocab_size=32000, 
            d_model=8192, 
            n_layers=64, 
            n_heads=64, 
            hidden_dim=21845
        )

    @classmethod
    def SNN_ASI(cls):
        """SNN-Core config: 100B parameters, extremely dense, uses ~3.125 GB storage."""
        return cls(
            vocab_size=32000, 
            d_model=10240, 
            n_layers=80, 
            n_heads=80, 
            hidden_dim=27306
        )
