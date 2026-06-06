class CriticalPeriodScheduler:
    """
    Critical Period Scheduler.
    Manages phases of extreme plasticity (IGNITION, RAPID WIRING) and consolidation
    (CONSOLIDATION, MATURATION, SLEEP) to optimize learning dynamics over time.
    """
    def __init__(self, sleep_interval=10000, sleep_duration=100):
        self.sleep_interval = sleep_interval
        self.sleep_duration = sleep_duration
        self.global_step = 0
        
        # Configuration for each phase
        self.phases = {
            'IGNITION': {
                'start_step': 0,
                'end_step': 500,
                'lr_mult': 10.0,
                'rebirth_mult': 3.0,
                'competition_intensity': 0.0,  # No lateral inhibition yet
                'hebbian_mult': 0.1,
            },
            'RAPID_WIRING': {
                'start_step': 500,
                'end_step': 5000,
                'lr_mult': 5.0,
                'rebirth_mult': 2.0,
                'competition_intensity': 1.0,  # Standard competition
                'hebbian_mult': 2.0,            # High Hebbian growth
            },
            'CONSOLIDATION': {
                'start_step': 5000,
                'end_step': 20000,
                'lr_mult': 1.0,
                'rebirth_mult': 1.0,
                'competition_intensity': 1.0,
                'hebbian_mult': 1.0,
            },
            'MATURATION': {
                'start_step': 20000,
                'end_step': 99999999,  # Infinite fallback
                'lr_mult': 0.5,
                'rebirth_mult': 0.1,   # Almost no rebirths
                'competition_intensity': 0.5,
                'hebbian_mult': 0.2,   # Locked in Hebbian
            }
        }

    def step(self):
        """Advance global step counter."""
        self.global_step += 1

    def get_current_phase(self):
        """
        Determine the current training phase and active multipliers based on global_step.
        Supports periodic sleep scheduling.
        """
        step = self.global_step
        
        # 1. Check if in SLEEP phase
        # Sleep occurs every sleep_interval steps, lasting sleep_duration steps.
        # e.g., steps 10000-10100, 20000-20100, etc.
        cycle_pos = step % self.sleep_interval
        if step > 0 and cycle_pos < self.sleep_duration:
            return {
                'phase': 'SLEEP',
                'lr_mult': 0.0,          # Freeze learning updates entirely
                'rebirth_mult': 0.0,
                'competition_intensity': 0.0,
                'hebbian_mult': 0.0,
                'in_sleep': True
            }
            
        # 2. Map step to standard phases
        active_phase = 'MATURATION'
        for phase_name, config in self.phases.items():
            if config['start_step'] <= step < config['end_step']:
                active_phase = phase_name
                break
                
        cfg = self.phases[active_phase]
        return {
            'phase': active_phase,
            'lr_mult': cfg['lr_mult'],
            'rebirth_mult': cfg['rebirth_mult'],
            'competition_intensity': cfg['competition_intensity'],
            'hebbian_mult': cfg['hebbian_mult'],
            'in_sleep': False
        }

    def trigger_adaptive_rewiring(self):
        """
        Adaptive transition: if training plateaus, temporarily re-enter RAPID_WIRING
        by resetting step counter to the beginning of RAPID_WIRING phase.
        """
        self.global_step = 500
