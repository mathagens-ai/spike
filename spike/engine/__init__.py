"""
SNN Engine — The Living Computational Creature

All modules use REAL mathematics. Zero fake data. Zero simulation.
Every computation is grounded in FluxBits AND+POPCOUNT, real gradient
tracking, and genuine biological-inspired parameter lifecycle management.

Author: Aryan / MecanLabs
"""

from .fluxbits import FluxCompiler
from .metabolism import MetabolismEngine, ParamCell
from .velocity import VelocityEngine
from .hebbian import HebbianTracker
from .critical_period import CriticalPeriodScheduler
from .latent_state import MultiScaleLatentState
from .norm import RMSNorm
from .embedding import FluxEmbedding
from .attention import FluxAttention
from .ffn import FluxFFN
from .block import SNNBlock
from .creature import SNNCreature
from .trainer import CreatureTrainer
from .inference import SNNInferenceEngine
from .generation import DecodingEngine

__all__ = [
    'FluxCompiler',
    'MetabolismEngine', 'ParamCell',
    'VelocityEngine',
    'HebbianTracker',
    'CriticalPeriodScheduler',
    'MultiScaleLatentState',
    'RMSNorm',
    'FluxEmbedding',
    'FluxAttention',
    'FluxFFN',
    'SNNBlock',
    'SNNCreature',
    'CreatureTrainer',
    'SNNInferenceEngine',
    'DecodingEngine',
]

