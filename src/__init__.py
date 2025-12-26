"""
VOID_RUNNER - Curiosity-Driven Exploration Research Framework
==============================================================
A comprehensive toolkit for intrinsic motivation in reinforcement learning.

Implements state-of-the-art exploration methods:
- RND (Random Network Distillation)
- DRND (Distributional RND - ICML 2024)
- NGU (Never Give Up - episodic + lifelong curiosity)
- ICM (Intrinsic Curiosity Module)
- Empowerment-based exploration
- Count-based methods (SimHash, pseudo-counts)
- Go-Explore style archives
"""

from .networks import PolicyNetwork, RNDTargetNetwork, RNDPredictorNetwork
from .curiosity import CuriosityModule, ICMModule
from .agent import VoidRunnerAgent
from .empowerment import EmpowermentModule, CausalCuriosity
from .episodic import EpisodicCuriosityModule, CuriosityMemoryBank
from .compound import CompoundCuriosity, AdversarialCuriosity, CuriosityType
from .metrics import ExplorationMetrics, CompressionMetrics, InformationGain
from .drnd import DRND, NGUCombinedCuriosity, NGUEpisodicMemory
from .counting import SimHashCounter, StateActionCounter, PseudoCountModule, GoExploreArchive

__all__ = [
    # Core
    "PolicyNetwork",
    "RNDTargetNetwork",
    "RNDPredictorNetwork",
    "CuriosityModule",
    "ICMModule",
    "VoidRunnerAgent",
    # State-of-the-Art (2024)
    "DRND",
    "NGUCombinedCuriosity",
    "NGUEpisodicMemory",
    # Count-Based
    "SimHashCounter",
    "StateActionCounter",
    "PseudoCountModule",
    "GoExploreArchive",
    # Advanced Curiosity
    "EmpowermentModule",
    "CausalCuriosity",
    "EpisodicCuriosityModule",
    "CuriosityMemoryBank",
    "CompoundCuriosity",
    "AdversarialCuriosity",
    "CuriosityType",
    # Metrics
    "ExplorationMetrics",
    "CompressionMetrics",
    "InformationGain",
]
