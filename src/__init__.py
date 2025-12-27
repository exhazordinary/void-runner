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
from .byol_explore import BYOLExplore, WorldModelCuriosity
from .skills import DIAYN, DADS, HierarchicalExplorer
from .multiagent import MultiAgentCuriosity, EMC, CERMICCuriosity, CompetitiveCuriosity
from .go_explore import GoExplore, GoExploreTrainer, CellArchive, StateEncoder
from .llm_curiosity import (
    ELLMExplorer, EurekaRewardGenerator, LanguageConditionedCuriosity,
    MockLLM, GridWorldDescriber, RoboticsDescriber
)
from .adversarial_curiosity import (
    AdversarialCuriosity, SelfPlayCuriosity, CuriosityBottleneck,
    StateDiscriminator, StateGenerator
)
from .curiosity_diagnostics import (
    CuriosityCollapseDetector, CuriosityHealthMonitor, CuriositySnapshot,
    instrument_curiosity_module
)
from .meta_curiosity import (
    CuriosityTransfer, AdaptiveCuriosity, CuriosityEnsemble,
    CuriosityScheduler, UniversalCuriosityEncoder
)

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
    "BYOLExplore",
    "WorldModelCuriosity",
    # Count-Based
    "SimHashCounter",
    "StateActionCounter",
    "PseudoCountModule",
    "GoExploreArchive",
    # Skill Discovery
    "DIAYN",
    "DADS",
    "HierarchicalExplorer",
    # Multi-Agent
    "MultiAgentCuriosity",
    "EMC",
    "CERMICCuriosity",
    "CompetitiveCuriosity",
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
    # Go-Explore
    "GoExplore",
    "GoExploreTrainer",
    "CellArchive",
    "StateEncoder",
    # LLM-Based Curiosity
    "ELLMExplorer",
    "EurekaRewardGenerator",
    "LanguageConditionedCuriosity",
    "MockLLM",
    "GridWorldDescriber",
    "RoboticsDescriber",
    # Adversarial Curiosity
    "AdversarialCuriosity",
    "SelfPlayCuriosity",
    "CuriosityBottleneck",
    "StateDiscriminator",
    "StateGenerator",
    # Diagnostics
    "CuriosityCollapseDetector",
    "CuriosityHealthMonitor",
    "CuriositySnapshot",
    "instrument_curiosity_module",
    # Meta-Curiosity
    "CuriosityTransfer",
    "AdaptiveCuriosity",
    "CuriosityEnsemble",
    "CuriosityScheduler",
    "UniversalCuriosityEncoder",
]
