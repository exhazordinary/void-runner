# VOID_RUNNER

```
██╗   ██╗ ██████╗ ██╗██████╗     ██████╗ ██╗   ██╗███╗   ██╗███╗   ██╗███████╗██████╗
██║   ██║██╔═══██╗██║██╔══██╗    ██╔══██╗██║   ██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
██║   ██║██║   ██║██║██║  ██║    ██████╔╝██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝
╚██╗ ██╔╝██║   ██║██║██║  ██║    ██╔══██╗██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗
 ╚████╔╝ ╚██████╔╝██║██████╔╝    ██║  ██║╚██████╔╝██║ ╚████║██║ ╚████║███████╗██║  ██║
  ╚═══╝   ╚═════╝ ╚═╝╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
                     Curiosity-Driven Exploration Research Framework
```

> *"The reward of curiosity is not just what you find, but what you become while searching."*

A **comprehensive research framework** for intrinsic motivation in reinforcement learning. Implements **20+ state-of-the-art exploration methods** from ICML 2024, NeurIPS, and ICLR, with original extensions including LLM-guided exploration and multi-agent curiosity.

## What's Implemented

### Prediction-Based Curiosity

| Method | Paper | Year | Status |
|--------|-------|------|--------|
| **RND** | [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) | 2018 | Complete |
| **ICM** | [Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363) | 2017 | Complete |
| **DRND** | [Distributional RND](https://arxiv.org/abs/2401.09750) | ICML 2024 | Complete |
| **NGU** | [Never Give Up](https://arxiv.org/abs/2002.06038) | 2020 | Complete |
| **BYOL-Explore** | [Bootstrap Your Own Latent](https://arxiv.org/abs/2206.08332) | 2022 | Complete |

### Count-Based Methods

| Method | Paper | Year | Status |
|--------|-------|------|--------|
| **SimHash** | [#Exploration](https://arxiv.org/abs/1611.04717) | 2017 | Complete |
| **VCSAP** | [State-Action Counting](https://www.sciencedirect.com/science/article/abs/pii/S089360802400981X) | 2024 | Complete |
| **Pseudo-Count** | [Unifying Count-Based](https://arxiv.org/abs/1606.01868) | 2016 | Complete |
| **Go-Explore** | [First Return Then Explore](https://arxiv.org/abs/1901.10995) | 2021 | Complete |

### Skill Discovery (Hierarchical RL)

| Method | Paper | Year | Status |
|--------|-------|------|--------|
| **DIAYN** | [Diversity is All You Need](https://arxiv.org/abs/1802.06070) | 2018 | Complete |
| **DADS** | [Dynamics-Aware Skill Discovery](https://arxiv.org/abs/1907.01657) | 2019 | Complete |
| **Hierarchical** | Two-level skill-based exploration | - | Complete |

### Multi-Agent Curiosity

| Method | Description | Status |
|--------|-------------|--------|
| **MultiAgentCuriosity** | Individual + joint curiosity signals | Complete |
| **EMC** | Episodic Multi-agent with Q-prediction | Complete |
| **CERMIC** | Calibrated curiosity (filters peer noise) | Complete |
| **CompetitiveCuriosity** | Agents compete to discover states first | Complete |

### Information-Theoretic Methods

| Method | Paper | Status |
|--------|-------|--------|
| **Empowerment** | [Variational Info Max](https://arxiv.org/abs/1509.08731) | Complete |
| **Causal Curiosity** | NeurIPS 2024 | Complete |
| **Episodic Memory** | k-NN based novelty | Complete |

### LLM-Based Exploration (2024 Frontier)

| Method | Inspiration | Status |
|--------|-------------|--------|
| **ELLMExplorer** | ELLM - LLM-suggested goals | Complete |
| **EurekaRewardGenerator** | [Eureka](https://arxiv.org/abs/2310.12931) - Evolutionary LLM rewards | Complete |
| **LanguageConditionedCuriosity** | Language abstractions | Complete |

## Key Research Insights

### 1. The Bonus Inconsistency Problem (ICML 2024)

Standard RND suffers from **bonus inconsistency**:
- **Initial inconsistency**: Uneven bonus distribution at training start
- **Final inconsistency**: Poor discrimination between visited/unvisited states

**DRND Solution**: Use N random networks instead of 1:
```python
from src import DRND
curiosity = DRND(obs_dim=4, n_networks=10)
bonus = curiosity.compute_intrinsic_reward(observation)
```

### 2. Two Timescales of Novelty (NGU)

- **Episodic** (fast): k-NN in embedding space, resets each episode
- **Lifelong** (slow): RND/DRND, persists across training

```python
from src import NGUCombinedCuriosity
ngu = NGUCombinedCuriosity(obs_dim=4, episodic_memory_size=5000)
bonus = ngu.compute_combined_bonus(obs)
ngu.reset_episode()  # At episode boundaries
```

### 3. Self-Supervised Exploration (BYOL-Explore)

Same loss for world model AND curiosity signal:
```python
from src import BYOLExplore
byol = BYOLExplore(obs_dim=4, action_dim=4)
bonus = byol.compute_intrinsic_reward(obs, action, next_obs)
```

### 4. Skills as Temporal Abstraction (DIAYN)

Learn diverse skills WITHOUT external reward:
```python
from src import DIAYN
diayn = DIAYN(obs_dim=4, action_dim=4, n_skills=10)
skill = diayn.sample_skill()
action, _ = diayn.get_action(obs, skill)
intrinsic = diayn.compute_intrinsic_reward(obs_tensor, skill_tensor)
```

### 5. LLMs Encode Exploration Priors

Language models know what's "interesting":
```python
from src import ELLMExplorer, MockLLM, GridWorldDescriber

llm = MockLLM()  # Replace with GPT-4/Claude for production
ellm = ELLMExplorer(
    state_describer=GridWorldDescriber(),
    llm=llm,
)
reward, info = ellm.compute_intrinsic_reward(state_dict)
print(f"Matched goal: {info['matched_goal']}")
```

## Benchmark Results

### Hard Exploration Environments

```bash
python benchmark_hard.py --methods none rnd drnd simhash --episodes 100
```

| Environment | Challenge | Best Method |
|-------------|-----------|-------------|
| **DeceptiveMaze** | Local optima trap | RND |
| **KeyDoor** | Temporal abstraction | NGU |
| **StochasticMaze** | Noisy transitions | DRND |
| **MultiGoal** | Breadth of exploration | SimHash |
| **MontezumaLite** | Full hierarchical | Go-Explore |

## Quick Start

```bash
# Install
git clone https://github.com/exhazordinary/void-runner.git
cd void-runner
pip install -r requirements.txt

# Basic training (RND)
python train.py --env SparseMaze-v0 --steps 100000

# Benchmark all methods
python benchmark_hard.py --methods rnd drnd ngu byol simhash go_explore --episodes 200
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       VOID_RUNNER FRAMEWORK                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                 INTRINSIC MOTIVATION MODULES                      │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                    │   │
│  │  Prediction     Count-Based    Information    Skill Discovery    │   │
│  │  ┌─────┐        ┌───────┐      ┌─────────┐    ┌───────┐          │   │
│  │  │ RND │        │SimHash│      │Empower- │    │ DIAYN │          │   │
│  │  │DRND │        │VCSAP  │      │ment     │    │ DADS  │          │   │
│  │  │ NGU │        │Pseudo │      │Causal   │    │Hierar-│          │   │
│  │  │BYOL │        │GoExpl │      │Episodic │    │chical │          │   │
│  │  │ ICM │        └───────┘      └─────────┘    └───────┘          │   │
│  │  └─────┘                                                          │   │
│  │                                                                    │   │
│  │  Multi-Agent              LLM-Based                               │   │
│  │  ┌───────────┐            ┌───────────────┐                       │   │
│  │  │Individual │            │ELLM Explorer  │                       │   │
│  │  │EMC/CERMIC │            │Eureka Rewards │                       │   │
│  │  │Competitive│            │Lang-Conditioned│                      │   │
│  │  └───────────┘            └───────────────┘                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                     ┌─────────────┴─────────────┐                       │
│                     │   COMPOUND CURIOSITY      │                       │
│                     │   (Adaptive Scheduling)   │                       │
│                     └─────────────┬─────────────┘                       │
│                                   │                                      │
│                     ┌─────────────┴─────────────┐                       │
│                     │    PPO AGENT (Dual V)     │                       │
│                     └───────────────────────────┘                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
void-runner/
├── src/
│   ├── __init__.py           # All exports
│   ├── networks.py           # Neural network architectures
│   ├── curiosity.py          # RND, ICM
│   ├── drnd.py               # DRND, NGU (ICML 2024)
│   ├── counting.py           # SimHash, VCSAP, Go-Explore archive
│   ├── byol_explore.py       # BYOL-Explore self-supervised
│   ├── empowerment.py        # Empowerment, causal curiosity
│   ├── episodic.py           # Memory-augmented curiosity
│   ├── compound.py           # Multi-objective curiosity
│   ├── skills.py             # DIAYN, DADS, hierarchical
│   ├── multiagent.py         # Multi-agent curiosity
│   ├── go_explore.py         # Neural Go-Explore
│   ├── llm_curiosity.py      # LLM-based exploration
│   ├── agent.py              # PPO agent with dual value heads
│   └── metrics.py            # Information-theoretic metrics
├── envs/
│   ├── __init__.py
│   ├── sparse_maze.py        # Basic sparse reward mazes
│   └── hard_exploration.py   # Challenging environments
├── utils/
│   └── visualize.py          # Visualization tools
├── train.py                  # Training script
├── benchmark.py              # Basic benchmark
├── benchmark_hard.py         # Hard exploration benchmark
└── README.md
```

## Hard Exploration Environments

| Environment | Description | Challenge |
|-------------|-------------|-----------|
| **DeceptiveRewardMaze** | Maze with misleading local rewards | Must overcome local optima |
| **KeyDoorEnv** | Find key before door opens | Temporal abstraction |
| **StochasticMaze** | Random transition noise | Robustness to stochasticity |
| **MultiGoalSparse** | Multiple scattered goals | Breadth of exploration |
| **MontezumaLite** | Rooms, keys, ladders | Full hierarchical challenge |

## Information-Theoretic Metrics

| Metric | What it measures |
|--------|-----------------|
| **State Entropy** H(S) | Diversity of states visited |
| **Action Entropy** H(A) | Diversity of behaviors |
| **Mutual Information** I(S;A) | Policy's state-dependence |
| **Coverage Rate** | Fraction of state space explored |
| **Exploration Efficiency** | New states per step |
| **Lempel-Ziv Complexity** | Trajectory compressibility |

## References

### Core Papers
- Burda et al., [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894), 2018
- Yang et al., [Distributional RND](https://arxiv.org/abs/2401.09750), ICML 2024
- Badia et al., [Never Give Up](https://arxiv.org/abs/2002.06038), 2020
- Pathak et al., [Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363), 2017
- Ecoffet et al., [Go-Explore](https://arxiv.org/abs/1901.10995), 2021
- Eysenbach et al., [DIAYN](https://arxiv.org/abs/1802.06070), 2018
- Ma et al., [Eureka](https://arxiv.org/abs/2310.12931), ICLR 2024

### Exploration Surveys
- [Awesome Exploration RL](https://github.com/opendilab/awesome-exploration-rl)
- Lilian Weng, [Exploration Strategies in Deep RL](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)

## License

MIT

---

*Built with curiosity about curiosity.*
