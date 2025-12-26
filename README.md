# VOID_RUNNER

```
██╗   ██╗ ██████╗ ██╗██████╗     ██████╗ ██╗   ██╗███╗   ██╗███╗   ██╗███████╗██████╗
██║   ██║██╔═══██╗██║██╔══██╗    ██╔══██╗██║   ██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
██║   ██║██║   ██║██║██║  ██║    ██████╔╝██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝
╚██╗ ██╔╝██║   ██║██║██║  ██║    ██╔══██╗██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗
 ╚████╔╝ ╚██████╔╝██║██████╔╝    ██║  ██║╚██████╔╝██║ ╚████║██║ ╚████║███████╗██║  ██║
  ╚═══╝   ╚═════╝ ╚═╝╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
                     Curiosity-Driven Exploration Research
```

> *"The reward of curiosity is not just what you find, but what you become while searching."*

A **comprehensive research framework** for intrinsic motivation in reinforcement learning. Implements state-of-the-art methods from ICML 2024, NeurIPS, and ICLR, with original extensions.

## What's Implemented

| Method | Paper | Year | Status |
|--------|-------|------|--------|
| RND | [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) | 2018 | ✅ |
| ICM | [Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363) | 2017 | ✅ |
| **DRND** | [Distributional RND](https://arxiv.org/abs/2401.09750) | ICML 2024 | ✅ |
| **NGU** | [Never Give Up](https://arxiv.org/abs/2002.06038) | 2020 | ✅ |
| SimHash | [#Exploration](https://arxiv.org/abs/1611.04717) | 2017 | ✅ |
| VCSAP | [State-Action Counting](https://www.sciencedirect.com/science/article/abs/pii/S089360802400981X) | 2024 | ✅ |
| Pseudo-Count | [Unifying Count-Based](https://arxiv.org/abs/1606.01868) | 2016 | ✅ |
| Go-Explore | [First Return Then Explore](https://arxiv.org/abs/1901.10995) | 2021 | ✅ |
| Empowerment | [Variational Info Max](https://arxiv.org/abs/1509.08731) | 2015 | ✅ |
| Causal Curiosity | [NeurIPS 2024](https://openreview.net/forum?id=LZI8EFLoFD) | 2024 | ✅ |

## Key Research Findings

### The Bonus Inconsistency Problem (ICML 2024)

Standard RND suffers from **bonus inconsistency**:
1. **Initial inconsistency**: Uneven bonus distribution at training start
2. **Final inconsistency**: Poor discrimination between visited/unvisited states

**DRND Solution**: Use N random networks instead of 1, enabling:
- Averaging reduces initial variance
- Implicit pseudo-count estimation: `y(x) ≈ 1/n(x)`

```python
from src import DRND
curiosity = DRND(obs_dim=4, n_networks=5)
bonus = curiosity.compute_intrinsic_reward(observation)
```

### Episodic vs Lifelong Curiosity (NGU)

Different timescales of novelty matter:
- **Episodic** (fast): k-NN in embedding space, resets each episode
- **Lifelong** (slow): RND/DRND, persists across training

NGU combines them: `r = r_episodic × clamp(r_lifelong, 1, L)`

```python
from src import NGUCombinedCuriosity
curiosity = NGUCombinedCuriosity(obs_dim=4)
bonus, info = curiosity.compute_intrinsic_reward(obs)
curiosity.reset_episode()  # At episode boundaries
```

### Count-Based Methods Still Work

Despite neural approaches, hash-based counting remains competitive:

```python
from src import SimHashCounter, StateActionCounter
counter = StateActionCounter(obs_dim=4, action_dim=4)
bonus = counter.compute_bonus(obs, action)
```

**VCSAP insight**: Count both states AND state-action pairs to prevent over-exploration.

## Benchmark Results

```
Method          Goals     Coverage    Entropy
------------------------------------------------------------
none            0.0       12.4%       2.42
rnd             1.5       12.9%       1.94
drnd            1.0       17.0%       2.09
simhash         0.0       17.4%       2.96
```

Run your own: `python benchmark.py --methods rnd drnd simhash ngu`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VOID_RUNNER FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              INTRINSIC MOTIVATION MODULES               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Prediction-Based      Count-Based      Information     │   │
│  │  ┌─────┐ ┌─────┐      ┌───────┐       ┌───────────┐   │   │
│  │  │ RND │ │DRND │      │SimHash│       │Empowerment│   │   │
│  │  └─────┘ └─────┘      └───────┘       └───────────┘   │   │
│  │  ┌─────┐ ┌─────┐      ┌───────┐       ┌───────────┐   │   │
│  │  │ ICM │ │ NGU │      │ VCSAP │       │  Causal   │   │   │
│  │  └─────┘ └─────┘      └───────┘       └───────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                       │
│                    │ COMPOUND CURIOSITY │                       │
│                    │   (Adaptive Mix)   │                       │
│                    └─────────┬─────────┘                       │
│                              │                                  │
│                    ┌─────────┴─────────┐                       │
│                    │    PPO AGENT      │                       │
│                    └───────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Basic training (RND)
python train.py --env SparseMaze-v0 --steps 100000

# State-of-the-art (DRND + NGU)
python train_advanced.py --curiosity novelty empowerment

# Benchmark all methods
python benchmark.py --methods none rnd drnd simhash vcsap --steps 50000
```

## Project Structure

```
void-runner/
├── src/
│   ├── networks.py      # Neural network architectures
│   ├── curiosity.py     # RND, ICM
│   ├── drnd.py          # DRND, NGU (ICML 2024)
│   ├── counting.py      # SimHash, VCSAP, Go-Explore
│   ├── empowerment.py   # Empowerment, causal curiosity
│   ├── episodic.py      # Memory-augmented curiosity
│   ├── compound.py      # Multi-objective system
│   ├── metrics.py       # Information-theoretic metrics
│   └── agent.py         # PPO agent
├── envs/
│   └── sparse_maze.py   # Sparse reward environments
├── utils/
│   └── visualize.py     # Visualization tools
├── train.py             # Basic training
├── train_advanced.py    # Advanced compound curiosity
├── benchmark.py         # Method comparison
└── README.md
```

## Information-Theoretic Metrics

Beyond reward, we measure exploration quality:

| Metric | What it measures |
|--------|-----------------|
| **State Entropy** H(S) | Diversity of states visited |
| **Action Entropy** H(A) | Diversity of behaviors |
| **Mutual Information** I(S;A) | Policy's state-dependence |
| **Coverage Rate** | Fraction of state space explored |
| **Exploration Efficiency** | New states per step |
| **Lempel-Ziv Complexity** | Trajectory compressibility |

## Key Insights from Research

### 1. No Single Method Wins
Different curiosity types excel in different scenarios:
- **Novel environment** → RND/DRND (broad exploration)
- **Sparse rewards** → NGU (revisit + novelty)
- **Control-focused** → Empowerment
- **Simple env** → SimHash (efficient, interpretable)

### 2. The Curiosity Decay Problem
As agents explore, curiosity naturally decays. Solutions:
- **Episodic memory** (NGU): Separate within/across episode novelty
- **Count decay**: Forget old visits in non-stationary environments
- **Re-exploration**: Revisit old areas to check for changes

### 3. Pseudo-Counts Are Underrated
The connection between prediction error and counting:
```
RND error ≈ 1/√(pseudo_count)
```
DRND makes this explicit with its variance-based pseudo-count term.

## Recent Advances (2024-2025)

From my research survey:

- **[LLM-based Intrinsic Motivation](https://arxiv.org/html/2410.23022)** (ONI): Use LLM feedback as reward signal
- **[Causal Curiosity](https://openreview.net/forum?id=LZI8EFLoFD)**: Curiosity about causal structure, not just novelty
- **[Vision Transformer Curiosity](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009245)**: Hierarchical ViT for better state representations
- **[Reward Shaping Theory](https://link.springer.com/article/10.1007/s00521-025-11340-0)**: PBIM, GRM, ADOPS for principled intrinsic rewards

## Future Directions

- [ ] LLM-based reward annotation (ONI-style)
- [ ] Hierarchical curiosity (multi-scale)
- [ ] Transfer of curiosity across environments
- [ ] Curiosity in multi-agent settings
- [ ] Meta-learning curiosity hyperparameters

## References

### Core Papers
- Burda et al., [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894), 2018
- Yang et al., [Distributional RND](https://arxiv.org/abs/2401.09750), ICML 2024
- Badia et al., [Never Give Up](https://arxiv.org/abs/2002.06038), 2020
- Pathak et al., [Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363), 2017

### Exploration Surveys
- [Awesome Exploration RL](https://github.com/opendilab/awesome-exploration-rl)
- Lilian Weng, [Exploration Strategies in Deep RL](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)

## License

MIT
