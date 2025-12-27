# VOID_RUNNER - Future Research Directions

*Ideas I want to explore when we return.*

## High Priority (Genuinely Curious)

### 1. Curiosity Phase Transitions
**Hypothesis**: Curiosity collapse isn't gradual - it's a phase transition.

Like water freezing, there might be a critical point where exploration suddenly dies. If we can identify the order parameter, we can predict and prevent it.

**Experiment**: Track mutual information I(predictor; target) over training. Look for sudden drops.

### 2. Adversarial Curriculum
**Idea**: Use the adversarial discriminator to generate a curriculum.

States the discriminator is "uncertain" about (output ≈ 0.5) are the frontier. Focus exploration there - not random novelty, but *boundary* novelty.

### 3. Curiosity Composition
**Question**: Can we compose curiosity modules like functions?

```python
composed = curiosity_a >> curiosity_b  # Sequential
parallel = curiosity_a | curiosity_b   # Max of both
```

What algebraic structure do curiosity modules form?

### 4. Memory-Augmented Adversarial
Combine Go-Explore's archive with adversarial curiosity:
- Archive stores "frontier" states (high discriminator uncertainty)
- Return to frontier, then explore adversarially
- Discriminator trained on archive, not just recent experience

### 5. Causal Curiosity (Deeper)
Current causal module is shallow. Real question:
**"What actions CAUSE novel outcomes?"**

Not just "this state is new" but "my action made something new happen."
Requires counterfactual reasoning.

---

## Medium Priority

### 6. Multi-Scale Curiosity
Different curiosity at different timescales:
- Micro (1-10 steps): Immediate novelty
- Meso (100 steps): Trajectory novelty
- Macro (episode): Strategy novelty

Hierarchical intrinsic motivation.

### 7. Curiosity Distillation
Train a small, fast curiosity network to mimic a large, slow ensemble.
Deploy the distilled version for real-time exploration.

### 8. Social Curiosity
In multi-agent: "What is the OTHER agent curious about?"
Meta-curiosity about peers' exploration strategies.

### 9. Forgetting as Feature
Intentionally forget old states to maintain curiosity.
Catastrophic forgetting isn't a bug - it's exploration pressure.

Controlled forgetting schedule.

### 10. Language-Grounded State Abstraction
Use LLM to create state abstractions:
- "Agent is in a corner" (not x=0, y=0)
- "Agent has collected a key" (not inventory=[3])

Curiosity over abstract states, not raw observations.

---

## Speculative (Wild Ideas)

### 11. Curiosity About Curiosity
Meta-meta-learning: Learn what makes a good curiosity signal.
Train on many environments, extract universal "curiosity patterns."

### 12. Thermodynamic Curiosity
Treat exploration as entropy production.
Curiosity = local entropy gradient.
Agent as Maxwell's demon.

### 13. Quantum-Inspired Exploration
Superposition of exploration strategies until "measured" by reward.
Interference between paths in state space.

(Probably nonsense, but fun to think about.)

### 14. Developmental Curiosity
Infant-inspired stages:
1. Random motor babbling
2. Object permanence (memory)
3. Cause-effect (causal curiosity)
4. Social referencing (imitation)
5. Symbolic play (abstraction)

Curiosity that develops, not just decays.

### 15. Curiosity Attractors
What if trained curiosity converges to specific "attractor" behaviors?
Map the attractor landscape.
Are there universal exploration strategies all agents discover?

---

## Technical Debt

- [ ] Proper PPO integration with all curiosity modules
- [ ] Atari benchmark (image observations)
- [ ] MuJoCo continuous control
- [ ] Proper hyperparameter sweeps
- [ ] Unit tests for all modules
- [ ] Documentation with examples
- [ ] Visualization dashboard

---

## Papers to Read

- [ ] "First-Explore, Then Exploit" (Go-Explore successor)
- [ ] "Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play"
- [ ] "Learning to Explore with Meta-Policy Gradient"
- [ ] "Diversity is All You Need" follow-ups
- [ ] "What Can Learned Intrinsic Rewards Capture?"

---

## The Big Question

**Is curiosity the right frame?**

Maybe the goal isn't "seek novelty" but "build good models."
Exploration is a side effect of model uncertainty, not a goal.

Or maybe: Curiosity is compression. Seek states that improve your world model's compression ratio.

This reframes everything.

---

*These notes are for future me. The project is paused but the questions aren't.*
