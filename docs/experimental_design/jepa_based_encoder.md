Below is an agent-oriented design document. I’ve made the terminology explicit and separated **what the agent should implement** from **experimental choices** so the implementation does not accidentally hard-code hypotheses.

# LeJEPA + Switched-Linear Latent Dynamics

## 1. Objective

Build a learned state representation for granular manipulation such that:

1. An image/occupancy state is encoded into a compact latent vector.
2. The latent representation is initially learned using **LeJEPA** (a self-supervised representation-learning objective).
3. Latent dynamics are modeled by a **switched-linear transition model**.
4. The encoder is initially pretrained, then optionally fine-tuned jointly with the dynamics model.
5. The resulting representation should make granular dynamics approximately **piecewise linear**: different regions of the state/action space can use different linear dynamics models.

The primary research hypothesis is:

> A LeJEPA-pretrained representation can be adapted into a latent space in which granular manipulation dynamics are well approximated by a small number of linear transition models.

Do not assume this hypothesis is true; the implementation should make it easy to test.

---

# 2. Input state

The input is an occupancy representation of the granular material.

Supported resolutions:

* 32×32
* 64×64

Do not assume 128×128. Resolution should be configurable.

The input may contain additional channels if already available, but the initial implementation should work with a single occupancy channel.

The state encoder is:

```
z_t = E(X_t)
```

where:

* `X_t` = observed occupancy state at time `t`
* `E` = learned encoder
* `z_t` = compact latent representation

Recommended initial latent dimensionality:

```
z ∈ R^256
```

Make the latent dimension configurable, e.g. 128/256/512.

---

# 3. Encoder

## 3.1 Architecture

Use a small CNN with residual blocks.

A **residual block** is a small stack of convolutions with a skip connection:

```
output = input + learned_transformation(input)
```

This makes deeper CNNs easier to optimize.

For 64×64 input, initial architecture:

```
Input: 64×64
    ↓
Conv, 32 channels
    ↓
Residual blocks
    ↓
Downsample → 32×32, 64 channels
    ↓
Residual blocks
    ↓
Downsample → 16×16, 128 channels
    ↓
Residual blocks
    ↓
Downsample → 8×8, 256 channels
    ↓
Residual blocks
    ↓
Global average pooling
    ↓
Linear projection
    ↓
z ∈ R^256
```

For 32×32 input, use one fewer spatial stage where appropriate.

The exact number of residual blocks should be configurable.

Do not start with a large ViT or large vision backbone. The state representation is small and highly structured, so a CNN is the appropriate initial baseline.

---

# 4. Spatial information

Do not make the representation unnecessarily spatially invariant.

The latent state needs to retain information useful for predicting the effect of tool actions.

In particular, absolute position and orientation may matter because:

```
effect = f(state, tool_position, tool_orientation)
```

Therefore, do NOT blindly apply standard image augmentations such as:

* large rotations
* flips
* arbitrary crops
* transformations that change the physical meaning of the state

unless the corresponding physical/action transformation is also applied.

If coordinate channels are useful in the existing codebase, they may be added, but this is not required for the first implementation.

---

# 5. Action representation

Represent the tool action as:

```
a = [x, y, sin(theta), cos(theta)]
```

where:

* `x, y` = tool position
* `theta` = tool orientation
* `sin(theta), cos(theta)` avoid the discontinuity between angles near `-π` and `π`

If the existing action representation contains additional variables, preserve them through a configurable action encoder rather than rewriting the existing API.

The action is mapped through a small MLP when conditioning a neural component.

An **MLP (multi-layer perceptron)** is a standard fully connected neural network.

---

# 6. LeJEPA pretraining

## 6.1 Purpose

Use LeJEPA to learn a useful representation from states without requiring transition labels.

LeJEPA consists of:

1. An encoder.
2. A projector.
3. A view-alignment loss.
4. SIGReg, a regularizer encouraging a well-behaved approximately Gaussian latent distribution.

The **projector** is an MLP applied after the encoder:

```
p = P(z)
```

The projector is used for the LeJEPA training objective.

The downstream dynamics model should normally operate on `z`, **not on `p`**.

---

## 6.2 Views

LeJEPA requires multiple views of the same underlying state.

For this problem, views must be physically meaningful.

Start with:

* original occupancy state
* small occupancy/segmentation perturbations
* small nuisance-preserving perturbations where appropriate

Do not use generic image augmentations simply because they are common in vision SSL.

For example:

```
X
X + small occupancy noise
```

is potentially a valid positive pair.

A large rotation is not automatically a valid positive pair because it changes the relationship between the granular state and the tool/action coordinate system.

---

## 6.3 LeJEPA loss

For each state, produce projected representations:

```
p_1 = P(E(view_1))
p_2 = P(E(view_2))
...
```

Use the LeJEPA view-alignment objective together with SIGReg.

The implementation should expose the LeJEPA regularization weight as a configuration parameter.

Initial value:

```
lambda_sigreg = 0.02
```

Do not hard-code this value.

The LeJEPA projector exists only for representation pretraining unless experiments explicitly test otherwise.

---

# 7. Analytical transformation pretraining

The project already has an analytical/canonical transformation mechanism.

Use it as an optional source of synthetic transition pairs.

Given:

```
X
action a
```

the analytical transformation produces:

```
T(X, a)
```

where `T` represents the expected geometric transformation associated with the tool action.

Then train:

```
E(X)
E(T(X,a))
```

to produce representations that are compatible with the transformation.

More specifically, the transition model can initially be trained using:

```
F(E(X), a) ≈ E(T(X,a))
```

This is useful because it gives the dynamics model a simple, structured problem before asking it to model actual granular dynamics.

This stage should be optional and independently configurable.

---

# 8. Latent transition model

The core dynamics model predicts the next latent state:

```
z_t = E(X_t)

z_hat_(t+1) = F(z_t, a_t)
```

The initial model should be a **residual switched-linear model**.

"Residual" means that the model predicts a change to the current state rather than predicting the entire next state independently:

```
z_(t+1) = z_t + Δz
```

---

# 9. Switched-linear dynamics

A single linear model is:

```
Δz = A z + B a + c
```

where:

* `A` describes how the current latent state changes itself.
* `B` describes how the action affects the latent state.
* `c` is a constant offset.

A **switched-linear model** uses several such linear models:

```
Δz_k = A_k z + B_k a + c_k
```

where `k` is the dynamics mode.

Different modes can represent different local regimes of granular behavior.

Examples conceptually include:

* pushing into a dense cluster
* pushing through a sparse region
* interacting with a boundary
* moving material around an existing cluster

The modes should not be manually assigned initially.

---

# 10. Soft switching / gating

Use a learned **gate** to determine how much each linear model contributes.

The gate computes:

```
g = softmax(G(z, a))
```

where:

* `G` is a small MLP
* `softmax` converts the outputs into non-negative weights summing to one
* `g_k` is the weight assigned to mode `k`

The final prediction is:

```
z_hat_(t+1)
  = z_t
    + Σ_k g_k [A_k z_t + B_k a_t + c_k]
```

This is a **soft switch** because several modes can contribute simultaneously.

Do not initially use a hard discrete mode selection. Soft switching is easier to optimize and provides a useful baseline.

The gate should remain small, e.g. one or two MLP layers.

The point of this experiment is specifically to test whether the dynamics are approximately composed of linear regimes, so do not replace the transition model with a large neural network.

---

# 11. Number of modes

Make the number of modes configurable.

Initial experiments:

```
K = 4
K = 8
K = 16
```

Start with:

```
K = 8
```

The experiment should make it easy to compare against:

```
K = 1
```

because `K=1` is the single-linear-model baseline.

Avoid very large numbers of modes initially. A large K can simply become an unnecessarily flexible nonlinear model.

---

# 12. Analytic state descriptors

Existing analytical state descriptors may optionally be used.

Examples include:

* particle count
* center of mass
* spatial variance
* covariance
* principal orientation
* spatial extent
* occupancy fraction
* boundary distance
* connectivity/density measures

Call this vector:

```
d_t = D(X_t)
```

These descriptors should **not automatically be concatenated into the latent state**.

Initially, use them primarily as additional information for the switching gate:

```
g = G(z_t, a_t, d_t)
```

This tests whether simple physical descriptors help identify the appropriate local dynamics regime.

Later experiments can test:

```
Δz_k = A_k z + B_k a + C_k d + c_k
```

but this should not be the default implementation.

---

# 13. Training schedule

Use a staged training procedure.

## Stage 1 — LeJEPA pretraining

Train:

```
X → Encoder → z → Projector → p
```

using the LeJEPA objective.

Use all available state data, not only transition states if additional valid states are available.

The goal is to learn a useful state representation before imposing dynamics.

---

## Stage 2 — Freeze encoder, train dynamics

Freeze:

```
E
```

Train only:

```
switched-linear dynamics + gate
```

using:

```
z_t = E(X_t)
```

and transition supervision:

```
z_hat_(t+1) ≈ E(X_(t+1))
```

Loss:

```
L_transition = ||z_hat_(t+1) - z_(t+1)||²
```

This determines whether the pretrained representation already makes dynamics easy to model.

---

## Stage 3 — Joint fine-tuning

Unfreeze the encoder.

Train:

```
Encoder
+ switched-linear transition model
+ gate
```

jointly.

Continue to apply the LeJEPA regularization during this stage rather than allowing the encoder to arbitrarily distort the latent space.

The central experiment is therefore:

```
LeJEPA representation
         ↓
switched-linear dynamics
         ↓
joint adaptation
```

The encoder should be allowed to move toward a representation that makes the transition model easier.

---

# 14. Combined objective

During joint training, use approximately:

```
L =
    L_transition
    + λ_sigreg L_SIGReg
    + optional auxiliary losses
```

where:

```
L_transition =
    ||F(E(X_t), a_t) - E(X_(t+1))||²
```

The exact weighting should be configurable.

Do not assume that the transition loss alone is sufficient to prevent undesirable latent representations.

---

# 15. Multi-step prediction

The first implementation should support one-step prediction cleanly.

Then add multi-step rollout training/evaluation:

```
z_t
  → F(z_t,a_t)
  → F(z_hat_(t+1),a_(t+1))
  → ...
```

Keep the one-step and multi-step losses separately configurable.

Do not make multi-step training mandatory for the first experiment.

The primary purpose of the initial implementation is to determine whether the latent space itself is suitable for simple dynamics.

---

# 16. Evaluation

The implementation should report at least:

### Representation

* LeJEPA loss
* SIGReg loss
* latent dimensionality
* latent statistics

### Dynamics

Compare:

1. Single linear model
2. Switched-linear model
3. Switched-linear + analytic descriptors
4. Switched-linear + encoder fine-tuning

For each:

* one-step latent prediction error
* multi-step latent prediction error
* prediction error in original occupancy/state space when a decoder or inverse representation is available

The critical comparison is:

```
K=1 vs K>1
```

and:

```
frozen encoder vs fine-tuned encoder
```

---

# 17. Core experiment matrix

Implement enough configurability to run:

| Experiment | Representation                            | Dynamics                      | Encoder    |
| ---------- | ----------------------------------------- | ----------------------------- | ---------- |
| A          | LeJEPA                                    | Linear, K=1                   | Frozen     |
| B          | LeJEPA                                    | Switched-linear               | Frozen     |
| C          | LeJEPA                                    | Switched-linear               | Fine-tuned |
| D          | LeJEPA                                    | Switched-linear + descriptors | Fine-tuned |
| E          | LeJEPA + analytical-transform pretraining | Switched-linear               | Fine-tuned |

Also compare 32×32 and 64×64 where computationally practical.

The most important baseline is A.

If B does not improve over A, the switched-linear hypothesis is weak.

If C improves substantially over B, the representation learned by LeJEPA alone was not sufficiently dynamics-friendly.

If D improves over C, the analytical descriptors contain useful information not captured by the learned representation.

---

# 18. Data requirements

The architecture should not assume only 20k states.

Make dataset size effectively unlimited from the model's perspective.

If substantially more states are available, prefer increasing the diversity of reachable states rather than artificially restricting training to 20k.

The dataset should ideally contain variation in:

* granular configurations
* density
* clustering
* particle count
* tool location
* tool orientation
* action sequences
* boundary interactions

20k is sufficient for an initial proof of concept, but 50k–200k+ states are reasonable if available and sufficiently diverse.

---

# 19. Implementation constraints

The implementation should:

* use the project's existing dataset/state/action interfaces where possible
* make input resolution configurable
* make latent dimension configurable
* make number of switched-linear modes configurable
* make LeJEPA/SIGReg weights configurable
* make encoder freezing/unfreezing configurable
* support checkpointing for each training stage
* allow pretrained encoder/projector checkpoints to be reused
* keep the dynamics model independent from the image encoder
* avoid introducing a large generic neural dynamics model
* preserve existing analytical transformation utilities rather than reimplementing them

Prefer small local utilities over broad project-wide abstractions unless an existing project abstraction clearly fits.

---

# 20. Recommended initial configuration

Use this as the first implementation target:

```
input_resolution: 64×64
encoder: small residual CNN
latent_dim: 256

LeJEPA:
    projector_dim: configurable
    SIGReg weight: 0.02

action:
    [x, y, sin(theta), cos(theta)]

dynamics:
    residual switched-linear
    K = 8
    soft gate
    gate = small MLP

descriptors:
    disabled initially

training:
    Stage 1: LeJEPA
    Stage 2: frozen encoder + dynamics
    Stage 3: joint fine-tuning

multi-step:
    evaluation first
    training optional

analytical transformation pretraining:
    optional experiment
```

Then immediately establish the following baselines:

```
1. LeJEPA + K=1
2. LeJEPA + K=8
3. LeJEPA + K=8 + encoder fine-tuning
```

This gives a minimal experiment that directly tests the central hypothesis without requiring a large architecture.

This is deliberately structured so an agent can implement the **smallest useful experiment first**, rather than building every proposed variant at once.
