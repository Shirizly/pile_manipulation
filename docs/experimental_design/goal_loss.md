# Goal-Conditioned Latent Representation Loss

## Objective

Add an auxiliary loss that encourages the state encoder to preserve the **geometric information needed to evaluate how well a state satisfies a desired goal**.

The key hypothesis is that, with a suitable representation, the goal-dependent value/cost of a state can be predicted by a simple linear function of the difference between their latent embeddings.

## Goal generation

For each training state `x`, sample one or more goal specifications `g`.

Generate one or more synthetic occupancy states satisfying each goal:

```
s_g1, s_g2, ..., s_gN
```

These states should contain a simple, low-height distribution of objects matching the desired goal shape. Generation should be inexpensive and should not require simulating a high pile.

The multiple realizations represent the same goal while differing in the exact arrangement of individual objects.

## Goal embedding

Encode each generated goal state using the same state encoder:

```
z_gi = E(s_gi)
```

Compute their mean embedding:

```
z_g = mean_i(z_gi)
```

This provides an embedding of the goal rather than of one particular particle arrangement.

The current state is encoded as:

```
z_x = E(x)
```

Define the goal-relative latent representation:

```
Δz = z_x - z_g
```

## Value prediction

For each `(x, g)` pair, compute a known goal-dependent target:

```
v(x | g)
```

The initial target can be a normalized geometric cost such as the fraction of material outside the goal region.

Fit a learned **linear value head**:

```
v_hat(x | g) = wᵀ Δz + b
```

where `w` and `b` are learned parameters.

The goal loss is:

```
L_goal = MSE(v_hat(x | g), v(x | g))
```

The linear restriction is intentional: the loss should encourage the encoder itself to organize the latent space so that goal satisfaction is approximately linearly readable from the state-to-goal latent difference.

## Integration

Add the goal loss as an auxiliary term to the encoder's existing training objective:

```
L_total = L_existing + λ_goal L_goal
```

Make `λ_goal` configurable.

The goal-loss implementation should be modular so that different goal-generation procedures and value/cost functions can be substituted without changing the encoder or training pipeline.
