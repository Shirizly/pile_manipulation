Yes — the previous version compressed away the implementation-level definitions. Here's a fuller design document while still keeping it to the architecture/specification rather than explaining the rationale.

# Action-Conditioned Latent Transformation Model

## 1. Objective

Learn a state representation and action-conditioned transition model for occupancy-grid states.

Given:

* current state:

  $$
  X_t\in\mathbb{R}^{H\times W}
  $$
* manipulation action:

  $$
  a_t=(x_p,y_p,\theta_p)
  $$
* next state:

  $$
  X_{t+1}
  $$

the model predicts the next state **in latent space**:

$$
E_\phi(X_t),a_t
\rightarrow
\hat z_{t+1}
$$

and compares it with:

$$
z_{t+1}=E_\phi(X_{t+1}).
$$

The initial training target is the known analytical spatial transformation:

$$
T_a(X_t).
$$

The representation is subsequently fine-tuned using actual granular-material transitions.

---

# 2. Model components

## 2.1 State encoder

The state encoder maps an occupancy image to a spatial latent representation:

$$
z_s=E_\phi(X)
$$

Recommended initial input:

$$
X\in\mathbb{R}^{128\times128\times1}.
$$

Use a CNN encoder with 3–4 spatial downsampling stages.

Example:

```text
128×128×1
    │
    ├── Conv 3×3, 32
    ├── Residual/Conv block
    │
    ▼
64×64×64
    │
    ├── Residual/Conv block
    │
    ▼
32×32×128
    │
    ├── Residual/Conv block
    │
    ▼
16×16×256
    │
    └── latent spatial feature map
```

The latent should initially remain spatial:

$$
z_s\in\mathbb{R}^{16\times16\times C}
$$

rather than immediately collapsing to a single vector.

A global representation can additionally be obtained by global average pooling:

$$
z_g=\operatorname{GAP}(z_s)
$$

with:

$$
z_g\in\mathbb{R}^{C}.
$$

---

# 3. Coordinate channels

Add explicit spatial coordinates to the encoder input.

For each pixel \((i,j)\), provide normalized coordinates:

$$
c_x(i,j),c_y(i,j)\in[-1,1].
$$

The basic encoder input is therefore:

$$
X_{\text{enc}}=
[X,c_x,c_y].
$$

For action-conditioned geometric pretraining, additionally provide coordinates relative to the plate:

$$
\Delta x=i-x_p
$$

$$
\Delta y=j-y_p.
$$

Rotate these into the plate coordinate frame:

$$
\Delta x'
=
\cos\theta\,\Delta x+\sin\theta\,\Delta y
$$

$$
\Delta y'
=
-\sin\theta\,\Delta x+\cos\theta\,\Delta y.
$$

These can be supplied as additional channels.

Possible encoder input:

$$
X_{\text{enc}}
=
[X,c_x,c_y,\Delta x',\Delta y'].
$$

---

# 4. Analytic state descriptors

Calculate a vector of known state descriptors:

$$
d(X)\in\mathbb{R}^{D_d}.
$$

Possible descriptors include:

* total occupancy / particle count;
* center of mass \(x,y\);
* spatial variance;
* covariance matrix;
* principal-axis orientation;
* \(x/y\) extent;
* distance to arena boundaries;
* occupied-area fraction;
* density statistics;
* connected-component statistics.

Normalize each descriptor before concatenation.

The learned and analytic representations are combined as:

$$
z=
[z_s,z_g,d(X)].
$$

If the predictor operates on a spatial latent, \(z_s\) remains the main state representation and \(z_g,d(X)\) are global conditioning features.

---

# 5. Action representation

Represent the plate pose as:

$$
a=
[x_p,y_p,\sin\theta,\cos\theta].
$$

Normalize \(x_p,y_p\) to the same coordinate convention as the image.

Pass this through a small MLP:

```text
[xp, yp, sinθ, cosθ]
        │
      Linear
        │
      SiLU/ReLU
        │
      Linear
        │
      action embedding ea
```

For example:

$$
e_a\in\mathbb{R}^{64\text{–}128}.
$$

---

# 6. Action-conditioned latent predictor

The predictor receives:

$$
z_s,\quad z_g,\quad d(X),\quad e_a.
$$

It predicts a latent change:

$$
\Delta z=P_\psi(z_s,z_g,d,e_a)
$$

and uses a residual transition:

$$
\boxed{
\hat z_{t+1}=z_t+\Delta z
}
$$

where:

$$
z_t=E_\phi(X_t).
$$

The predictor is a shallow CNN operating on the spatial latent.

Example:

```text
16×16×C latent
       │
       ├── Conv/Residual block
       │        ↑
       │      FiLM(ea)
       │
       ├── Conv/Residual block
       │        ↑
       │      FiLM(ea)
       │
       └── Conv 3×3
              │
              ▼
           Δz_s
```

---

# 7. Action conditioning

Use FiLM conditioning at multiple predictor layers.

Given action embedding \(e_a\), each conditioned block generates:

$$
\gamma(e_a),\beta(e_a)
$$

and applies:

$$
h'=
\gamma(e_a)\odot h+\beta(e_a).
$$

The action can also be concatenated to global features:

$$
h_{\text{global}}
=
[z_g,d(X),e_a].
$$

Optionally broadcast this vector spatially and concatenate it to the latent feature map.

---

# 8. Optional decoder

A CNN decoder maps the spatial latent back to an occupancy grid:

$$
\hat X=D_\omega(z_s).
$$

Example:

```text
16×16×C
   │
 Upsample
   │
32×32
   │
 Upsample
   │
64×64
   │
 Upsample
   │
128×128×1
```

The decoder is primarily used during representation/geometry pretraining and can subsequently be retained as an auxiliary reconstruction head.

---

# 9. Training stages

## Stage 1 — State reconstruction

Train:

$$
X\rightarrow E(X)\rightarrow D(E(X))
$$

with:

$$
\mathcal L_{\text{recon}}
=
\ell(D(E(X)),X).
$$

This establishes an information-preserving initial embedding.

---

## Stage 2 — Analytical transformation prediction

For each state \(X\) and plate pose \(a\), calculate:

$$
X'=T_a(X).
$$

Encode both:

$$
z=E(X)
$$

$$
z'=E(X').
$$

Train:

$$
P(z,a)\rightarrow\hat z'
$$

with:

$$
\mathcal L_{\text{latent}}
=
\|\hat z'-z'\|^2.
$$

Optionally decode:

$$
\hat X'=D(\hat z')
$$

and add:

$$
\mathcal L_{\text{image}}
=
\ell(\hat X',X').
$$

Initial combined loss:

$$
\mathcal L
=
\lambda_z\mathcal L_{\text{latent}}
+
\lambda_x\mathcal L_{\text{image}}.
$$

---

## Stage 3 — Real dynamics prediction

Replace the analytically transformed target with the experimentally observed next state:

$$
X_t,a_t,X_{t+1}.
$$

Compute:

$$
z_t=E(X_t)
$$

$$
z_{t+1}=E(X_{t+1})
$$

and predict:

$$
\hat z_{t+1}
=
z_t+P(z_t,a_t).
$$

Train:

$$
\mathcal L_{\text{dyn}}
=
\|\hat z_{t+1}-z_{t+1}\|^2.
$$

The encoder \(E\) and predictor \(P\) are jointly trainable during this stage.

---

# 10. Analytic descriptor prediction

Optionally have the predictor produce the next-state descriptors:

$$
\hat d_{t+1}
=
P_d(z_t,d_t,a_t).
$$

Compare against:

$$
d_{t+1}=d(X_{t+1}).
$$

Loss:

$$
\mathcal L_d
=
\|\hat d_{t+1}-d_{t+1}\|^2.
$$

Overall dynamics loss:

$$
\mathcal L_{\text{dyn}}
=
\lambda_z\mathcal L_z+
\lambda_d\mathcal L_d+
\lambda_x\mathcal L_x.
$$

The descriptor branch can be auxiliary rather than part of the primary prediction target.

---

# 11. Recommended initial dimensions

For the first implementation:

| Component             | Initial choice                                        |
| --------------------- | ----------------------------------------------------- |
| Input                 | \(128\times128\times1\)                               |
| Coordinate channels   | 2–4                                                   |
| Encoder levels        | 3–4                                                   |
| Encoder channels      | 32 → 64 → 128 → 256                                   |
| Spatial latent        | \(16\times16\times128\) or \(16\times16\times256\)    |
| Global latent         | 128–256                                               |
| Action representation | \(x,y,\sin\theta,\cos\theta\)                         |
| Action embedding      | 64–128                                                |
| Predictor depth       | 2–4 residual/conv blocks                              |
| Conditioning          | FiLM + optional broadcast concatenation               |
| Transition            | residual \(z'=z+\Delta z\)                            |
| Decoder               | 3–4 upsampling stages                                 |
| Initial losses        | latent + reconstruction                               |
| Later loss            | latent dynamics + optional reconstruction/descriptors |

---

# 12. Primary experiment

Compare the following progressively:

1. **Analytical transformation**

   $$
   T_a(X)
   $$

2. **Learned transformation**

   $$
   E(X),a\rightarrow E(T_a(X))
   $$

3. **Learned dynamics**

   $$
   E(X_t),a_t\rightarrow E(X_{t+1})
   $$

4. **Jointly adapted representation**
   Continue training \(E\) during real-dynamics prediction.

Evaluate both:

* transformation prediction error;
* downstream granular-dynamics prediction performance.

# Variations

## Variation 1 — Analytical-transform residual

Replace full learned transformation with:

$$
\hat X'
=
T_a(X)+R_\psi(X,a).
$$

The neural network learns only the residual relative to the analytical spatial transformation.

For latent prediction, use:

$$
\hat z'
=
E(T_a(X))+R_z(E(X),a).
$$

---

## Variation 2 — Spatial Transformer

Replace explicit coordinate-based transformation learning with a differentiable spatial transformer:

$$
X
\xrightarrow{\text{pose-conditioned warp}}
X_{\text{warp}}
$$

followed by a shallow CNN residual model:

$$
X_{\text{warp}}
\rightarrow
X_{\text{warp}}+R_\psi(X,a).
$$

The latent predictor receives the transformed latent plus a learned residual.

---

## Variation 3 — SE(2)-equivariant encoder

Replace the ordinary CNN encoder with a rotation/translation-equivariant CNN.

The latent representation consists of equivariant spatial feature maps:

$$
E(gX)\approx gE(X)
$$

for \(g\in SE(2)\) or a discretized rotation subgroup.

Action conditioning is performed in the corresponding group representation rather than exclusively through ordinary coordinate channels and FiLM.
