# Rank collapse when training a latent encoder through a linear transition model

**Status:** open problem, seeking outside input.
**Date:** 2026-09-17.
**Context:** granular manipulation (pushing ~20 rigid cubes with a flat plate toward a target region). We want a latent state representation in which the dynamics are approximately linear, and we want it to support *action ranking* — choosing, among ~1000 candidate pushes from one state, the one that best improves a goal-dependent value.

Everything below is measured, not simulated or estimated. Code paths and run logs are named at the end.

---

## 1. Setup and notation

A state is $N \approx 20$ cube poses, rasterised to a single-channel occupancy image
$$X \in [0,1]^{G \times G}, \qquad G \in \{32, 64\},$$
over a fixed $128\,\text{mm} \times 128\,\text{mm}$ workspace. A cube is $5$ mm, so it is $2.5$ px across at $G=64$ and $1.25$ px at $G=32$.

An action is a straight push, parameterised by its start and stop points in the plane. We encode it as
$$a = [x_s,\; y_s,\; \sin\theta,\; \cos\theta,\; \Delta x,\; \Delta y] \in \mathbb{R}^6 ,$$
with push length $\ell = \lVert(\Delta x, \Delta y)\rVert \in [0, 80]$ mm.

An **encoder** $E_\theta : [0,1]^{G\times G} \to \mathbb{R}^{D}$, $D = 256$, is a small residual CNN (GroupNorm, SiLU, global average pool, linear projection).

The **transition model** is residual and *switched* by push length — six equal-width bins $b(\ell) \in \{0,\dots,5\}$ over $[0, 80]$ mm, each with its own affine map:
$$\hat z_{t+1} \;=\; z_t \;+\; W_{b(\ell)}^\top \begin{bmatrix} z_t \\ a_t \\ 1\end{bmatrix}, \qquad W_b \in \mathbb{R}^{(D+7)\times D}.$$

Write $z = E_\theta(X_t)$, $z' = E_\theta(X_{t+1})$, and the **true latent displacement** $d = z' - z$, **predicted** $\hat d = W_b^\top [z; a; 1]$.

### 1.1 The evaluation metric we actually care about

`slateN` — the fraction of the oracle's advantage over a random pick that a model captures, when choosing from all $n$ candidates of one pool:
$$\text{slateN} \;=\; \frac{\bar v - v_{\text{chosen}}}{\bar v - v_{\text{best}}},$$
where $v$ is the realised post-push value of a candidate, $\bar v$ its pool mean, $v_\text{best}$ the pool optimum, and $v_\text{chosen}$ the value of the candidate the model ranked first. $0$ = no better than random; $1$ = oracle. This is the number that decides; everything else below is diagnostic.

---

## 2. What was trained before the problem appeared

**Stage 1 (self-supervised, no actions).** The encoder is trained with a LeJEPA-style objective on *single states*: two physically meaningful views $v_1, v_2$ of the same state (particle dropout, footprint-radius jitter, small occupancy noise — deliberately no rotations, flips or crops, which would break the pile/tool frame relation), a projector $P$, and
$$\mathcal{L}_{\text{S1}} \;=\; \big\lVert P(E(v_1)) - P(E(v_2))\big\rVert^2 \;+\; \lambda_{\text{sig}}\,\mathrm{SIGReg}(P(E(\cdot))) \;+\; \lambda_{\text{sig}}\,\mathrm{SIGReg}(E(\cdot)).$$

**SIGReg** (Sketch Isotropic Gaussian Regulariser) is an Epps–Pulley test statistic evaluated on $M=1024$ random 1-D projections $A_j$ of the batch, comparing the empirical characteristic function of $\langle z, A_j\rangle$ against that of $\mathcal{N}(0,1)$:
$$\mathrm{SIGReg}(z) \;=\; \frac{1}{M}\sum_{j=1}^{M} B \int_0^3 \Big[\big(\mathbb{E}_B[\cos(t\langle z,A_j\rangle)] - e^{-t^2/2}\big)^2 + \big(\mathbb{E}_B[\sin(t\langle z,A_j\rangle)]\big)^2\Big] e^{-t^2/2}\,\mathrm{d}t,$$
discretised over 17 knots, with $B$ the batch size. **Measured floor:** on true $z\sim\mathcal{N}(0,I_{256})$ at $B=192$ this statistic equals $\mathbf{1.08}$. Any value far above that means the latent is *not* isotropic Gaussian.

Stage 1 works in the sense of not collapsing the variance, but produces a latent with effective rank ${\approx}7$ of $256$ (see §5 for the rank definition). Stage 2 (freeze $E$, fit $W_b$ in closed form by ridge) then gives a usable dynamics fit. **Stage 3 — unfreezing the encoder and training it through the transition error — had never been run.** That is what this report is about.

### 2.1 Prior art in this codebase

An earlier experiment (different architecture, FiLM-conditioned) trained an encoder through a plain latent transition loss $\lVert \hat z_{t+1} - z_{t+1}\rVert^2$ with no regulariser. It **collapsed**: per-dimension standard deviation fell to ${\sim}2\times10^{-6}$, with model loss and baseline loss converging to ${\sim}10^{-9}$ together. A per-channel variance hinge fixed it. This is the known failure mode we were trying to avoid, and it motivated everything in §3.

---

## 3. Three objectives tried, and the reasoning behind each

All three share: target detached (stop-gradient), $W_b$ learned jointly and initialised at $0$ (so $\hat d = 0$ at step 0), AdamW with OneCycle, $\lambda_{\text{sig}} = 0.02$, $G=32$, $D=256$, batch 128, $8 \times 250$ steps, 49,152 training transitions, 6,144 held out at the file level.

### V1 — scale-normalised ratio (attached denominator)

To avoid §2.1's shrink-collapse, normalise by the magnitude of the thing being predicted:
$$\mathcal{L}_{\text{V1}} \;=\; \frac{\sum_i \lVert z_i + \hat d_i - z'_i\rVert^2}{\sum_i \lVert z'_i - z_i\rVert^2} \;+\; \lambda_{\text{sig}}\,\mathrm{SIGReg}(z).$$
Intent: the loss is now a latent-$R^2$-like quantity, so shrinking $z$ scales numerator and denominator equally and buys nothing.

### V2 — detached denominator

$$\mathcal{L}_{\text{V2}} \;=\; \frac{\sum_i \lVert z_i + \hat d_i - z'_i\rVert^2}{\mathrm{sg}\!\left[\sum_i \lVert z'_i - z_i\rVert^2\right]} \;+\; \lambda_{\text{sig}}\,\mathrm{SIGReg}(z).$$
Intent: V1 lets the encoder manipulate the denominator; detach it so it acts as a pure scale constant.

### V3 — batch-standardised latent

Standardise per dimension using statistics shared by both endpoints, with gradients flowing through (BatchNorm-style):
$$\mu = \mathbb{E}_B\!\big[[z; z']\big], \quad \sigma = \mathrm{std}_B\!\big[[z; z']\big], \quad \tilde z = \frac{z - \mu}{\sigma}, \quad \tilde z' = \frac{z' - \mu}{\sigma},$$
then apply V1's ratio to $\tilde z, \tilde z'$. Intent: make the objective exactly invariant to the encoder's output scale, so neither shrinking nor inflating changes it at all.

---

## 4. Results

Effective rank is defined as the exponential of the entropy of the normalised squared singular-value spectrum of the centred latent (see §5.1). Computed on 8,192 training states in float64. Random init gives ${\approx}49$.

| objective | seed | per-dim std (init → final) | min per-dim std | **effective rank (init → final)** | held-out latent $R^2$ | SIGReg (final) |
|---|---|---|---|---|---|---|
| V1 attached | 0 | 0.048 → 1.03 | 2.4e-2 | **49.2 → 1.1** | **+0.336** | 46.6 |
| V1 attached | 1 | 0.045 → 0.51 | 2.9e-2 | **47.3 → 1.0** | **−0.142** | 374.8 |
| V2 detached | 0 | 0.048 → **1.1e-3** | 2.8e-4 | 49.2 → 2.9 | −0.245 | 59.2 |
| V3 standardised | 0 | 0.048 → 98.6 | 7.1e+1 | **49.2 → 1.9** | **+0.409** | 15.4 |

Reference points: SIGReg floor on true $\mathcal{N}(0,I)$ at this batch size is **1.08**. Stage-1-only encoders reach effective rank ${\approx}7$; a frozen randomly-initialised encoder holds ${\approx}40$.

**Every objective degenerates, in one of two ways, and the two are mutually exclusive:**

- V1 → the latent's *direction* collapses (rank $\to 1$) while its scale grows.
- V2 → the latent's *scale* collapses (std $\to 10^{-3}$, heading toward §2.1's $2\times10^{-6}$).
- V3 → scale is made irrelevant by construction; rank still collapses to 1.9.

**Held-out latent $R^2$ rises as rank falls.** The best $R^2$ we have ($+0.409$) comes from the most degenerate latent. It is also not sign-stable across seeds under V1 ($+0.336$ vs $-0.142$).

---

## 5. Analysis

### 5.1 Effective rank

For centred latents $Z_c \in \mathbb{R}^{n \times D}$ with singular values $s_1 \ge \dots \ge s_D$, let $p_k = s_k^2 / \sum_j s_j^2$. Then
$$\mathrm{erank}(Z) \;=\; \exp\!\Big(-\sum_k p_k \log p_k\Big) \;\in\; [1, D].$$
$\mathrm{erank} = 1$ means all variance lies along a single direction.

### 5.2 Why V1 concentrates variance in one direction

Write the loss as $\mathcal{L} = \mathcal{E}/\mathcal{D}$ with $\mathcal{E} = \sum_i \lVert z_i + \hat d_i - z'_i\rVert^2$ and $\mathcal{D} = \sum_i \lVert d_i \rVert^2$. The gradient with respect to any encoder parameter $\theta$ is
$$\frac{\partial \mathcal{L}}{\partial \theta} \;=\; \frac{1}{\mathcal{D}}\frac{\partial \mathcal{E}}{\partial \theta} \;-\; \frac{\mathcal{E}}{\mathcal{D}^2}\frac{\partial \mathcal{D}}{\partial \theta}.$$
The second term is a **reward for increasing $\mathcal{D}$** — i.e. for making the latent displacement *larger* — with a gain $\mathcal{E}/\mathcal{D}^2$ that is substantial whenever prediction is imperfect. The encoder therefore has two ways to lower the loss: predict better, or move more.

"Move more, and predictably" has an exact degenerate solution. Choose a unit vector $u$ and a scalar map $g$, and let the encoder produce
$$z = u\,\phi(X) \quad\text{with}\quad z' - z = u\,\big(g(a) \big),$$
i.e. concentrate all action-driven variation along $u$. Then $\hat d = W_b^\top[z;a;1]$ can reproduce $g(a)$ **exactly** with a linear map ($A = 0$, $B$ picking off $a$), so $\mathcal{E}\to 0$, while $\mathcal{D}$ can be made arbitrarily large by scaling $\phi$. This is rank 1, and it is the global optimum of V1 up to the SIGReg penalty. The measured $\mathrm{erank} \to 1.0$–$1.1$ is exactly this.

### 5.3 Why V2 shrinks

With $\mathcal{D}$ detached, the gradient is $\frac{1}{\mathcal{D}}\partial_\theta \mathcal{E}$ alone. Uniformly scaling the encoder output by $s$ scales $\mathcal{E}$ by $s^2$, and the gradient never sees the compensating change in $\mathcal{D}$ (which is recomputed, detached, on the next batch). So the objective is locally minimised by $s \to 0$. This is §2.1's failure mode, reintroduced.

### 5.4 Why V3 still collapses

Per-dimension standardisation fixes $\mathrm{Var}(\tilde z_k) = 1$ for every $k$, which removes the scale degree of freedom entirely and does rule out both V1's and V2's exploits. **But it does not constrain the correlation structure.** A latent with
$$\tilde z = u\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0,1), \quad u \in \mathbb{R}^{256},\ |u_k| = 1 \ \forall k$$
has unit variance in every coordinate and rank exactly 1. Standardisation is a diagonal operation; rank is a property of the off-diagonal covariance. The measured $\mathrm{erank} = 1.9$ is consistent with precisely this.

The remaining unconstrained degree of freedom is therefore the **off-diagonal covariance** of $z$, and no variant tried so far penalises it.

### 5.5 Why SIGReg does not prevent this — and the sense in which it *does* see it

SIGReg evaluates marginal Gaussianity along random projections. For a rank-1 latent $z = u\varepsilon$, the projection onto a random unit $A_j$ is
$$\langle z, A_j\rangle \;=\; \varepsilon \,\langle u, A_j\rangle,$$
a Gaussian with standard deviation $|\langle u, A_j \rangle|\,\sigma_\varepsilon$. For random $A_j$ in $D=256$ dimensions, $\mathbb{E}[\langle u, A_j\rangle^2] = 1/D$, so almost every projection has variance ${\approx}\sigma_\varepsilon^2/256$ — far from the unit variance SIGReg tests against. **So the statistic is not blind to rank collapse:** a degenerate latent does produce a large SIGReg value, and indeed the measured values (15.4 to 374.8) sit 14× to 350× above the 1.08 floor.

The problem is not detection but **weighting**. At $\lambda_{\text{sig}} = 0.02$ the penalty contributes $0.3$ to $7.5$ to the loss while the transition term is $\mathcal{O}(1)$ and *falling*. The optimiser is paying the regulariser rather than satisfying it. In the successful Stage-1 runs the same statistic converges to ${\approx}1.5$, i.e. near its floor. **None of the runs in §4 is a fair test of Stage 3: they are tests of an under-regularised variant of it.**

### 5.6 Consequence for latent $R^2$ as a metric

Held-out latent $R^2$ is defined against the "do nothing" baseline *in the encoder's own latent space*:
$$R^2 \;=\; 1 - \frac{\sum_i \lVert z_i + \hat d_i - z'_i\rVert^2}{\sum_i \lVert z'_i - z_i \rVert^2}.$$
Both numerator and denominator depend on $E_\theta$. When the encoder is being optimised, $R^2$ is **not a measure of how well the dynamics are modelled** — it is a measure of how easy the encoder has made its own prediction problem. The degenerate solution of §5.2 drives $R^2 \to 1$ while discarding essentially all state information.

This is not a hypothetical concern: across the four runs, $R^2$ is *anti*-correlated with rank, and the highest $R^2$ ($+0.409$) belongs to the lowest-rank non-shrunk latent ($\mathrm{erank}=1.9$). **Latent $R^2$ is only meaningful when the latent is fixed a priori** (e.g. frozen encoder, closed-form $W$), which is how it was used in the Stage-1/Stage-2 experiments that preceded this.

---

## 6. Ruled out

Checked directly, not assumed:

- **SIGReg input shape.** It accepts $(T, B, D)$ and reduces over $B$; calling it with $T=1$ is correct and does not change the statistic.
- **Data/rasterisation.** Verified visually and numerically on the exact loader path: material leaves the push origin and arrives at the push tip, consistent at both resolutions; no axis transposition. Mass per frame $101.4$ px at $G=64$ (${\approx}5$ px per cube, matching $\pi r^2$ with $r=1.25$ px) and $23.3$ px at $G=32$.
- **Operator initialisation.** $W = 0$ gives $\mathcal{L} = 1.0$ at step 0, as observed.
- **Encoder normalisation.** GroupNorm, not BatchNorm, so train/eval monitoring is not a confound.

## 7. Known weaknesses of the evidence

- One seed for V2 and V3; two for V1. The rank collapse reproduces across the seeds tried, but the $R^2$ values do not.
- $8$ epochs $\times 250$ steps at $G=32$ on 96 of 192 available training files — sized to answer "does it collapse", not to produce a usable encoder.
- **The signal is sparse**: mean $|X_{t+1} - X_t| = 0.0121$, i.e. ${\approx}1.2\%$ of the frame changes per transition, and $8.2\%$ of transitions leave the occupancy *bit-identical* (the push missed the pile). At $G=32$ a cube spans $1.25$ px, so a 20-cube pile is ${\approx}23$ lit pixels — close to an isolated-dot point cloud.
- No `slateN` has been measured for any Stage-3 encoder. All numbers above are diagnostics, and §5.6 argues the main one is unreliable.

---

## 8. The questions we would like input on

1. **Is an explicit decorrelation term the right fix?** The natural candidate is a VICReg-style off-diagonal covariance penalty,
   $$\mathcal{L}_{\text{cov}} = \frac{1}{D}\sum_{k \neq l} \big[\mathrm{Cov}(z)\big]_{kl}^2,$$
   or whitening $z$ before the transition loss. Is there a reason to prefer one in a setting where the *downstream* use is a linear dynamics model (whitening changes the geometry the linear map operates in)?

2. **Or is the scale-free formulation itself the mistake?** An alternative is plain MSE with a $\lambda_{\text{sig}}$ large enough that SIGReg actually binds (driving the latent to isotropic unit-scale, which fixes both scale *and* rank as a side effect). This is closer to the original design. Is there a principled way to set $\lambda_{\text{sig}}$ other than sweeping until the statistic approaches its floor?

3. **Is a stop-gradient target sufficient here?** We detach $z'$, but both $z$ and $z'$ come from the *same* encoder, so the representation can still co-adapt. Would an EMA target encoder (BYOL-style) change the analysis in §5.2?

4. **Is the task itself too weak a learning signal?** With ${\approx}1.2\%$ of pixels changing per transition and $8.2\%$ of transitions being no-ops, is a transition-prediction objective on this data expected to produce a useful representation at all, or should the representation be learned from a different objective and the dynamics fitted afterwards?

---

## 9. Reproduction

| | |
|---|---|
| training script | `experiments/EXP-0021-five-way-ranking-comparison/code/train_transition_gate.py` |
| flags | V1: default · V2: `--detach-denom` · V3: `--standardise-z` |
| invocation | `--res 32 --epochs 8 --seed {0,1}` |
| raster check | `experiments/EXP-0021-five-way-ranking-comparison/code/viz_transition_raster.py` → `results/transition_raster_check.png` |
| logs | `experiments/temp/exp0021-transition-gate{,-s1,-detach,-std}.log` |
| checkpoints + per-epoch history | `experiments/temp/exp0021-transition-gate*/` (`encoder_transition.pt`, `history.json`) |
| encoder | residual CNN, $G{\to}$ channels 64/128/256, GroupNorm+SiLU, GAP, linear to $D{=}256$ |
| SIGReg | vendored, `le-wm/module.py`, 17 knots, 1024 projections |
