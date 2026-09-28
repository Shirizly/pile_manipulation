The idea:

> **A retrieval-based, action-conditioned local dynamics model for pile manipulation: retrieve transitions from a physically restricted, action-centric memory, then transfer the observed local state change onto the query state.**

This sits squarely in the older families of **memory-based learning, locally weighted models, instance-based action models, and analog forecasting**, but your application creates an unusually interesting version because the state is a spatial material configuration and the relevant state support is itself action-dependent. Memory-based control has a long robotics lineage, including Moore's early nearest-experience approach, locally weighted MPC, and kernel/nonparametric MPC. ([Cambridge Comp Lab][1])

---

# 1. Formalize the model first

The basic version is

$$
s' = s + \Delta(s,a)
$$

with a database

$$
\mathcal D =
\{(s_i,a_i,s'_i,\Delta_i)\}_{i=1}^N.
$$

For a query \((s_q,a_q)\), retrieve

$$
i^* =
\arg\min_i d\big(\Phi(s_q,a_q),\Phi(s_i,a_i)\big)
$$

and predict

$$
\hat{s}'_q = s_q + T_{i^*\rightarrow q}(\Delta_i).
$$

The transformation \(T\) is important. In your setup it should mostly be the transformation between the **donor's action-centric coordinate frame and the query's action-centric coordinate frame**.

The naïve implementation is therefore already interesting:

```text
query state + query action
        │
        ▼
action-centric representation
        │
        ▼
hard physical restrictions
        │
        ▼
nearest transition
        │
        ▼
donor Δ
        │
        ▼
transform Δ into query frame
        │
        ▼
query state + transformed Δ
```

But I would make two changes immediately.

### First: retrieve more than one transition

Treat 1-NN as the simplest baseline, not the final method:

$$
\hat{\Delta}
=
\frac{\sum_{i\in \mathcal N_k}w_i\Delta_i}
{\sum_i w_i}.
$$

This gives you three useful things simultaneously:

* a less noisy prediction,
* a principled way of testing whether the local transition function is smooth,
* an extremely cheap **uncertainty/coverage measure**.

For example,

$$
U(q)=\operatorname{Var}_{i\in\mathcal N_k}(\Delta_i)
$$

and the distance to the nearest neighbor become natural predictors of model failure.

That latter point is particularly attractive given your larger benchmarking idea.

### Second: don't make the final representation globally translationally invariant

Your existing warped push-frame idea is exactly the right direction. Transform the transition into a canonical frame based on the action, but preserve physical scale.

For a straight push:

* origin = push start or contact/reference point;
* x-axis = push direction;
* y-axis = perpendicular direction;
* action length remains a physical scalar;
* tool width/length remain explicit;
* speed/duration remain explicit if they matter.

Then the retrieval problem becomes something closer to

$$
\text{“Have I seen this *kind of local pile configuration under this kind of push* before?”}
$$

rather than

$$
\text{“Does this whole 12 cm × 12 cm image look similar?”}
$$

That is a much more plausible nonparametric model.

---

# 2. The hardest part is your "relevant state" idea

I think your concern here is exactly right:

> **the relevant state is not equivalent to the set of pixels/objects that moved.**

And I would actually make this one of the central methodological contributions.

For a rigid body pushing an isolated object, the relevant region may be approximately local to the swept volume.

For a compact granular pile, however, a stationary object can matter because it can:

* block another object;
* provide a reaction force;
* constrain a pile;
* create a contact chain;
* be a boundary condition;
* alter how the pushed material redistributes.

So I would define the concept as the **causal support** or **interaction support** of an action, rather than "changed region."

---

# 3. Construct an action-dependent interaction support

I would build this hierarchically.

Let \(S(a)\) be the swept region of the tool.

### Level 0 — direct interaction

Identify objects intersecting the swept volume:

$$
C_0 = \{o_j : o_j \cap S(a)\neq\emptyset\}.
$$

These are unquestionably relevant.

### Level 1 — mechanical support/contact neighborhood

Construct an object proximity/contact graph.

For example:

$$
G=(V,E)
$$

where an edge means two objects are sufficiently close that mechanical interaction is plausible.

Then expand outward from \(C_0\):

$$
C_1 = \operatorname{neighbors}(C_0).
$$

For a compact clump you may need multiple hops.

Importantly, this can include objects that **do not move at all** in the observed transition.

### Level 2 — geometric context

Add a spatial buffer around the interaction/support region.

Something like

$$
R = \operatorname{dilate}(C_1,r)
$$

where \(r\) is expressed in object/tool diameters rather than pixels.

This catches near contacts that may change the contact mode.

### Level 3 — boundary context

Explicitly preserve nearby:

* walls,
* corners,
* container boundaries,
* obstacles.

These can be particularly important for pile manipulation because they are static but mechanically consequential.

This gives you something like:

```text
                 context shell
       ┌───────────────────────────┐
       │                           │
       │    support/contact       │
       │       region              │
       │         ┌───────┐         │
       │         │direct │         │
       │         │contact│         │
       │         └───────┘         │
       │                           │
       └───────────────────────────┘
```

But I would **not collapse this to a binary mask**.

Instead, retain three channels/weights:

```text
direct interaction     weight 1.0
mechanical support     weight 0.5
context/buffer         weight 0.1–0.2
```

and make the retrieval metric weighted accordingly.

---

# 4. An extremely important detail: don't replace irrelevant state with zero

This is one place I think the first implementation could easily go wrong.

Suppose the local relevant area is

```text
██████████
████░░░░██
████░░░░██
██████████
```

and you literally erase everything outside the region by setting it to empty.

You have now created a new physical state in which those regions contain nothing.

That's potentially disastrous.

Instead:

### Keep the original state intact.

Use the mask **only for the distance calculation**.

So

$$
d(s_q,s_i)
=
\sum_x M(x)\,d_x(s_q(x),s_i(x)).
$$

Outside \(M\), the distance contribution is zero/ignored rather than "empty."

Likewise, when transferring the delta:

$$
\hat{s}'(x)=
\begin{cases}
s_q(x)+\hat{\Delta}(x),&x\in M\\
s_q(x),&x\notin M.
\end{cases}
$$

That gives you an actual localized transition operator rather than a cropped state model.

This distinction will matter enormously in the pile setting.

---

# 5. Don't assume that "changed pixels" are the right answer

You have a particularly nice opportunity in simulation.

Use the simulator to discover what the actual causal support looks like.

For a transition

$$
(s,a)\rightarrow s'
$$

you can perform counterfactual experiments:

1. remove/perturb some object or spatial region;
2. replay the same action;
3. measure how much the resulting \(\Delta\) changes.

Define, for example,

$$
I(R;a,s)
=
d\left(
\Delta(s,a),
\Delta(s\setminus R,a)
\right).
$$

If removing region \(R\) produces essentially no change in the transition, that region was not causally important for that action.

This lets you compare:

**heuristic relevance**

versus

**actual physical relevance.**

That's potentially a very strong experiment in its own right.

You could discover something like:

```text
changed-region heuristic
        ↓
direct swept-volume
        ↓
contact graph
        ↓
contact graph + spatial shell
        ↓
counterfactual causal support
```

and measure the prediction/sample-efficiency improvement at each stage.

That is substantially more interesting than simply reporting that kNN works.

---

# 6. I would make contact mode a hard retrieval constraint

There is an important problem with nearest-neighbor regression in granular manipulation:

Two states can be visually very similar while having different **contact modes**.

For example:

```text
A:       ███
       █████

B:       ███
       █████
```

may appear almost identical at 32×32 resolution, while in A the tool touches the upper object and in B it contacts the lower one.

The resulting transitions can be qualitatively different.

So I would have retrieval occur in two stages.

### Stage 1: hard compatibility

Reject transitions that differ in things such as:

* tool geometry;
* push type;
* action direction class;
* approximate action length;
* whether the tool contacts material;
* number/location of initial contacts;
* wall/corner interaction;
* perhaps material/object type.

### Stage 2: nearest-neighbor similarity

Within the compatible subset, compare the local state.

This is much better than allowing an extremely similar but physically incompatible transition to win.

The robot-pushing literature already has closely related ideas: local models assigned to different contact locations and case-based models conditioned on interaction circumstances. The survey literature on robot pushing explicitly discusses case-based prediction and the rationale of learning locally around familiar interaction configurations. ([Frontiers][2])

---

# 7. Representation for the retrieval metric

For your particular setup, I would test three increasingly sophisticated representations.

### Baseline A — occupancy image

Canonical 32×32 or 64×64 occupancy/density image.

Similarity:

$$
d_{\text{occ}} = \|M\odot(s_i-s_q)\|_2^2.
$$

This is deliberately dumb and useful as a baseline.

### Baseline B — signed/distance representation

Convert occupancy to a signed distance field or distance transform.

This makes small boundary displacements less catastrophic.

Then:

$$
d_{\text{SDF}}
=
\|M\odot(\operatorname{SDF}(s_i)-\operatorname{SDF}(s_q))\|_2^2.
$$

For clumps this may be considerably better than pixelwise occupancy.

### Baseline C — structured interaction descriptor

Use:

```text
local occupancy
+ tool-relative occupancy
+ occupancy along swept path
+ leading-edge occupancy
+ density profile across push direction
+ wall distances
+ contact count
+ local connected-component statistics
```

This has a useful scientific purpose: you can determine whether sophisticated retrieval representations are actually necessary.

I would **not initially learn the embedding**. A learned embedding turns this from a clean nonparametric baseline into another representation-learning problem, and it becomes harder to determine what makes the method work.

---

# 8. The delta representation deserves its own ablation

Your proposed

$$
s' = s+\Delta
$$

is probably the right first implementation.

But there is an important issue.

Suppose donor state A is:

```text
  ████
 █████
```

and its delta corresponds to a whole clump translating right.

Applying that raw image delta to a slightly different query pile can result in:

```text
original query
    +
donor pixel delta
    =
ghosts / holes / non-conservation
```

because image-space differences are not necessarily portable.

So I would implement two versions.

### Version 1 — additive delta

Exactly your idea:

$$
\hat{s}'=s_q+T(\Delta_i).
$$

This is the simplest and should be the primary baseline.

### Version 2 — transport/displacement delta

Represent the donor transition as a local displacement/transport field:

$$
u_i(x)
$$

and apply the corresponding deformation to the query state:

$$
\hat{s}'(x)
\approx
s_q(x-u_i(x)).
$$

This is probably more physically appropriate for rigid clumps and scattered particles.

In simulation you can obtain this essentially for free from object IDs and positions. For real data you may need segmentation/tracking/optical-flow-style estimation.

That gives you a nice progression:

```text
raw Δ
   ↓
spatially transformed Δ
   ↓
transport/displacement field
```

without changing the underlying retrieval concept.

---

# 9. Dataset design is probably more important than the index

I would build the dataset specifically around **state/action coverage**, rather than simply recording trajectories.

A transition record should contain approximately:

```python
Transition(
    state,
    action,
    next_state,

    canonical_state,
    canonical_next_state,
    delta,

    interaction_mask,
    contact_mask,
    support_mask,

    contact_signature,
    action_signature,

    episode_id,
    scene_id,
    material_id,
    object_count,
    pile_regime,
)
```

And importantly, I would **keep the original full state**.

The derived masks/features should be cached indexing information, not replacements for the raw transition.

---

# 10. In simulation, deliberately create branching datasets

This is something I would strongly recommend.

Instead of:

```text
state 1 → action 1 → state 2
                    ↓
                  action 2
                    ↓
                  state 3
```

use simulator branching:

```text
                 → action A → transition A
                /
state S ───────→ action B → transition B
                \
                 → action C → transition C
```

because your primary question is not just

> "Have we seen this state?"

It is

> "Have we seen this **local state under a comparable action**?"

For each starting configuration, sample many actions.

That makes the dataset vastly more useful for studying coverage.

For example, you could have:

```text
5,000 states
×
32 actions/state
=
160k transitions
```

rather than 160k largely correlated sequential transitions.

This also gives you a controlled way to study the effect of dataset size.

---

# 11. Your dataset should deliberately span the pile regimes

I would not rely on naturally occurring frequencies.

Create a stratified state distribution approximately along:

$$
\text{scattered}
\rightarrow
\text{partially interacting}
\rightarrow
\text{compact}
$$

with additional axes such as:

* object count;
* occupied area;
* density;
* connected-component count;
* distance to wall;
* pile aspect ratio;
* anisotropy;
* number of contact chains.

You can then ask:

> How many examples does this model require before each regime becomes predictable?

That directly supports one of your larger research goals.

---

# 12. The really interesting sample-efficiency experiment

Don't merely plot prediction error versus total number of transitions.

Measure **local coverage**.

For every held-out query, calculate:

$$
d_1(q)=\text{distance to nearest compatible transition}
$$

and perhaps

$$
d_k(q)=\text{distance to kth compatible transition}.
$$

Then plot:

$$
\text{prediction error} \quad\text{vs}\quad d_1
$$

and

$$
\text{prediction error} \quad\text{vs}\quad k\text{-NN dispersion}.
$$

You may find something like:

```text
nearest-neighbor distance
          ↓
prediction reliability
          ↓
rollout reliability
          ↓
MPC performance
```

That would connect almost perfectly to your broader benchmark idea.

---

# 13. This model naturally gives you a cheap confidence metric

Suppose the nearest five transitions are:

```text
Δ1
Δ2
Δ3
Δ4
Δ5
```

If they all predict almost the same thing:

$$
\operatorname{Var}(\Delta_i)\approx0
$$

you have strong empirical local support.

If they disagree substantially:

```text
Δ1 → →
Δ2 → 
Δ3 ↑
Δ4 ←
Δ5 ↗
```

the query is sitting in a poorly sampled or discontinuous part of state/action space.

Therefore you get two extremely cheap quantities:

$$
\boxed{\text{distance to known experience}}
$$

and

$$
\boxed{\text{disagreement among known experience}}
$$

without training an uncertainty network.

I would absolutely record these during MPC.

---

# 14. This also creates a very natural dataset-acquisition strategy

Once the system runs, you can identify queries for which:

$$
d_1(q) \text{ is large}
$$

or

$$
\operatorname{Var}(\Delta_i) \text{ is large}.
$$

Those become candidate experiments.

So your eventual system can use:

```text
retrieve
   ↓
prediction confidence
   ↓
if confidence low
   ↓
collect informative transition
   ↓
add to memory
```

This connects naturally to work on online/nonparametric MPC and active learning. Kernel/nonparametric dynamics have explicitly been used in MPC, and more recent work studies incorporating uncertainty and online data acquisition into MPC. ([Proceedings of Machine Learning Research][3])

---

# 15. How I'd implement the search efficiently

I would avoid an elaborate database initially.

The computation pipeline can be:

```text
Transition preprocessing
        │
        ├── canonical local image
        ├── action signature
        ├── contact signature
        └── metadata
                │
                ▼
          vector index
                │
                ▼
        candidate retrieval
                │
                ▼
       exact custom distance
                │
                ▼
          top-k neighbors
```

For the first version:

**FAISS or a GPU batched matrix search** is sufficient.

Don't prematurely optimize this.

At 64×64, however, you're dealing with 4096-dimensional descriptors. That isn't terrible, but it makes pure image-space kNN less attractive.

The hierarchy above solves that:

1. bucket/filter on action/contact signature;
2. retrieve approximate candidates;
3. calculate your expensive physically meaningful metric on those candidates.

For MPC, this is especially attractive because CEM already gives you batches of hundreds/thousands of queries. You can batch the retrieval operation.

---

# 16. A key issue: what happens during MPC rollouts?

This is probably the biggest conceptual problem with the approach.

At the beginning:

$$
s_0
$$

is a real state.

But after prediction:

$$
\hat{s}_1
$$

is synthetic.

Then you retrieve from

$$
(\hat{s}_1,a_1)
$$

and generate

$$
\hat{s}_2.
$$

So the model is recursively querying itself.

This can cause the classic distribution problem:

```text
real state
   ↓
small retrieval error
   ↓
synthetic state
   ↓
slightly farther from dataset
   ↓
bad retrieval
   ↓
larger error
   ↓
...
```

Therefore I would make **one-step coverage distance during rollout** a first-class logged quantity.

For every model rollout:

$$
d_t = d\big((\hat{s}_t,a_t),\mathcal D\big).
$$

Then see whether MPC failures occur when

$$
d_t
$$

gets large.

This could turn out to be much more meaningful than ordinary rollout MSE.

---

# 17. One very useful extension: local linear correction

I would implement this after 1-NN and kNN.

Instead of assuming

$$
\Delta_q=\Delta_i,
$$

assume local smoothness:

$$
\Delta(s,a)
\approx
\Delta_i + J_i
\begin{bmatrix}
s-s_i\\
a-a_i
\end{bmatrix}.
$$

Estimate \(J_i\) from the nearest \(k\) examples.

This is essentially the natural progression:

```text
1-NN
 ↓
k-NN weighted average
 ↓
kernel regression
 ↓
local linear regression
```

The literature on locally weighted models and kernel predictive control provides strong precedent for this family of approaches. ([ResearchGate][4])

But I suspect your pile dynamics may contain **mode discontinuities**, so local linearization could fail badly around contact transitions. That's a feature, not necessarily a bug: it would tell you how much of the apparent difficulty is due to local smoothness versus mode switching.

---

# 18. There is a particularly nice experiment around CEM vs gradient descent

Your observation that CEM is beating gradient descent becomes even more interesting with this model.

This retrieval model is inherently piecewise:

$$
\hat{\Delta}(s,a)
=
\Delta_{i^*(s,a)}.
$$

Therefore the model is approximately piecewise constant and has terrible/undefined gradients at retrieval boundaries.

Yet CEM can use it very naturally.

So you get an interesting experiment:

| Dynamics model  | Gradient planner | CEM |
| --------------- | ---------------: | --: |
| CNN             |                ✓ |   ✓ |
| switched-linear |                ✓ |   ✓ |
| retrieval       | ✗/poorly defined |   ✓ |

If retrieval performs well under CEM despite having no useful gradient, then the result strengthens the hypothesis that **accurate local outcome prediction may matter more than differentiability for your manipulation MPC problem**.

Conversely, if retrieval performs badly despite good one-step accuracy, that points directly to the distinction between **predictive accuracy and planning utility** that you're already interested in.

---

# 19. I would define the first experimental matrix like this

Don't try every possible sophistication simultaneously.

### Retrieval representation

```text
R1: global occupancy
R2: action-frame occupancy
R3: action-frame + interaction mask
R4: action-frame + contact/support/context hierarchy
```

### Retrieval

```text
K1: 1-NN
K2: weighted k-NN
K3: kernel regression
K4: local linear model
```

### Transition representation

```text
D1: additive Δ
D2: masked additive Δ
D3: transport/displacement
```

### Dataset

```text
N1: random trajectory data
N2: stratified states/actions
N3: branching state/action dataset
N4: progressively acquired dataset
```

You don't need the entire Cartesian product.

A very sensible first sequence is:

```text
R1 + K1 + D1
        ↓
R2 + K1 + D1
        ↓
R3 + K1 + D1
        ↓
R4 + K1 + D1
        ↓
R4 + K2 + D1
        ↓
R4 + K2 + D3
```

That will tell you where the value actually comes from.

---

# 20. Evaluation should be much richer than MSE

I'd divide evaluation into four levels.

### Level 1 — retrieval quality

Measure:

$$
d_1,\quad d_k,\quad kNN\ variance.
$$

Question:

> Does the dataset contain a genuinely similar experience?

### Level 2 — one-step prediction

Measure both global and task-relevant error.

For example:

$$
E_\Delta
=
\|M\odot(\hat{\Delta}-\Delta^*)\|.
$$

Also evaluate:

* mass conservation;
* moved-area overlap;
* centroid displacement;
* density redistribution;
* error near the goal-relevant region.

### Level 3 — rollout

Run fixed action sequences through the learned model and measure error versus horizon.

This diagnoses compounding.

### Level 4 — MPC

Run CEM MPC and measure:

* success;
* number of actions;
* execution time;
* planning time;
* total wall-clock time;
* task objective;
* number of failed/unproductive actions.

Then correlate levels 1–3 against level 4.

That's almost exactly the experimental structure you've been converging toward.

---

# 21. I would make "MPC utility of a transition" a possible metric

This could become more interesting than ordinary prediction error.

For a transition \(i\), ask:

> If I use transition \(i\) as the analog for query \(q\), how much does that change the quality of the action selected by MPC?

You can compare:

$$
a^*_{\text{true}}
$$

versus

$$
a^*_{\text{retrieval}}.
$$

Possible cheap surrogates include:

$$
\Delta V
=
V(s,a^*_{\text{true}})
-
V(s,a^*_{\text{retrieval}}).
$$

That is closer to what you actually care about than pixel prediction accuracy.

Eventually you may discover:

```text
pixel MSE       ─┐
SDF error        │
Δ error          ├──→ weakly related to MPC
goal error       │
NN distance      │
neighbor spread  ├──→ strongly related?
local mode error ┘
```

and *that* could be one of the strongest results of the overall project.

---

# 22. The dataset itself should become an experimental object

I would explicitly publish/evaluate **data coverage curves**.

For dataset size \(N\):

$$
N=1k,2k,5k,10k,20k,\ldots
$$

measure:

$$
\text{prediction error}(N)
$$

and

$$
\text{MPC performance}(N).
$$

But additionally measure:

$$
P(d_1 < \epsilon)
$$

for different \(\epsilon\).

Then you can say something much more informative than "we used 100k samples":

> At 20k transitions, 95% of task-relevant states have a compatible analog within distance \(d\).

That tells you **why** the model works and gives a meaningful measure of sample complexity.

This is one area where the old memory-based-control literature is conceptually relevant: the central benefit of storing experiences is precisely that prediction becomes fast when a sufficiently similar previous experience exists. ([Cambridge Comp Lab][1])

---

# 23. How I'd structure the actual implementation

I'd make it roughly:

```text
retrieval_dynamics/
    transition.py
    action_frame.py
    relevance.py
    representation.py
    distance.py
    index.py
    predictor.py
    uncertainty.py

dataset/
    raw/
    processed/
    index/

evaluation/
    one_step.py
    rollout.py
    coverage.py
    mpc.py
```

The important conceptual interfaces are:

```python
frame = action_frame(state, action)

support = relevance_model(
    state=state,
    action=action,
)

query_repr = representation(
    frame,
    support,
)

neighbors = index.search(
    query_repr,
    action_signature,
    support_signature,
)

prediction = predictor(
    query_state,
    query_action,
    neighbors,
)
```

This keeps "what counts as relevant" separate from "how similarity is measured," which I think will be very important experimentally.

---

# 24. I would not use a neural embedding initially

Given the work you've already been doing with learned representations, this is one experiment where I think the scientific value comes partly from **not learning everything**.

Start with:

$$
\boxed{
\text{physics-informed locality}
+
\text{simple representation}
+
\text{nonparametric retrieval}
}
$$

Then introduce a learned embedding only as an explicit later experiment.

Otherwise you won't know whether performance came from:

* more appropriate physical locality,
* better metric learning,
* more capacity,
* more data,
* or retrieval itself.

The cleanest story is initially:

> **Can a model with essentially no learned parameters predict granular manipulation dynamics when it is allowed to retrieve physically analogous experiences?**

That is a surprisingly strong baseline against your CNN.

---

# 25. The most important design decision I'd make now

I would define the method around **three nested notions of locality**:

$$
\boxed{
\text{action locality}
\rightarrow
\text{mechanical locality}
\rightarrow
\text{data locality}
}
$$

**Action locality:** only actions compatible with the requested interaction are candidates.

**Mechanical locality:** only state regions capable of influencing that interaction contribute to similarity.

**Data locality:** among those, retrieve transitions with the most similar local configuration.

That is much more principled than "crop around the tool."

And it gives you a very clear research question:

> **How much physical structure can be exploited to turn a global high-dimensional dynamics-learning problem into a local retrieval problem?**

---

## My recommended first prototype

I would actually make V0 extremely small:

$$
\boxed{
\text{64×64 action-frame occupancy}
+
\text{hard action/contact filtering}
+
\text{masked 1-NN}
+
\text{masked additive }\Delta
}
$$

Then evaluate it against your CNN and switched-linear models on exactly the same held-out transitions.

Immediately log:

```text
nearest-neighbor distance
prediction error
MPC performance
planning time
```

That will tell you whether the underlying idea has legs before you spend time on transport fields, learned metrics, or sophisticated contact inference.

The particularly promising part is that the **same dataset can simultaneously become the dynamics model, the sample-complexity benchmark, and the empirical uncertainty estimator**. That fits unusually well with the larger MPC benchmarking direction you've been developing.

[1]: https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.html?utm_source=chatgpt.com "Department of Computer Science and Technology – Technical reports: UCAM-CL-TR-209"
[2]: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.00008/full?utm_source=chatgpt.com "Frontiers | Let's Push Things Forward: A Survey on Robot Pushing"
[3]: https://proceedings.mlr.press/v144/maddalena21a.html?utm_source=chatgpt.com "KPC: Learning-Based Model Predictive Control with Deterministic Guarantees"
[4]: https://www.researchgate.net/publication/261416244_Locally_Weighted_Learning_Model_Predictive_Control_for_nonlinear_and_time_varying_dynamics?utm_source=chatgpt.com "(PDF) Locally Weighted Learning Model Predictive Control for nonlinear and time varying dynamics"
