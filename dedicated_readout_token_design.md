# Dedicated Readout Tokens for Sequential Recommendation: Design and Justification

## 1. Background and Motivation

### 1.1 Current model

The current model is a sequential recommendation model with a two-tower readout. Given a user behavior sequence $X = [x_0, \dots, x_{L-1}]$, a causal Transformer produces hidden states, and the output embedding at selected positions is scored against the item vocabulary by dot product:

$$
\text{logit}(y \mid \text{prefix}) = \langle h_t^{(N_\ell)}, \; e_{\text{out}}(y) \rangle
$$

where $h_t^{(N_\ell)}$ is the final-layer hidden state at position $t$, $N_\ell$ is the number of layers, and $e_{\text{out}}(\cdot)$ is the output item embedding table. Training computes loss only at the $K$ positive-interaction positions of each sequence; serving takes the readout embedding and performs ANN retrieval. The deployed configuration is shallow: $N_\ell = 5$ layers over sequences of length $L \approx 500$.

### 1.2 The dual-role problem

When the readout is taken at an item's own position, the residual stream at that position must serve two objectives simultaneously:

1. **Semantic representation.** As a key/value source, position $t$ must expose *what item $x_t$ is* to all later positions that attend to it.
2. **User-level summarization.** At its output, position $t$ must aggregate the entire prefix $x_{0:t}$ into a summary of the user's cumulative interests, sufficient to predict the next positive item.

Both objectives place gradients on the *same* $d$-dimensional residual stream $h_t$. The tension is especially sharp in recommendation: next-interest prediction depends on the **sum of all user interests**, not on the semantics of the current item — yet the readout is anchored to a specific item.

Formally, let $\mathcal{L}_{\text{sem}}$ denote the (implicit) objective of being a useful KV source and $\mathcal{L}_{\text{pred}}$ the prediction loss at position $t$. The parameter and activation updates satisfy

$$
\nabla h_t \;=\; \nabla_{h_t}\mathcal{L}_{\text{pred}} \;+\; \sum_{s > t} \nabla_{h_t}\mathcal{L}_{\text{pred}}^{(s)}\Big|_{\text{via attention } s \to t}
$$

The two gradient families pull $h_t$ toward different targets: the first toward the embedding of the *next* item, the second toward a faithful encoding of the *current* item. With only 5 layers, there is no depth budget to stratify the two roles across layers (shallow layers for semantics, deep layers for aggregation), so the conflict is head-on.

### 1.3 Identity leakage under dot-product readout

The two-tower readout makes the interference geometrically explicit. The residual stream decomposes as

$$
h_t^{(N_\ell)} \;=\; e_{\text{in}}(x_t) \;+\; \sum_{\ell=1}^{N_\ell} \Delta^{(\ell)}_t
$$

where $e_{\text{in}}(x_t)$ is the input embedding and $\Delta^{(\ell)}_t$ is the update written by layer $\ell$ (attention + FFN). The identity shortcut guarantees that a residue of $e_{\text{in}}(x_t)$ survives into the readout vector. The predicted logits therefore contain an additive term

$$
\langle h_t^{(N_\ell)},\, e_{\text{out}}(y)\rangle \;\supset\; \langle e_{\text{in}}(x_t),\, e_{\text{out}}(y)\rangle
$$

which biases retrieval toward items similar to the *current* item — a systematic **last-item similarity bias**. Two amplifiers apply in the current regime:

- **Shallow depth.** With $N_\ell = 5$, the stream is rewritten only five times; the relative magnitude of the $e_{\text{in}}(x_t)$ residue is far larger than in deep models.
- **Selective contamination.** Because loss is placed only at positive positions, the dual-role interference selectively corrupts the representations of exactly those items whose semantic KV quality matters most downstream; non-positive items remain clean. This asymmetry is itself a testable signature (§9.2).

### 1.4 Mechanistic evidence for the division-of-labor hypothesis

- **Logit-lens analyses** of causal LMs show deep-layer representations rotating away from the current token's semantics toward the prediction target — direct evidence that the two roles compete for the same stream.
- **Attention sinks** show that large models spontaneously repurpose low-information tokens as aggregation slots — i.e., the division of labor emerges on its own *when capacity permits*.
- **Register tokens in ViTs** (Darcet et al.) show that explicitly providing dedicated tokens cleans up the semantic representations of content tokens.

A 5-layer model is far below the scale at which such division emerges spontaneously. This design makes it explicit.

### 1.5 The inductive-bias account: why anchored readout suits language but not interest prediction

Sections 1.2–1.3 describe the failure mechanically. This section states the underlying cause at the domain level: **next-token prediction with anchored readout is an architecture whose built-in prior matches the statistical structure of natural language and mismatches that of interest prediction.** The argument has four layers.

**(i) Target-distribution structure.** Characterize a sequential domain by its mutual-information spectrum $I_k = I(y;\, x_{t-k})$ — the contribution of the history element $k$ steps back to the prediction target.

- *Language*: $I_k$ decays steeply in $k$. Local context carries the dominant share of next-token information ($p(y \mid x_t)$ is already a strong approximation of $p(y \mid x_{0:t})$; low-order statistics resolve much of next-token entropy).
- *Interest prediction*: the target is drawn from a user-level mixture of interests; $p(y \mid x_{0:t})$ behaves as a recency-modulated, weakly order-sensitive **aggregate functional** of the whole history, and $I_k$ is comparatively flat. $p(y \mid x_t)$ is a poor approximation of it.

This is a spectrum, not a dichotomy: session-based regimes have strong local structure (the raison d'être of Markov-chain recommenders such as FPMC), so the precise statement is that every domain mixes a **local-transition component** and a **global-aggregation component**, with language at the local extreme and interest prediction weighted toward global — the mixture weight varying by dataset.

**(ii) Architectural priors.** Anchored readout embeds a zero-layer direct path: the residual shortcut guarantees the logits contain $\langle e_{\text{in}}(x_t), e_{\text{out}}(y)\rangle$, i.e. a learnable transition matrix $W = E_{\text{in}} E_{\text{out}}^\top$ — the architecture ships a **first-order Markov prior for free** (the mechanistic-interpretability result that a Transformer's direct path computes bigram-like statistics). A dedicated readout token is instead a learned-query attention-pooling head: the lowest-complexity functions it reaches are global weighted aggregations, with no identity path from any single item. In standard terms, anchored readout makes local-continuation functions cheaply reachable; dedicated readout makes set-aggregation functions cheaply reachable. Language's target function lies in the former's low-complexity region; interest prediction's lies in the latter's. That is the mismatch, stated precisely.

**(iii) Interference intensity is domain-dependent.** The dual-role conflict of §1.2 is not uniformly severe. The gradient conflict between "represent $x_t$ faithfully" and "predict $y$" at the same stream scales with how far the two targets diverge:

$$
\text{interference} \;\propto\; D\big(p(y \mid x_t)\,\|\,p(y \mid x_{0:t})\big)
$$

In language the two roles are nearly aligned — the current token itself tightly constrains the next — so the dual role is approximately *one computation serving two purposes*, and anchored readout costs little. In recommendation the two are weakly related and the conflict is head-on. LLMs are thus not "getting away with" anchored readout; in their domain the two roles nearly coincide.

**(iv) Depth as a compensating resource.** Even under mismatch, depth compensates: each layer's rewrite dilutes the $e_{\text{in}}(x_t)$ residue, and many layers permit stratified division of labor, so a deep model can *launder out* the wrong prior with capacity (LLMs stack tens of layers under a dense per-position loss that symmetrizes the roles). A shallow model has no such budget — the cost of the mismatch is maximal precisely at small $N_\ell$. This yields a testable scaling prediction: the begin-vs-item gap should shrink monotonically with depth, consistent with §9.1 and risk §12.5.

**Retrospective closure with §13.** Under this account the Beauty/Steam split is not an anomaly: Beauty is a high-Markov-weight dataset (sparse, short histories — the traditional home turf of transition models), where the free transition prior is partly useful and its removal costs top-rank precision; Steam is aggregation-weighted, where the prior is pure bias and its removal wins uniformly. The corresponding remedy is not to re-contaminate the stream but to make the mixture explicit — re-inject the transition component as a gated additive term at the readout, $h_{b} + \alpha\,\tilde e_{\text{in}}(x_{t})$ with learnable $\alpha$ — turning the local/global mixture weight from an architectural accident into a per-dataset learnable quantity (a minimal hybrid of FPMC-style transition and user-level aggregation within this architecture).

**One-sentence summary.** *Next-token prediction with anchored readout hard-wires a first-order transition prior that matches the local statistical structure of language but mismatches the aggregate, weakly-local structure of interest prediction; a dedicated readout token replaces this prior with a global-pooling one, and gated re-injection makes the mixture between the two an explicit, learnable quantity.*

## 2. Core Design

### 2.1 Dedicated readout token

Introduce a special token `<begin>` serving as the exclusive user-level summarization site:

- **Item positions** are responsible only for their own semantics; they carry no prediction loss (baseline configuration).
- **The `<begin>` position** starts from a task-token embedding (no identity contamination from any item), and performs full-depth aggregation: at *every* layer it attends over the entire visible prefix. It is a readout head enjoying $N_\ell$ rounds of attention over the context — strictly more aggregation capacity than an item position's incidental summarization.

### 2.2 Readout and prediction

The output embedding at `<begin>` is scored against the item vocabulary by dot product, exactly as in the current two-tower model. The target is the next positive item after the fork point. Only the readout *location* changes; the retrieval mechanism, negative sampling scheme, and serving stack are untouched.

### 2.3 Branch collapse to a single token

Serving is one forward pass → one embedding → ANN retrieval; there is no autoregressive decoding. Hence each prediction branch needs exactly **one** `<begin>` token. Multi-step branch targets (predicting the next-next item) do not exist on the inference path and are demoted to an optional auxiliary loss. With branch length 1, the tree-attention machinery degenerates to "$K$ prefix-truncated query rows" — no intra-branch causality, no inter-branch structure beyond mutual isolation.

## 3. Training Construction Specification

### 3.1 Notation

- Sequence $X = [x_0,\dots,x_{L-1}]$; fork points = $K$ positive positions $t_1 < \dots < t_K$; target $y_i$ = next positive item after $t_i$.
- Physical layout: $[\,x_0,\dots,x_{L-1},\,b_1,\dots,b_K\,]$, total length $N = L + K$; branch token $b_i$ sits at physical index $L+i-1$. All $b_i$ share the same `<begin>` token ID.

### 3.2 Position IDs

$$
\text{pos}[j] = j \;\; (j < L), \qquad \text{pos}[L+i-1] = t_i + 1
$$

If timestamp encodings are used, set $\text{time}[b_i]$ to the request time of the fork (the timestamp of the event following $t_i$), matching serving-time semantics.

### 3.3 Attention mask

Define an **anchor function**

$$
f(j) = \begin{cases} j & j < L \\ t_i & j = L+i-1 \end{cases}
$$

The $N \times N$ visibility mask is generated by a single predicate:

$$
M[q,k] = 1 \iff \big(k < L \;\wedge\; k \le f(q)\big) \;\vee\; k = q
$$

This one predicate implies all required rules: standard causal attention within the context; $b_i$ sees only positions $\le t_i$; `<begin>` tokens are mutually invisible; context never sees any `<begin>`. The mask is derivable at data-collation time from the length-$N$ anchor vector $f$, with no dense matrix storage.

### 3.4 Relative position encoding

The model uses relative position embeddings. The scheme is compatible **provided** relative distances are computed from logical position IDs rather than physical indices:

$$
\text{rel}(q,k) \;=\; \text{pos}[q] - \text{pos}[k], \qquad \text{bias}(q,k) = B\big[\beta(\text{rel}(q,k))\big]
$$

where $\beta(\cdot)$ is the existing bucketing function and $B$ the learned bias table. Most implementations derive distances from $\text{arange}(N)$; this is the single required model-code change. Correctness rests on two properties:

1. **Collisions are harmless.** $b_i$ shares logical position $t_i+1$ with context token $x_{t_i+1}$, but the mask guarantees any colliding pair is mutually invisible.
2. **No out-of-distribution distances.** Within $b_i$'s visible set, $\text{rel} \ge 1$ with minimum exactly 1 (to the fork point) — identical, bucket by bucket, to the distance profile of a `<begin>` appended to a length-$(t_i{+}1)$ sequence at inference.

### 3.5 Labels and loss

Labels are assigned explicitly (never shift-by-one):

$$
\text{label}[L+i-1] = y_i, \qquad \text{label}[j] = \varnothing \;\; (j < L)
$$

Primary loss is the existing retrieval objective (sampled softmax / InfoNCE with the current negative-sampling scheme), applied at the $K$ `<begin>` positions:

$$
\mathcal{L} \;=\; \frac{1}{K}\sum_{i=1}^{K} -\log \frac{\exp\langle h_{b_i}, e_{\text{out}}(y_i)\rangle}{\exp\langle h_{b_i}, e_{\text{out}}(y_i)\rangle + \sum_{y^- \in \mathcal{N}_i} \exp\langle h_{b_i}, e_{\text{out}}(y^-)\rangle}
$$

Normalization (per-token vs. per-user averaging across variable $K$) must be fixed explicitly. An optional auxiliary next-item loss at item positions is discussed in §7 (it interacts with parameter sharing).

### 3.6 Worked example

$L=5$, $K=2$, forks at $t_1 = 1$, $t_2 = 3$:

```
physical index :   0    1    2    3    4    5    6
token          :  x0   x1   x2   x3   x4   b1   b2
pos_id         :   0    1    2    3    4    2    4
anchor f       :   0    1    2    3    4    1    3
label          :   -    -    -    -    -   y1   y2
```

Mask (rows = query, cols = key):

| Q \ K | x0 | x1 | x2 | x3 | x4 | b1 | b2 |
|---|---|---|---|---|---|---|---|
| x0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| x1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| x2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| x3 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| x4 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| b1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| b2 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |

Relative distances within visible regions: $b_1 \to [x_0,x_1,b_1] = [2,1,0]$; $b_2 \to [x_0,\dots,x_3,b_2] = [4,3,2,1,0]$ — element-wise identical to a `<begin>` at the end of a length-$(t_i{+}1)$ sequence.

### 3.7 Feature-time consistency

The attention mask blocks attention-level leakage only. Every side feature attached to an item (statistical features, popularity scores) must be an **event-time snapshot**, not the training-time latest value; otherwise the context visible to $b_i$ already encodes post-fork information. This is orthogonal to the mask and is the most common leakage source in recommendation.

## 4. Train/Serve Equivalence (the central correctness claim)

**Proposition.** For each branch $i$, the function computed at $b_i$ during training is *identical* to the function computed by standard causal inference on the sequence $[x_0,\dots,x_{t_i}, \texttt{<begin>}]$.

**Argument (induction over layers).** Let $V_i = \{x_0,\dots,x_{t_i}, b_i\}$ be $b_i$'s visible set. At layer 1, every position in $V_i$ computes attention over exactly the keys it would see in the standalone inference sequence, with identical relative distances (§3.4, property 2) and identical input embeddings; hence identical outputs. Inductively, if layer $\ell{-}1$ states agree on $V_i$, then layer $\ell$ attention at any $q \in V_i$ ranges over the same keys with the same states, weights, and position biases — so layer $\ell$ states agree. Context positions $\le t_i$ never attend to any `<begin>` or to positions $> $ themselves, so their states are unaffected by the presence of other branches. $\blacksquare$

Two design rules are load-bearing for this proof and must never be relaxed:

- **Branch isolation.** If any $b_i$ could attend to $b_j$ (or context to any $b$), $b_i$ would observe structure that does not exist at serving time, breaking equivalence.
- **Logical-ID relative distances.** Physical-index distances would give $b_i$ a distance profile never seen at inference.

**Consequence: no serving-time degradation from $K \to 1$.** Each $b_i$ trains as a fully self-contained reader — it never sees, exchanges information with, or shares loss with its $K{-}1$ siblings. Serving with a single `<begin>` at the sequence end reproduces, bit for bit, the computation any one branch performed in training. $K$ exists only in training-side accounting: one forward pass replays $K$ complete, independent inference snapshots of the same user at $K$ historical moments. The training regime never contains any mode other than "readout at the end of a prefix."

The only residual alignment question is **distributional**, not structural: the training distribution of prefix lengths (induced by fork sampling) should cover the serving distribution of user-history lengths. Forking at positive interactions naturally approximates this — each positive occurred immediately after a real request — and under-covered segments (very long or very short histories) can be patched by resampling, with no architectural change.

## 5. Throughput as Capability

Recommendation training operates in a near one-pass, streaming regime: data is effectively unbounded, each example is seen once, and model capability is governed by **data consumed per unit compute**. The multi-fork construction is therefore a capability multiplier, not an engineering nicety.

**Cost accounting.** Per sequence, attention cost is $O\!\big(L^2 + \sum_i t_i\big) \approx O(L^2 + K\bar{t})$ and FFN cost $O\big((L{+}K)\,d^2\big)$, versus $O(L^2)$ and $O(L\,d^2)$ for a single-fork forward. Since $K \ll L$, the marginal cost of $K{-}1$ extra supervision signals is a vanishing fraction of the forward pass:

$$
\frac{\text{signals per FLOP (multi-fork)}}{\text{signals per FLOP (single-fork)}} \;\approx\; \frac{K}{1 + O(K/L)} \;\approx\; K
$$

**Calibration.** The $K$ predictions share one context; gradients are correlated, and the effective sample size is a *discounted* $K$ (the $K$ positives of one user carry redundant information). Because marginal cost is near zero, "discounted $K\times$" still strictly dominates "full-price $1\times$"; the discount affects expected-gain calibration, not the decision.

**Evaluation corollary.** All comparisons must be aligned on **compute / wall-clock**, not update count: under a fixed compute budget the multi-fork model consumes several times more user sequences, and that difference is part of the method, because the binding production constraint is compute.

## 6. Depth Analysis and the Readout-Only Deep Extension

### 6.1 What `<begin>` does and does not add

Serial depth is architecturally bounded by $N_\ell$: layer $\ell$ attention reads only layer $\ell{-}1$ states of other positions, so the longest computation chain from any input to any readout is $N_\ell$, with or without `<begin>`. The token adds a **parallel dedicated stream**, not depth: structurally, $b_i$ is an $N_\ell$-layer learned-query cross-attention reader over the context's intermediate representations (a PMA/Perceiver-style aggregator grafted onto the causal backbone).

The real gain is **de-multiplexing the depth budget**. In the baseline, 5 layers are implicitly split between semantics and aggregation on one stream — each role gets effectively fewer than 5. After the split, item streams spend all 5 layers on semantics and the `<begin>` stream spends all 5 on aggregation: in the interference-limited regime, each role's usable depth nearly doubles at fixed $N_\ell$.

### 6.2 Readout-only additional layers: genuine serial depth, near-zero cost

A natural extension *does* increase serial depth: stack $n$ additional layers **after** the $N_\ell$-layer backbone in which only the `<begin>` tokens act as queries, cross-attending to the backbone's final-layer context states (plus FFN on the `<begin>` stream). The computation chain becomes: context semantics ($N_\ell$ serial layers) → readout aggregation ($n$ further serial layers), i.e., true depth $N_\ell + n$ *for the readout path*.

Cost per extra layer: $O(K \cdot \bar{t} \cdot d)$ attention + $O(K d^2)$ FFN — negligible against the backbone (roughly two orders of magnitude cheaper than adding a full backbone layer). At serving, it is $n$ single-query layers appended after the existing forward; the context computation is untouched. Branch isolation and anchor-based masking carry over unchanged into these layers (each $b_i$'s query attends only to keys $\le t_i$), preserving the equivalence proposition of §4.

For a capacity-limited 5-layer model whose *entire primary loss* flows through the readout path, this is likely the highest-leverage capacity addition available and should precede per-layer expert FFNs in the ablation queue.

## 7. Parameter Sharing for `<begin>`

Learned MoE routing is unnecessary: token type is known a priori, so any separation can be hard-routed by type with zero routing overhead. The real question is where on the sharing spectrum to sit.

**Default: fully shared parameters.** Precedent is strong (CLS, ViT registers, soft prompts all achieve functional specialization through a learned token embedding alone), and the interference argument of §1.2 concerns *activations* — one $d$-dimensional stream per position — whereas parameter-space conflict is diluted across millions of weights where two tasks can occupy disjoint subspaces.

**Coupling with the auxiliary loss.** With loss only at `<begin>`, the shared FFN's direct gradients are shaped almost entirely by the readout task — sharing is benign. Enabling the item-position auxiliary loss (§3.5) reintroduces dual-role pressure *at the parameter level*: the same FFN must learn both item-semantic transformation and aggregate-to-item-space mapping. These two knobs must therefore be ablated **jointly**, not independently.

**Separation spectrum, by cost:** token embedding (free, already present) → per-token-type bias/LayerNorm (near-free) → **dedicated readout MLP after the final layer** (cheap; the map "aggregate → item space" is a function only `<begin>` needs, and it occupies exactly the classical user-tower-head slot — zero serving-stack change; recommended first separation experiment) → hard-routed per-layer dual FFN (parameters ~2× on the Transformer body, but the parameter budget is dominated by the item embedding table, and compute increases only by $K$ tokens). For a small model, dedicated readout parameters also act as free capacity injected precisely into the path carrying all primary-loss gradients — possibly worth more than interference isolation itself.

## 8. Two Orthogonal Multiplicity Dimensions: $K$ and $h$

| | $K$ (fork points) | $h$ (interest heads) |
|---|---|---|
| Meaning | $K$ distinct historical moments of one user | $h$ readout tokens at the *same* fork |
| Purpose | throughput amortization (§5) | break the single-vector bottleneck |
| Mutual visibility | isolated (required by §4) | isolated by default (competition-driven specialization) |
| Loss coupling | independent, summed | competitive: $\min_j$ or softmax-weighted over $j$ |
| Serving fate | **collapses to 1** (each branch is self-contained) | **persists**: $h$ tokens, $h$ ANN queries, union |

The serving-fate criterion is general: a token dimension survives to serving iff its members were trained to be *complementary* (removing one loses coverage); it collapses iff members are mutually redundant replays. $K$-siblings never interact — dropping $K{-}1$ is lossless. $h$-heads are trained by min-loss competition into complementary interest coverage — all must ship.

Full form ($K \times h$ tokens $b_{i,j}$): $\text{pos}[b_{i,j}] = t_i+1$, $f(b_{i,j}) = t_i$, mask predicate unchanged except that the $k=q$ clause keeps same-fork heads mutually invisible; per-fork loss $\mathcal{L}_i = \min_j \ell(h_{b_{i,j}}, y_i)$ or $\sum_j w_j \ell_j$ with $w = \text{softmax}(-\ell/\tau)$; no negatives drawn among a user's own $K \times h$ queries. Recommended sequencing: land $K$ and readout migration with $h=1$; open $h$ as a second phase (it multiplies serving-side ANN cost, unlike $K$).

## 9. Why This Should Work Here: Regime-Specific Effectiveness Arguments

### 9.1 The 5-layer / 500-length regime favors the design

The scale-related caveat (spontaneous depth-wise role stratification eroding the gain) points at *large* models. In the deployed regime every amplifier points the other way:

1. **No depth to stratify.** Five layers cannot split into semantic and aggregation strata; the conflict is unavoidable without an architectural split.
2. **Large identity residue.** The stream is rewritten only 5 times, so $\|e_{\text{in}}(x_t)\| / \|h_t^{(5)}\|$ is large — the last-item bias term of §1.3 is maximal exactly here.
3. **High superposition pressure.** Recommendation models typically use modest $d$; two tasks crowd a small stream harder than a wide one.
4. **No spontaneous registers.** Sink/register phenomena emerge in large models; a 5-layer model will not self-organize this division — providing it explicitly is pure gain.
5. **Free capacity where it counts.** All primary-loss gradient flows through the readout path; dedicated readout depth/parameters (§6.2, §7) inject capacity precisely there at negligible compute.

One expectation to keep calibrated: compressing 500 events into one $d$-vector is a single-vector bottleneck that readout relocation does **not** solve — that is $h$'s job (§8). The gain ceiling of the base design is set by interference removal alone.

### 9.2 Falsifiable pre-experiment diagnostics (runnable on the current model)

1. **Identity-leakage magnitude.** Measure $\cos(\tilde h_t, \tilde e_{\text{in}}(x_t))$ at positive positions, where $\tilde{\cdot}$ denotes mean-centering (embedding spaces carry a dominant common direction that inflates raw cosines). Baseline: $\cos(\tilde h_t, \tilde e_{\text{in}}(x_s))$ for random $s$. A significant self-vs-random gap confirms the §1.3 mechanism and is the leading indicator of switching gains. Valid regardless of tying: $h_t$ and $e_{\text{in}}$ share the residual-stream space by construction.
2. **Table alignment (untied tables).** Measure $\cos(e_{\text{in}}(x), e_{\text{out}}(x))$ across the vocabulary. High alignment (common in recommendation, driven by repeat consumption and co-occurrence) means representation-level leakage converts directly into logit bias — and means untying is only nominal separation, demoting the tie/untie ablation. Near-orthogonality means leakage costs dimensional budget rather than causing direct bias.
3. **Behavioral repetition bias.** Compare the rate at which top-$K$ retrieval returns items similar to the sequence tail (same category, or $e_{\text{out}}$-space neighbors) against the ground-truth repeat/same-category consumption rate. The excess is the bias, measured assumption-free and closest to online behavior.

### 9.3 The core ablation is unusually clean

Current model ($K$ positive-position readouts) vs. this design ($K$ `<begin>` readouts): identical supervision count, identical targets, near-identical context forward cost; the *only* difference is readout location. Marginal cost of the new design is $K$ extra tokens — effectively free. Comparisons aligned on compute (§5).

## 10. Inference

1. Append a single `<begin>` to the user sequence (carrying current request time and optional request-side conditioning).
2. Standard causal mask; naturally incrementing positions; no special structure of any kind.
3. One forward pass; take the `<begin>` output embedding (through the readout-only layers of §6.2 if enabled — $n$ single-query layers appended to the existing forward).
4. ANN retrieval against the item vocabulary ( $h$ queries with union-merge if multi-interest is enabled).

The training-side multi-fork structure is invisible to serving.

## 11. Evaluation Plan

1. **Main comparison** (§9.3), compute-aligned, at $h=1$, fully shared parameters.
2. **Interference-hypothesis validation:** run §9.2 diagnostics on the baseline before and after switching; additionally test whether positive items' quality *as KV sources* improves (their usefulness as context in other sequences' retrieval).
3. **Component ablations,** in recommended order: (c′) **gated transition-prior re-injection** (§1.5): $h_b + \alpha\,\tilde e_{\text{in}}(x_t)$ with learnable $\alpha$ (scalar, then gated-vector variant) — restores the local-transition component explicitly without re-contaminating the stream; predicted $\alpha$: significantly positive on high-Markov-weight data, near zero on aggregation-weighted data; (a) readout-only deep layers $n \in \{0,1,2\}$ (§6.2); (b) dedicated readout MLP vs. fully shared (§7); (c) auxiliary item-position loss × parameter sharing, ablated jointly (§7); (d) $h \in \{1,2,4\}$ multi-interest (§8); (e) per-layer dual FFN; (f) tied vs. untied tables, contingent on diagnostic 9.2-(2).

## 12. Risks and Open Questions

1. **Indirect gradients on item representations.** With no direct loss at item positions, semantic representations are shaped only by "being useful when attended to by `<begin>`" — possibly too sparse; the auxiliary loss exists as the remedy but re-couples parameter roles (§7). Empirical question.
2. **Gradient correlation across forks.** The effective-sample-size discount on the $K\times$ throughput gain must be reflected when attributing wins between architecture and data volume.
3. **Prefix-length distribution coverage.** Fork sampling must cover the serving distribution of history lengths (§4), especially tails.
4. **Vocabulary non-stationarity.** New-item onboarding and cold start are orthogonal to this design, but readout relocation must not degrade existing cold-start machinery; requires regression validation.
5. **Scale sensitivity.** All §9.1 arguments are regime-specific; if the backbone is later scaled substantially deeper/wider, the net gain of explicit division should be re-measured.

---

## 13. Empirical Validation (2026-07-08, this repository)

This section records the first offline instantiation of the design: a clean §9.3-style A/B on public leave-one-out benchmarks, with the §4 equivalence proposition verified numerically and the §9.2 diagnostics run on both arms.

### 13.1 Experimental proxy for the production regime

| Design-doc element | Offline instantiation |
|---|---|
| Backbone | HSTU-style stack (SiLU pointwise attention, per-head relative position bias), 4 blocks, $d=256$, 4 heads, attn dim 128, FFN 512, dropout 0.2 |
| Two-tower readout | **Untied** learnable output item table $e_{\text{out}}$; scores by dot product |
| Sampled softmax | $M=1024$ uniform negatives per step, shared across the batch; negatives colliding with the row's own positive are masked to $-10^9$ |
| Sequence / forks | $L=20$ (left-padded), fork at **every** train-prefix position ⇒ $K \le 19$ per user; branch slots for pad positions are loss-masked (static shapes) |
| Loss normalization (§3.5) | Per-token averaging: $\sum \text{CE} / \#\{\text{label}>0\}$ per batch |
| Datasets | Amazon Beauty (sparse: 22,363 users, 12,101 items, 117k supervised positions) and Steam (dense: 334,325 users, 13,044 items, 2.31M supervised positions); chronological leave-one-out; full-catalog ranking eval (no sampled negatives at eval) |
| Optimizer | AdamW, constant LR $10^{-3}$, wd $10^{-4}$, batch 256 users, early stop on val NDCG@10 (patience 15/10 evals) |

Both arms share one model class, one data layout, one loss, one negative-sampling stream, and — by construction — **identical supervision positions and targets**. The only difference is the readout site:

- **Arm A `--readout item`** (baseline): tokens $[x_0..x_{L-1}]$, causal mask; $h_j$ predicts label $j$ — the item's own residual stream is the readout (§1.2's dual role).
- **Arm B `--readout begin`** (treatment): tokens $[x_0..x_{L-1}, b_0..b_{L-1}]$; $b_j$ anchors to context position $j$ (§3.3), carries logical position $j+1$ (§3.2), and predicts the same label $j$. Item positions carry no loss.

Code: `src/models/readout_hstu.py` (mask/logical-position-aware HSTU), `examples/train_readout_hstu.py` (data construction, both arms, diagnostics). Known deviations from the deployed regime the doc targets: $L=20$ vs 500, 4 blocks vs 5, no side features / timestamps (so §3.7 does not arise), fork density = every position rather than sampled positives. §9.1's amplifiers (long context, extreme summarization load) are therefore *weaker* here; effect sizes should be read as a lower bound on the regime the doc argues about.

### 13.2 Implementation notes (doc § → code)

- **§3.3 anchor mask** — one predicate generates both arms' masks: `M[q,k] = (k < L ∧ k ≤ f(q) ∧ x_k ≠ pad) ∨ k = q`, with `f = arange(L)` for arm A and `f = [arange(L), arange(L)]` for arm B. Branch mutual invisibility and context-never-sees-branch fall out of the `k < L` clause, exactly as claimed.
- **§3.4 logical-distance relative bias** — the HSTU relative-position table is indexed by `clip(pos[q] − pos[k], 0, 63)` where `pos = [arange(L), arange(L)+1]`; this was indeed the *single required model-code change* relative to the stock HSTU block (which uses physical `arange` differences).
- **§2.3 branch collapse** — serving-side readout is $b_{L-1}$ inside the same static layout; by branch isolation this is bit-identical to appending one `<begin>`, so no separate inference path was written.
- **Labels** are assigned explicitly per fork (never shift-by-one), per §3.5.

### 13.3 Train/serve equivalence — §4 proposition verified numerically

Scratch test (`test_readout_equiv.py`, random params, fp32 CPU, tolerance $10^{-5}$), all four checks pass:

1. **Context invariance**: hidden states of context positions in the full $[x, b_0..b_{L-1}]$ layout equal those of the plain causal $[x]$ layout — branches perturb nothing.
2. **Per-fork standalone equivalence**: for every $j$, $h_{b_j}$ in the full $K$-fork layout equals the hidden state of a lone `<begin>` in a layout containing only fork $j$ — training is literally $K$ independent inference snapshots.
3. **Mask spot-check**: generated mask rows reproduce the §3.6 worked example table exactly.
4. **Isolation**: the branch-block of the mask is the identity; the context→branch block is all zeros.

### 13.4 Pre-registered diagnostics (§9.2) — the mechanism is real

Mean-centered cosines, computed on 4,096 validation sequences with each arm's best checkpoint (3 seeds each, mean ± std):

| Quantity | Arm A (item) | Arm B (begin) | Random baseline |
|---|---|---|---|
| Readout vector vs $e_{\text{in}}(x_t)$, Beauty | **0.307 ± 0.013** | **0.028 ± 0.005** | ~0.000 |
| Readout vector vs $e_{\text{in}}(x_t)$, Steam | **0.589 ± 0.008** | **0.052 ± 0.009** | ~0.001 |
| In/out table alignment $\cos(e_{\text{in}}, e_{\text{out}})$, Beauty / Steam | 0.017 / 0.092 | 0.003 / 0.010 | — |

Three findings:

1. **§1.3's identity residue exists and is large.** The baseline's readout vector carries a strong additive component of the *current* item's input embedding — 0.31 on Beauty and 0.58 on Steam against a ~0 random baseline. The Steam value is striking: over half the (centered) readout direction is last-item identity. The 5-layer-shallow-model amplifier argument (§9.1-2) holds even at 4 blocks.
2. **The `<begin>` token removes it.** One architectural split drops the leakage by ~11× (0.31→0.03, 0.58→0.05). The residual ~0.03–0.05 is consistent with genuine "next item correlates with current item" signal rather than a shortcut, since the begin token has no identity path from $e_{\text{in}}(x_t)$ into its stream except through attention.
3. **§9.2-(2)'s regime split materialized.** The item arm *learns* nontrivial in/out table alignment on Steam (0.091, repeat-heavy consumption) but near-orthogonality on Beauty (0.017). Per the doc, Steam is therefore the "leakage converts directly into logit bias" regime — which is exactly where the treatment wins end-to-end (below). In the begin arm the alignment collapses to ~0: with a clean readout, the model no longer benefits from aligning the towers.

**Same-user control (post-hoc, addressing an attribution gap in §9.2-1).** The random-item baseline cannot separate *positional contamination* (H1, the §1.3 shortcut) from a *legitimate interest summary* (H2): the user's history items are mutually similar, so a faithful summary would also show elevated cosine against the last item. The discriminating control is the recency profile — centered cos(readout, $e_{\text{in}}(x_{L-1-k})$) for $k = 0..7$ over the **same user's** history (seed-42 checkpoints, 4,096 val sequences; popularity-matched random baseline ≈ uniform ≈ 0):

| Arm / dataset | k=0 (anchored item) | k=1 | k=2 | k=3 | k=5 | k=7 |
|---|---|---|---|---|---|---|
| item, Beauty | **0.325** | 0.044 | 0.041 | 0.032 | 0.022 | 0.014 |
| begin, Beauty | 0.032 | 0.015 | 0.011 | 0.008 | 0.004 | 0.003 |
| item, Steam | **0.583** | 0.144 | 0.085 | 0.070 | 0.051 | 0.044 |
| begin, Steam | 0.054 | 0.020 | 0.011 | 0.006 | 0.002 | −0.002 |

The item arm shows a **discontinuous 4–7× jump exactly at the anchored position** ($k=0$ vs $k=1$: 0.325 vs 0.044 on Beauty, 0.583 vs 0.144 on Steam), sitting on top of a smooth recency decay for $k \ge 1$; the begin arm's profile is smooth everywhere (its $k=0/k=1$ ratio ≈ 2–3 matches the natural decay slope). Decomposing: of the item arm's raw leak, the same-user/recency component accounts for only ~0.05 (Beauty) / ~0.15 (Steam) — the **position-anchored excess is ~0.28 / ~0.44**, i.e. H1 dominates and the headline numbers above survive the stronger control essentially intact. A secondary observation: even at $k \ge 1$ the item arm sits well above the begin arm (0.144 vs 0.020 at $k=1$ on Steam). With untied towers a clean summary has no reason to live in $e_{\text{in}}$-space at all (scoring is against $e_{\text{out}}$) — the begin arm's near-zero profile confirms this, and the item arm's elevated tail is best read as the identity component *reflected through item–item similarity*, not as richer interest encoding.

Additional unanticipated observation: the *item positions'* own streams retain **less** identity in the begin arm (last-item stream cos 0.14 ± 0.01 vs 0.31 ± 0.02 on Beauty; 0.35 vs 0.58 on Steam). With no prediction loss anywhere on the context, item streams are shaped purely by being useful KV sources for `<begin>` queries (risk §12.1) — apparently this pushes them toward *relational* rather than identity-preserving encodings. Not the failure mode feared in §12.1 (metrics did not collapse), but worth tracking at scale.

### 13.5 Main A/B results

**Beauty (sparse), 3 seeds (42/123/7), mean ± std, full-catalog test:**

| Arm | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | MRR | best val NDCG@10 | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| item | 0.03542 ± 0.0009 | 0.02542 ± 0.0007 | 0.04907 ± 0.0004 | 0.02980 ± 0.0004 | 0.06705 ± 0.0019 | 0.03432 ± 0.0005 | 0.02743 ± 0.0006 | 0.04016 ± 0.0001 | 16.3 |
| begin | 0.03455 ± 0.0003 | 0.02411 ± 0.0003 | **0.04964 ± 0.0006** | 0.02898 ± 0.0004 | **0.07010 ± 0.0009** | 0.03411 ± 0.0004 | 0.02659 ± 0.0004 | **0.04225 ± 0.0004** | 27.7 |
| Δ | −2.4% | −5.2% | +1.2% | −2.7% | **+4.6%** | −0.6% | −3.1% | **+5.2%** | +11 epochs |

**Steam (dense), 3 seeds (42/123/7), mean ± std, full-catalog test:**

| Arm | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | MRR | best val NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| item | 0.16734 ± 0.0006 | 0.14641 ± 0.0004 | 0.20011 ± 0.0006 | 0.15694 ± 0.0004 | 0.24651 ± 0.0005 | 0.16859 ± 0.0004 | 0.15253 ± 0.0003 | 0.18170 ± 0.0002 |
| begin | **0.16859 ± 0.0004** | **0.14696 ± 0.0002** | **0.20246 ± 0.0004** | **0.15784 ± 0.0002** | **0.25062 ± 0.0003** | **0.16995 ± 0.0001** | **0.15321 ± 0.0001** | **0.18265 ± 0.0001** |
| Δ | +0.7% | +0.4% | +1.2% | +0.6% | +1.7% | +0.8% | +0.4% | +0.5% |

The Steam result is *per-seed uniform*: the begin arm beats the item arm on **all 7 test metrics in every one of the 3 seeds**, and the HR@10 gap (+0.0024) is ~4× the pooled seed std — outside noise.

**Cost accounting (§5, §9.3):** the begin arm adds $L$ branch tokens (2× tokens, but branch rows are single-query attention). Measured wall-clock per epoch: Beauty 12.7s → 13.1s (+3%), Steam 20.4s → 25.4s (+25%). Supervision count is identical by construction, so no compute-alignment discount is needed for this comparison (the §5 throughput multiplier is *not* exercised here — both arms are multi-fork; it would apply against the repo's legacy one-prefix-per-sample trainer, which sees each user $K$ times per epoch through $K$ full forwards).

### 13.6 Reading of the results

1. **On the dense / high-repeat dataset (Steam), the treatment wins uniformly** — all seven test metrics and validation, in all 3 seeds, at matched best-epoch (~12 for both arms). This is the dataset where the diagnostics say the baseline's leakage actually distorts logits (readout leak 0.59, table alignment 0.09). Direction and mechanism agree with §1.3.
2. **On the sparse dataset (Beauty), the outcome is a recall/precision trade**, consistent across all 3 seeds: deep-rank recall improves (HR@20 +4.6%, HR@10 +1.2%) while top-rank precision degrades (NDCG@5 −5.2%, MRR −3.1%). Interpretation: the identity residue functions as a built-in *last-item similarity prior*. On sparse data with short histories, "recommend things similar to the last item" is a strong top-rank heuristic, so removing it costs precision; what the clean summary buys instead is broader interest coverage at deeper ranks. This is precisely the single-vector-bottleneck boundary drawn in §9.1's calibration note — interference removal helps coverage, but the missing similarity prior is a *feature* the baseline got for free. The natural follow-ups are the doc's own: an auxiliary item-position loss (§7) or $h>1$ interest heads (§8) to recover top-rank sharpness without re-contaminating the readout.
3. **Validation vs test divergence on Beauty** (val NDCG@10 +5.2% for begin, all seeds; test NDCG@10 −2.7%) matches this campaign's previously documented val→test composition shift on Beauty and again suggests the treatment's gains concentrate on prediction-at-the-current-frontier rather than on the popularity-shifted test slice.
4. **Convergence**: the begin arm peaks ~11 epochs later on Beauty at a higher validation optimum — consistent with learning a harder function (true aggregation) instead of leaning on the identity shortcut.
5. **Caveat**: absolute numbers are below the repo's legacy full-softmax HSTU rows (e.g. Beauty HR@10 0.049 vs 0.065) because sampled softmax with $M=1024$ of 12k items is a weaker training signal. This is deliberate — the experiment models the production sampled-softmax regime — but means arm-vs-arm deltas, not absolute values, are the object of interest.

### 13.7 Verdict and next steps

The load-bearing claims of the design survive contact with data: the §4 equivalence holds exactly; the §1.3 identity-leakage mechanism is confirmed and quantitatively large; the anchor-mask construction trains stably at negligible extra cost; and on the dataset whose diagnostics match the doc's "leakage → logit bias" regime, the treatment wins across the board. The sparse-data precision regression identifies the first production-relevant risk not emphasized in §12: **the last-item similarity bias being removed is partly a useful prior** — in §1.5's terms, Beauty carries a high local-transition mixture weight whose free prior the treatment discards. Recommended follow-up order, per §11.3 and these results:

1. **Stratified test analysis (zero training cost, do first):** split test targets by whether they fall in the last item's $e_{\text{out}}$-neighborhood / category; prediction — item arm wins the "target ≈ last item" slice, begin arm wins the complement. This directly tests the §13.6-2 interpretation and quantifies how much of the Beauty delta is metric composition (leave-one-out rewarding the transition prior) rather than capability.
2. **(c′) gated re-injection** $h_b + \alpha\,\tilde e_{\text{in}}(x_t)$ (§1.5, §11.3): a one-line change that restores the transition prior as an explicit learnable mixture instead of re-contaminating the stream via gradients; predicted to recover Beauty top-rank precision with $\alpha > 0$ while leaving Steam unaffected with $\alpha \approx 0$ — which would unify the two datasets' outcomes as two values of one parameter.
3. **(c) auxiliary item-position loss × parameter sharing** (joint, §7) if $\alpha$-re-injection is insufficient, then (a) readout-only deep layers, then (d) $h \in \{2,4\}$.

Additionally, the same-user recency-profile control of §13.4 proved strictly stronger than the random-item baseline it was designed to supplement and should be promoted into §9.2 as the standard leakage diagnostic. One calibration note on §13.1's "lower bound" reading: it holds for the *leakage-removal benefit* (the §9.1 amplifiers strengthen at $L=500$, 5 layers), but not automatically for the *net metric delta* — the single-vector summarization load also grows with $L$ while the baseline's implicit transition prior does not, so the sparse-regime precision cost may scale up alongside the gains, raising rather than lowering the priority of (c′) and $h>1$ at production scale.

### 13.8 Stratified test analysis (§13.7-1) — the mixture decomposition is real

Zero-training follow-up executed 2026-07-08. Test targets are split by two **arm-independent** definitions of "target ≈ last item":

- **bigram slice**: the (last item → target) pair occurs at least once as consecutive events in the training prefixes — the domain of a first-order transition model;
- **semantic slice**: target is among the last item's top-50 cosine neighbors in the frozen t5-XXL *text* embedding space (external to both arms' learned parameters).

Metrics per slice, 3 seeds averaged, full-catalog ranks:

**Beauty** (bigram slice = 8.6% of test, semantic slice = 11.9%):

| Slice | n | Arm | HR@5 | NDCG@5 | HR@10 | HR@20 | MRR |
|---|---|---|---|---|---|---|---|
| bigram_in | 1,925 | item | **0.3668** | **0.2701** | **0.4722** | **0.5889** | **0.2666** |
| | | begin | 0.3193 | 0.2324 | 0.4242 | 0.5358 | 0.2321 |
| bigram_out | 20,438 | item | 0.0042 | 0.0024 | 0.0092 | 0.0179 | 0.0049 |
| | | begin | **0.0077** | **0.0045** | **0.0144** | **0.0262** | **0.0072** |
| sem_in | 2,653 | item | **0.1415** | **0.1065** | **0.1784** | 0.2165 | **0.1057** |
| | | begin | 0.1240 | 0.0902 | 0.1681 | **0.2170** | 0.0922 |
| sem_out | 19,710 | item | 0.0211 | 0.0145 | 0.0317 | 0.0469 | 0.0169 |
| | | begin | **0.0225** | **0.0152** | **0.0337** | **0.0503** | **0.0178** |

**Steam** (bigram slice = 59.5% of test — the dataset is compositionally transition-heavy):

| Slice | n | Arm | HR@5 | NDCG@5 | HR@10 | HR@20 | MRR |
|---|---|---|---|---|---|---|---|
| bigram_in | 199,056 | item | 0.2754 | 0.2427 | 0.3246 | 0.3915 | 0.2497 |
| | | begin | **0.2762** | **0.2428** | **0.3263** | **0.3947** | **0.2498** |
| bigram_out | 135,269 | item | 0.0083 | 0.0048 | 0.0170 | 0.0332 | 0.0096 |
| | | begin | **0.0103** | **0.0060** | **0.0202** | **0.0385** | **0.0110** |
| sem_in | 20,928 | item | 0.2233 | 0.1439 | 0.3102 | 0.4087 | 0.1433 |
| | | begin | **0.2342** | **0.1505** | **0.3254** | **0.4277** | **0.1494** |

Findings, in decreasing order of importance:

1. **The prediction of §13.7-1 is confirmed on Beauty, in a sharper form than predicted.** The item arm's entire advantage lives inside the 8.6% bigram slice (+15% HR@5 there); on the 91.4% complement the begin arm is not slightly but *massively* better — **+84% HR@5, +56% HR@10, +47% MRR** (0.0077 vs 0.0042, etc.). A composition check closes exactly: 0.086·(0.319−0.367) + 0.914·(0.0077−0.0042) ≈ −0.0009, the observed ALL-slice HR@5 delta. The Beauty "precision regression" of §13.6-2 is therefore **entirely metric composition**: leave-one-out on sparse Amazon data is, for ~9% of its samples, a bigram-completion benchmark that rewards the free transition prior; on everything else the clean aggregator nearly doubles the baseline.
2. **The slice split also exposes how extreme the Markov concentration is.** For both arms, bigram_in vs bigram_out performance differs by ~50–90× on Beauty. These models earn almost all their leave-one-out score on transition-explained targets — a caution against reading ALL-slice metrics as "interest understanding" on sparse benchmarks.
3. **On Steam the begin arm wins both slices, including the transition slice itself.** With 59.5% of test targets bigram-explained and abundant data, the model evidently learns transitions fine *through attention* — the architectural identity shortcut adds bias, not signal (consistent with §1.5-iii: at high data density the aggregation path subsumes the local component). The complement gain (+23% HR@5) mirrors Beauty's pattern at smaller amplitude.
4. **§1.5's mixture language is quantitatively apt**: Beauty = low-coverage transition slice with high per-slice payoff for the hard-wired prior (data too sparse to relearn transitions through attention); Steam = high-coverage transition slice whose transitions the network learns itself. The (c′) prediction refines accordingly — α on Beauty should be significantly positive (it must substitute for unlearnable-from-data transitions), α on Steam ≈ 0 (nothing to add).

### 13.9 Gated transition-prior re-injection (c′) — executed, verdict nuanced

Implemented exactly as §1.5 specifies: readout $= h_b + \alpha\, e_{\text{in}}(x_t)$, scalar $\alpha$ initialized at 0, **excluded from weight decay** (so its fitted value reflects data, not the regularizer), begin arm only. 3 seeds × both datasets, plus one symmetry-breaking rerun. Raw per-seed results in `experiment_results.md` (rows tagged `begin_reinjscalar`).

**Fitted α and effective transition term (α × in/out table alignment):**

| Dataset | seed 42 | seed 123 | seed 7 | α·align per seed |
|---|---|---|---|---|
| Beauty | −0.58 | +1.53 | +1.80 | −0.002 / +0.005 / +0.007 |
| Steam | −3.62 | +4.88 | +5.48 | **+0.18 / +0.42 / +0.50** |

**Headline results:**

1. **α is sign-unidentifiable but basin-asymmetric.** The loss is invariant to jointly flipping α and the embedding geometry, and seeds land in both basins. On Steam the basins are equivalent (the α<0 seed co-flipped its table alignment to −0.049; the effective term α·align is positive in *all* seeds). On Beauty the negative basin is strictly worse on every metric. A symmetry-breaking init (α₀ = 0.1) reliably selects the positive basin: the rerun of the negative-basin seed landed at α = +2.32 and improved across the board (val 0.04253 vs 0.04218, HR@10 0.04995 vs 0.04865). **Recommendation: always initialize α at a small positive value.**
2. **Beauty (α>0 seeds): ALL-slice top-rank parity recovered, val best-in-class.** HR@5 0.03533 vs item 0.03542 (parity; plain begin 0.03455), HR@10 0.05087 (best of all arms, +3.7% vs item), HR@20 0.07085 (+5.7% vs item), val NDCG@10 up to 0.04319 (campaign best; item 0.04016). Residual gap to item remains in NDCG@5/MRR (−4%/−3%).
3. **Steam: metric-neutral but 2.4× faster convergence.** All slices at begin-arm parity, but best epoch drops from ~12 to **5** in all three seeds — the direct path is usable from step one, so the model reaches its optimum in less than half the training. Under §5's throughput framing this is a real gain even at quality parity.
4. **The readout stays clean.** Pre-injection leakage of $h_b$ remains 0.03–0.08 across all reinject runs — the transition term is fully quarantined in the explicit α path, as designed.

**Mechanism surprise (stratified slices on the reinject arm).** The α term does **not** restore the item arm's bigram-slice performance — it wins through the complement instead:

| Beauty slice | item | begin | reinj (α>0) |
|---|---|---|---|
| bigram_in HR@5 | **0.3668** | 0.3193 | 0.3182 (unchanged) |
| bigram_out HR@5 | 0.0042 | 0.0077 | **0.0087** (+13% vs begin, +107% vs item) |
| ALL HR@5 | 0.0354 | 0.0346 | 0.0353 (parity, different route) |

The composition identity closes on both sides: item reaches 0.0354 as a *transition-slice specialist* (0.086·0.367 + 0.914·0.004), reinject reaches 0.0353 as a *complement generalist* (0.086·0.318 + 0.914·0.0087). Interpretation: the item arm's bigram dominance is **memorized first-order transitions** in $W = E_{\text{in}}E_{\text{out}}^\top$, shaped by joint gradients from layer 0 throughout training. A readout-side additive term reusing the *input* table cannot re-create that memorization (the input role dominates $e_{\text{in}}$'s gradients); what α actually buys is a **soft last-item similarity prior** that helps near-miss targets in the long tail. On Steam, where attention already learns transitions, the α path is pure re-routing (large α, zero metric change, faster convergence).

**Doc revisions implied.**

- §1.5 / §11.3-c′'s prediction "α ≈ 0 on aggregation-weighted data" used the wrong observable: α measures how much transition computation the model *routes* through the direct path, not how much the data *needs* it — routing is free, so α grows wherever the path is useful for optimization, including Steam. The falsifiable content survives at the behavioral level (slice metrics), which is where it was confirmed. Parameter-value predictions on non-identifiable gates should be avoided in future design docs.
- The mixture-weight remedy is validated at ALL-slice level but the *transition-memorization* component specifically requires the prior inside the joint training path, not appended at readout. The natural next ablation (beyond §11.3's list) is a **dedicated transition table**: score $+\ \alpha\langle e_{\text{trans}}(x_t), e_{\text{out}}(y)\rangle$ with a third, untied table $E_{\text{trans}}$ — an explicit FPMC factor whose gradients are not shared with the input role. Predicted to recover bigram_in toward 0.367 while keeping the clean readout and the complement gains; parameter cost one extra table, compute cost nil.
- The Steam convergence speedup (best epoch 12 → 5 at parity) suggests reporting **compute-to-quality** alongside final quality in all subsequent ablations (§5's evaluation corollary, now with direct evidence).

### 13.10 Direct-term logit decomposition — quantifying "how much of the model is a bigram machine"

New diagnostic (extends §9.2), run on all six item/begin checkpoints per dataset (3 seeds averaged). Decompose the anchored readout's scoring into the zero-layer direct path and the rest: retrieve once with the full readout, once with the pure direct term $e_{\text{in}}(x_t) E_{\text{out}}^\top$ (§1.5-ii's free transition matrix), and once with the readout minus its $e_{\text{in}}(x_t)$ projection.

| Dataset / arm | markov-only HR@10 (share of full) | resid-only (share) | proj-frac of ‖h‖ | hit-agree@10 | overlap@10 |
|---|---|---|---|---|---|
| Beauty, item | 0.02731 (**55.7%**) | 0.04305 (87.7%) | 30.6% | 39.1% | 9.9% |
| Beauty, begin (control) | 0.00100 (2.0%) | 0.04946 (99.6%) | 5.5% | 0.1% | 0.2% |
| Steam, item | 0.13674 (**68.3%**) | 0.12546 (62.7%) | **59.5%** | **65.6%** | 12.8% |
| Steam, begin (control) | 0.00097 (0.5%) | 0.20221 (99.9%) | 6.8% | 0.2% | 0.2% |

Readings:

1. **The anchored model is substantially a bigram machine.** A zero-layer lookup through the model's own tables — no attention, no depth — reproduces 68% of the item arm's HR@10 on Steam and 66% of its actual hits; on Beauty, 56% / 39%. On Steam the direct direction carries **59.5% of the readout vector's norm**.
2. **Redundancy structure differs by regime.** On Beauty, resid-only keeps 87.7% (attention re-encodes much of the transition signal — the direct path is partly redundant); on Steam, resid-only (62.7%) < markov-only (68.3%) — the direct path is the *primary* carrier and attention does not duplicate it.
3. **The begin-arm control is decisive**: with no identity path, the same measurement collapses to 0.5–2% — the direct-term dominance in the item arm is architectural, not a property of the data or tables per se.
4. **Low overlap@10 (10–13%) with high hit-agree (39–66%)** is the expected signature: the two rankers disagree across their (mostly wrong) tails but agree precisely on the hits, i.e. the direct term is where the item arm's correct answers come from.

Practical use: this diagnostic answers "is long-context extension pointless for this model?" *before* paying for a long-context training run — a model whose hits are 60%+ direct-term-explained cannot benefit from more history in its dominant scoring component, since $e_{\text{in}}(x_t)$ uses exactly one event. Recommended as the first white-box check (alongside the §13.8 slice composition and last-item-only truncation) when diagnosing context-length insensitivity in production anchored-readout models.

### 13.11 Request-side conditioning of `<begin>` (§10) — the largest single lever measured in this campaign

Per the production constraints under discussion (no user-id embedding — cold-start; no sequence-derived aggregates — ceiling lock-in), the only injectable signal in the public datasets is the **fork's request time** (§3.2: the label event's timestamp, known trivially at serving). Implementation: 6 features (weekly + annual sin/cos, train-range-normalized linear time, validity flag) through a learned projection added onto `<begin>` input embeddings only; item streams untouched. Two variants: default (LeCun) vs **zero-initialized** projection (training starts exactly at the unconditioned baseline — the same symmetry-breaking logic that fixed α's sign basin in §13.9).

**Steam, 3 seeds:** HR@10 **0.26161**, NDCG@10 **0.19240** — **+29.2% / +21.9% over the begin baseline**, and +26% over every prior model in this repository (best TIGER stack 0.2077, HSTU 0.2074). Mechanism verified by a shuffle test on the trained model: true time 0.263, time shuffled across samples 0.162 (collapses *below* the no-time baseline 0.202 — wrong dates actively mislead, so the model genuinely conditions on the clock), zero features 0.00005 (OOD input destroys the readout — production deployments need feature-missing robustness: feature dropout in training or fallback). Steam consumption is release/sale-driven and the dataset spans years; a date-conditioned popularity prior is worth a third of the baseline's entire performance.

**Beauty, 3 seeds — initialization decides everything:**

| Variant | val NDCG@10 | HR@5 | HR@10 | NDCG@10 | HR@20 | MRR |
|---|---|---|---|---|---|---|
| begin baseline | 0.04225 | 0.03455 | 0.04964 | 0.02898 | 0.07010 | 0.02659 |
| item baseline | 0.04016 | 0.03542 | 0.04907 | 0.02980 | 0.06705 | 0.02743 |
| +time, default init | 0.03812 (unstable, epochs 51–141) | 0.03345 | 0.05102 | 0.02767 | 0.07276 | 0.02457 |
| **+time, zero init** | **0.04570** | **0.03835** | **0.05605** | **0.03198** | **0.07980** | **0.02886** |

With random projection init the weak day-granularity signal mostly injects optimization noise (val below baseline in 2/3 seeds, erratic best epochs). Zero init pins the starting point to the baseline and lets the data grow the pathway — and the result is a **uniform new Beauty best across all seven metrics and all arms**, including the top-rank precision that neither the clean begin arm nor α-re-injection could recover (HR@5 0.03835 vs item 0.03542, +8.3%; NDCG@5 0.02629 vs 0.02542). The §13.6-2 precision regression is closed *without* any transition-prior machinery — request-time context alone did it, stably (epochs 21–41, tight seed spread).

**Verdict:** the §10 request-side conditioning channel is validated end-to-end and is the largest single lever measured in this campaign (+29% dense, +13% sparse over the begin baseline). Its value is independent of the leakage magnitude — it applies equally to models whose anchoring diagnostics come back clean. Engineering rules discovered: (1) zero-init every additive conditioning pathway (third instance of the pattern: α basins, feature projections); (2) train with feature-missing robustness before deploying; (3) request-time features at train time must be per-fork event-time values (§3.7), never training-time constants.

### 13.12 Late-fusion ablation — where in the network does request time act?

§13.11 established *that* request-time conditioning is the largest lever; this ablation asks *where* the signal earns its keep. Two placements of the same 6 features:

- **Input mode** (§13.11's config): features projected onto the `<begin>` *input* embedding — time is visible to every attention layer, so it can change **what the readout aggregates** (e.g. attend to seasonal history when the request is seasonal).
- **Late mode** (`--time_mode late`): the transformer runs unconditioned; a residual MLP head fuses time into the finished readout vector, $q = h + \text{MLP}([h; t])$ with zero-init output layer (the §13.11 rule applied to the head). Time can only **transform a fixed summary and add a time prior** — it cannot change the aggregation.

**Steam, 2 seeds:** late fusion reaches HR@10 **0.24162** / NDCG@10 **0.17948** (seeds 0.24072/0.24252 — tight), vs the begin baseline 0.20246 and input mode 0.26161. In gain terms: late fusion recovers **66% of the input-mode improvement**; the remaining **34% strictly requires time inside the transformer**.

Readings:

1. **The majority of the request-time benefit is a separable serving-time transform.** Two-thirds of the gain needs no architecture change at all — a small residual MLP over an already-trained-style readout captures the date-conditioned popularity prior. This matters for production retrofits: an existing frozen-backbone deployment can bank most of the win with a head-only change (though this run trains backbone+head jointly; a frozen-backbone variant is the natural follow-up before claiming that literally).
2. **A third of the gain is aggregation conditioning.** The input-mode surplus (0.2416 → 0.2616) is the part where time changes *which history* the readout summarizes, not just how the summary is scored. This is the component that motivates §10's design (conditioning the readout token itself) over a post-hoc feature cross.
3. **Diagnostics stay clean in both modes** (readout leak 0.04–0.06 late vs −0.04 input; both far from the item arm's 0.3–0.6 self-leak), so the comparison is not confounded by the leakage mechanism.

Not yet run: the 3-cell factorization isolating the residual connection from the zero-init (`--late_w2_std`, `--late_no_residual`), and the Beauty late arm. Predicted ordering from the §13.11 init result: residual+zero-init ≥ residual+std-init > no-residual, with the gap largest on sparse data.
