
# Fixed‑Point Gradient Boosting (ZK‑friendly) — Design & Walkthrough

This document explains the end‑to‑end design of the fixed‑point gradient boosting implementation we built to be **integer‑only** (ZK‑friendly).

---

## 1) Goals & Constraints

**Primary goals**
- **All training and inference in fixed‑point integers** (no float math anywhere in the learning loop) so it can be translated to zero‑knowledge circuits.
- Preserve the core **XGBoost histogram tree algorithm** semantics (gain formula, regularization) while replacing floats with scaled integers.

**Key constraints**
- Deterministic operations (no hidden float nondeterminism).
- Bounded ranges to prevent overflow and sigmoid/logit saturation.
- Keep it **fast in Python** without sacrificing ZK‑friendliness.

---

## 2) Fixed‑Point Representation

We choose a global scale `SCALE = 100_000` (5 decimal digits). Any real value `x` is represented as `x_fp = round(x * SCALE)`.

```python
SCALE = 100_000
def float_to_fixed(x: float) -> int: return int(round(x * SCALE))
def fixed_to_float(x: int) -> float: return x / SCALE     # for display only
def fixed_div_scalar(num: int, den: int) -> int:          # num/den in fixed
    return (num * SCALE) // den if den != 0 else 0
```

**Why this works**
- Integer addition/subtraction is exact.
- We rescale after multiplication/division to keep values on the same fixed scale.
- Using `int64` gives plenty of headroom before overflow at this scale.

**Trade‑offs**
- Higher SCALE = better precision but slower arithmetic and higher overflow risk.
- Lower SCALE = faster but coarser. `1e5` is a practical middle ground for tabular tasks.

---

## 3) Sigmoid & Logit (No Floats)

### 3.1 Sigmoid approximation (inference & gradient)
We use a **wider piecewise‑linear** sigmoid clamped to `[0, 1]`:

```python
def sigmoid_vec(x_fp: np.ndarray) -> np.ndarray:
    two = 2 * SCALE
    return np.where(x_fp <= -two, 0,
           np.where(x_fp >= two, SCALE, (x_fp + two) // 4)).astype(np.int64)
```

**Why this choice**  
- It is **integer‑only**, monotone, cheap, and avoids exp() in circuits.
- The widened linear region reduces saturation, keeping gradients useful.

### 3.2 Logit initializer (bias term)
We avoid `log` and use the identity `logit(p) = 2 * atanh(2p‑1)` with the odd‑power series:

\[
\operatorname{atanh}(z) \approx z + \frac{z^3}{3} + \frac{z^5}{5}\quad (|z|<1)
\]

Correctly scaled **pure integer** implementation:

```python
def _pow3(z: int) -> int:  # z^3 / SCALE^2
    return (z * z * z) // (SCALE * SCALE)

def _pow5(z: int) -> int:  # z^5 / SCALE^4
    z2 = z * z
    return (z2 * z2 * z) // (SCALE * SCALE * SCALE * SCALE)

def fixed_logit_from_p(p_fp: int) -> int:
    p_fp = max(min(p_fp, int(0.999 * SCALE)), int(0.001 * SCALE))  # clamp
    z = 2 * p_fp - SCALE
    atanh = z + (_pow3(z) // 3) + (_pow5(z) // 5)
    return 2 * atanh
```

**Why the series?**  
- It’s smooth near the center and easy to constrain in ZK.  
- **Important bug we fixed:** earlier we accidentally introduced an extra implicit `*SCALE` inside power helpers which **over‑scaled** the initial logit, saturating the first sigmoid and killing gradients. The corrected `_pow3/_pow5` keep everything on the same fixed scale.

---

## 4) Histogram Tree Training (Integer‑only)

We implement the standard XGBoost gain with L2 regularization in fixed‑point integers.

### 4.1 Pre‑binning: compute once per feature
For each feature `j`, compute **global uniform** bin edges (min–max) and **digitize** all samples **once**:

```python
edges_j = fixed_linspace_fp(min_j, max_j, num_bins + 1)     # int edges
ids_j   = np.searchsorted(edges_j[:-1], x_col_fp, 'right')  # int bin IDs
```

**Why pre‑bin?**
- Eliminates expensive per‑node digitize.
- Allows reusing `ids_j[node_idx]` to build histograms fast with `np.bincount`.

**Accuracy note**  
Global uniform bins are coarse deeper in the tree. We mitigated with **`num_bins = 128`**. A future upgrade is **quantile bins** (still integer‑only) to match distributions better.

### 4.2 Build histograms (vectorized, int‑only)
Given sample indices for the node (`idx`) and precomputed `ids_j`:
```python
G = np.bincount(ids_j[idx], weights=grad_fp[idx],  minlength=B).astype(np.int64)
H = np.bincount(ids_j[idx], weights=hess_fp[idx], minlength=B).astype(np.int64)
```

**ZK semantics**  
In proofs, you don’t prove “we ran NumPy”. You prove the **histogram definition** holds:
\[
G[b] = \sum_{i \in \text{node}} [\text{bin}(x_i)=b]\cdot g_i
\]
Either by **one‑hot constraints** (simple) or **sorting+prefix sums** (scalable).

### 4.3 Evaluate all splits at once (prefix sums)
```python
GL = np.cumsum(G[:-1]);    HL = np.cumsum(H[:-1])
Gtot, Htot = G.sum(), H.sum()
GR, HR = Gtot - GL, Htot - HL

left  = fixed_div_vec(GL*GL, HL + lambda_fp)
right = fixed_div_vec(GR*GR, HR + lambda_fp)
parent = fixed_div_scalar(Gtot*Gtot, Htot + lambda_fp)

gain = ((left + right - parent) // 2)
gain[HL==0 | HR==0] = very_negative
best_bin = argmax(gain)
```

**Why this way**
- Replaces a Python loop over bins with vectorized NumPy ops.
- Preserves the exact fixed‑point gain formula (no approximations).

### 4.4 Splitting & recursion
We pass **index arrays** (`idx`) down the tree (no Python list slicing). The **stored split** is a **fixed‑point threshold** `split_fp = edges_j[best_bin]`. Prediction compares integers: `x_fp[feat] <= split_fp`.

**Why store thresholds, not bins?**
- Keeps prediction **stateless** (no edges needed at inference).  
- It’s consistent with how you’d represent splits inside a ZK circuit.

---

## 5) Booster Loop (Vectorized)

For each boosting iteration:
1. `p_pred_fp = sigmoid_vec(y_pred_fp)`  
2. `grad_fp = p_pred_fp - y_fp`  
3. `hess_fp = p_pred_fp * (SCALE - p_pred_fp) // SCALE`  
4. Fit one tree on `(grad_fp, hess_fp)`  
5. Update logits **vectorized**:  
   `y_pred_fp -= (learning_rate_fp * tree_out_fp) // SCALE`

**Why vectorize?**  
- Avoids Python loops over samples — the main speed killer in pure Python.

---

## 6) Numerical Stability & Safety

- **Clamping** probabilities in logit init to `[0.001, 0.999]` prevents `atanh` blow‑ups.
- **Leaf value** is clipped to `[-1, 1]` to keep updates bounded.
- **Division by zero** in gains is masked (`HL==0 or HR==0 →` disallow split).

**Regularization**
- We use standard L2 (`lambda`) in the denominators of gain and leaf computations, identical in spirit to XGBoost but in fixed‑point integers.

---

## 7) Reproducibility & Determinism

- Fixed seeds for feature sampling per tree (NumPy `default_rng`) keep runs deterministic.
- No floating‑point sources; all core arrays are `int64`.

---

## 8) Performance Optimizations (and *why*)

1. **Pre‑bin once per feature**  
   *Why:* avoids O(n log B) digitize per node; reuse `ids_j` for every node.

2. **Pass index arrays down the tree**  
   *Why:* cheap subsetting (`ids_j[idx]`) without Python loops / list comprehensions.

3. **`np.bincount` histograms**  
   *Why:* compiled, fast integer aggregation with weights; maps cleanly to ZK semantics.

4. **Vectorized split gains with prefix sums**  
   *Why:* remove inner Python loop over bins; gains computed in bulk.

5. **Integer‑only sigmoid & logit**  
   *Why:* ZK‑friendly and keeps gradients in a reasonable range (wider sigmoid).

6. **128 bins**  
   *Why:* recover split fidelity lost by global (uniform) pre‑bins without big runtime cost.

**What we deliberately didn’t do (yet)**
- **Quantile bins** (better accuracy): still integer‑only; a good next step.
- **Parallelization / C++/Rust**: would shrink the speed gap vs XGBoost substantially.

---

## 9) Datasets & Benchmarking

We include:
- **Breast Cancer** (small sanity check)  
- **Credit Default** (~30k rows)  
- **Covertype (binary y==1)** (~581k rows, 54 features)

**Why no `StandardScaler`**
- It reintroduces floating‑point preprocessing and slows things down.  
- The model converts features to fixed‑point integers internally and learns thresholds in that space.

**XGBoost baseline**
- Kept for reference; it’s C++/multithreaded, so expect it to be much faster and slightly more accurate (better quantile sketches, richer implementation).

---

## 11) Common Pitfalls (and how we avoided them)

- **Sigmoid saturation in the first round** → fixed by proper logit scaling (`_pow3/_pow5`).  
- **Per‑node digitize** → replaced by pre‑binning to cut runtime drastically.  
- **Float leaks** (scalers, `math.log`, float linspace) → removed.  
- **Division by zero in gains** → masked invalid splits.  
- **Inconsistent dtypes** → core arrays are `np.int64` end‑to‑end.

---

## 12) Quick Start

```python
# Train
model = GenericFixedPointXGB(n_estimators=100, max_depth=5, num_bins=128, lambda_=1.0, gamma=0.0)
model.fit(X_train, y_train)

# Predict
proba = model.predict_proba(X_test)   # floats in [0,1] for convenience
pred  = model.predict(X_test)         # 0/1 ints
```

**Tip:** If accuracy lags on large datasets, try:
- `num_bins=256`, or
- switch to **quantile bins** per feature (integer‑only; precomputed once).
