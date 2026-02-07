
# XGBoost from Scratch

This guide walks through the full development journey of a custom XGBoost implementation, starting from a basic binary classifier and evolving into a fast, accurate, fixed-point and multiclass-capable system. Diagrams, equations, and annotated code are included to make the ideas accessible.

---

## Initial Boosting Setup

**Problem**: Poor accuracy and no real optimization.

```python
# Initial prediction update
y_pred -= tree.predict(X)
```

This lacked:
- Logistic loss (gradient/hessian)
- Learning rate
- Probability conversion (sigmoid)

---

## Proper Gradient Boosting (Binary)

### Logistic Loss Gradients

$$
\text{Gradient:}\quad g = p - y$$
$$
\text{Hessian:}\quad h = p (1 - p)
$$

```python
p = sigmoid(y_pred)
grad = p - y
hess = p * (1 - p)
```

### Sigmoid Activation

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

> **Fixes**: Allowed for probability-based updates using logistic loss.

---

## Stability Fixes

### Stable Sigmoid to Avoid Overflow

```python
def stable_sigmoid(x):
    x = np.array(x)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1 / (1 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1 + exp_x)
    return out
```

> Avoided numerical issues when input to `exp` was large/negative.

---

## Efficiency & Speed

- **Clipped tree output**: `np.clip(value, -10, 10)`
- **Subsampling + feature sampling**:
  ```python
  subsample = 0.8
  colsample_bytree = 0.8
  ```
- **StandardScaler** for input normalization

---

## Fixed-Point Version

### Motivation:
Simulate low-resource environments (embedded systems).

```python
from decimal import Decimal, getcontext
getcontext().prec = 10
```

### Taylor Sigmoid

```python
def taylor_sigmoid(x, terms=4):
    x = Decimal(x)
    result = Decimal(1)
    term = Decimal(1)
    sign = -1
    for i in range(1, terms + 1):
        term *= x / Decimal(i)
        result += sign * term
        sign *= -1
    return 1 / (1 + result)
```

> No floating point, just pure math. Slow, but portable.

---

## Multiclass Support

### Softmax Function

```python
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
```

### Class-Wise Tree Ensembles

```python
self.trees = defaultdict(list)  # One tree list per class
```

> Each class gets its own set of boosting rounds!

---

## Evaluation and Metrics

- `accuracy_score` from `sklearn`
- `f1_score(..., average='weighted')`
- Timing with `time.time()`
- Visual benchmark tables (accuracy, F1, time)

---

## Bonus Fixes

- Conditional `LabelEncoder`:
  ```python
  if y.dtype.kind in {'U', 'S', 'O'}:
      y = LabelEncoder().fit_transform(y)
  ```
- Added multiple datasets: breast cancer, moons, circles, digits, iris, wine, letter

---

## Summary

- We started with a fragile prototype and evolved it into a solid, testable, explainable model.
- Each phase introduced enhancements in accuracy, speed, or generality.
- Fixed-point versions allow for simulation of constrained hardware.
- Multiclass and benchmarking support closes the loop on real-world usefulness.
