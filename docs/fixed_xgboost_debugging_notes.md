
# Improving Fixed-Point Sigmoid Accuracy in XGBoost Experiments

This document explains the process we followed to debug and fix accuracy issues in our fixed-point XGBoost implementation.

---

## Initial Setup for Debugging

To diagnose what was going wrong:

- Trained both the **floating-point version** and the **fixed-point version** of XGBoost on the `breast_cancer` dataset.
- Printed the **logits** and **sigmoid outputs** for the first 10 test samples.

**Why:** Comparing raw logits and outputs side-by-side made it easy to spot numerical issues early and measure progress as fixes were applied.

---

## What Went Wrong First

The initial outputs were clearly off:

| Logit  | Simple Sigmoid | Fixed Sigmoid |
|--------|----------------|---------------|
|  3.971 | 0.9815         | 0.0000        |
|  4.911 | 0.9927         | 0.0000        |
|  5.160 | 0.9943         | 0.0000        |
| -0.490 | 0.3798         | 0.7832        |
| -1.331 | 0.2089         | 0.7741        |

In many cases, outputs were not just slightly different—they were **completely flipped**.

---

## Diagnosis: Taylor Expansion Wasn’t Enough

Our fixed-point sigmoid approximation initially used:

```python
terms = 4
getcontext().prec = 10
```

This was far too crude. Taylor expansions of `exp(x)`:
- Only converge well near **x ≈ 0**
- Perform poorly on **negative values**, which are exactly what `exp(-x)` needs

---

## First Fix: More Terms, More Precision

We improved the approximation by:
- Increasing to **10 Taylor terms**
- Setting `getcontext().prec = 20` for Decimal calculations

**Result:**

| Logit  | Simple | Fixed | Difference |
|--------|--------|--------|------------|
| -4.859 | 0.0077 | 0.0082 | very close |
| -4.292 | 0.0135 | 0.0139 | very close |
| -4.842 | 0.0078 | 0.0097 | reasonable |

Negative logits were back on track. But positive values were still unstable.

---

## Second Fix: Numerically Stable Sigmoid

### Before:
```python
sigmoid(x) = 1 / (1 + exp(-x))
```

This formula works in theory but fails for large `x`, where `exp(-x)` gets too small and loses precision.

### After:
```python
if x >= 0:
    return 1 / (1 + exp(-x))
else:
    return exp(x) / (1 + exp(x))
```

This avoids overflow/underflow by keeping the exponent small.

### New Issue:
Now, **negative logits** were overestimated (e.g., producing 0.75 instead of 0.01). The problem was still the poor approximation of `exp(x)` for large negative values.

---

## Final Fix: Clamping Inputs and Outputs

To fully stabilize the sigmoid:

### Input Clamping:
```python
x = min(5, max(-5, x))  # force x ∈ [−5, 5]
```

### Output Clamping:
```python
result = max(0.0001, min(0.9999, sigmoid_value))
```

### Additional tuning:
- Increased Taylor terms to **15**
- Set Decimal precision to **28**

---

## About Clamping

### What Is Input Clamping?

Clamping means bounding a value within a predefined range. For logits:

```python
x_clamped = min(5, max(-5, x))
```

### Why Clamp?

The sigmoid function:

`sigma(x) = 1 / (1 + exp(-x))`

Behaves like:
- `sigma(x) ≈ 0` when `x << -10`
- `sigma(x) ≈ 1` when `x >> 10`

So clamping to `[-5, 5]` avoids wasting effort approximating values that would yield saturated 0/1 outputs anyway.

### When Clamping Is a Good Idea

- Prevents instability in `sigmoid`, `softmax`, `log`
- Helps fixed-point math behave reasonably
- Used in TensorFlow, PyTorch, and XGBoost internally

### When Not to Clamp

- Scientific simulations requiring full precision
- Training time gradient manipulation

For **inference**, clamping is a well-accepted and practical solution.

---

## Why This Combo Works

| Change                             | Why We Did It                                  |
|-----------------------------------|------------------------------------------------|
| Increased Taylor terms to 10–15   | Better approximation for `exp(x)`              |
| Raised decimal precision to 28    | Prevented rounding errors in `Decimal`         |
| Used stable sigmoid formulation   | Reduced overflow/underflow risk                |
| Clamped inputs to [-5, 5]         | Kept logits in safe numerical range            |
| Clamped outputs to [0.0001, 0.9999]| Ensured safe output for loss functions         |

Together, these changes brought the fixed-point sigmoid outputs much closer to the floating-point ones, enabling reliable training and evaluation.
