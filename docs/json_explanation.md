# Model Serialization for GenericFixedPointXGB

This document explains exactly what is being saved in the JSON model files when serializing a trained `GenericFixedPointXGB` model.

---

##  Content of the Saved JSON File

Each saved model file contains:

###  Model Hyperparameters

These describe the model configuration used during training:

* **initial\_logit** — the initial bias logit, derived from class proportions.
* **n\_estimators** — total number of trees trained.
* **max\_depth** — maximum depth allowed for each tree.
* **learning\_rate** — learning rate used in boosting.
* **lambda\_** — L2 regularization parameter.
* **gamma** — minimum gain threshold required to perform a split.

These fields allow full reproducibility of the training setup.

---

###  Trained Trees Structure

The key component saved is the list of trained trees.

#### Tree Representation

* Stored under the key: `trees`
* It is a list, where each entry represents one trained tree.
* Each tree is stored recursively as nested structures.

#### Node Encoding

Each node in a tree is either:

* **Leaf node:**

  * A single numeric value (the prediction output for that leaf).

* **Internal node:**

  * Encoded as a list with the special marker `"__tuple__"` to distinguish tuples from JSON lists:

```json
["__tuple__", feature_index, split_value, left_subtree, right_subtree]
```

Where:

* **feature\_index** — which feature the node splits on.
* **split\_value** — the threshold for that feature.
* **left\_subtree** — recursively the left branch.
* **right\_subtree** — recursively the right branch.

#### Type Conversion

* All internal numeric types (e.g. numpy `int64`/`float64`) are automatically converted into standard Python `int` and `float` before serialization to ensure JSON compatibility.

---

###  Training Metadata (NEW: For ZK-Friendly Auditing)

For each tree, we now store additional metadata describing how training proceeded at every node.

#### For every node (split or leaf):

* **features\_considered** — The list of feature indices considered for splitting at that node (due to feature subsampling).

#### For every feature considered:

* **histograms:**

  * **bin\_edges** — The binning thresholds used for that feature at this node.
  * **grad\_hist** — The sum of gradients per bin.
  * **hess\_hist** — The sum of Hessians per bin.

#### For every leaf:

* **leaf\_value** — The output value assigned to that leaf.

This full metadata allows auditing of:

* The exact splits chosen.
* The histogram computations.
* The gain computations that led to each split decision.

---

##  Why we store this

> This additional metadata allows us to move beyond just reproducible inference.
>
> It allows fully auditable training steps suitable for Zero-Knowledge (ZK) proofs, including:
>
> * Verifying histogram aggregation correctness.
> * Verifying gain computations.
> * Verifying split decisions.
> * Fully reconstructing the training path without needing the original dataset.

In ZK-friendly settings, this design allows us to prove that a model was trained honestly without revealing the full dataset inside the proof.

---

##  Model Loading

* Upon loading, the `list_to_tuple()` function reverses the encoding:

  * `"__tuple__"` lists are converted back into Python tuples.
  * Numeric types are directly read back as Python native types.
  * All metadata remains fully available for downstream verification.

