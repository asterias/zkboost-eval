import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
from tqdm import tqdm
import sys
import json
import os

# ============================ Fixed-point config ============================
SCALE = 100_000
INV_SCALE = 1 / SCALE  # only for user-facing conversions, not used in train/infer math

def float_to_fixed(x: float) -> int:
    # Only used at boundaries / I/O (params, initial conversions)
    return int(round(x * SCALE))

def fixed_to_float(x: int) -> float:
    # Never used in train/infer math
    return x * INV_SCALE

def fixed_div_scalar(num: int, den: int) -> int:
    # (num / den) in fixed point; returns SCALE-scaled int
    return (num * SCALE) // den if den != 0 else 0

def fixed_div_vec(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.int64)
    m = den != 0
    out[m] = (num[m] * SCALE) // den[m]
    return out

def fixed_clip(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

# ---- corrected fixed-point logit initializer (atanh series) ----
def _pow3(z: int) -> int:
    # z^3 / SCALE^2  => keeps SCALE scaling
    return (z * z * z) // (SCALE * SCALE)

def _pow5(z: int) -> int:
    # z^5 / SCALE^4
    z2 = z * z
    z5 = z2 * z2 * z
    return z5 // (SCALE * SCALE * SCALE * SCALE)

def fixed_logit_from_p(p_fp: int) -> int:
    # p_fp is SCALE-scaled probability
    p_fp = fixed_clip(p_fp, float_to_fixed(0.001), float_to_fixed(0.999))
    z = 2 * p_fp - SCALE   # SCALE-scaled
    atanh = z + (_pow3(z) // 3) + (_pow5(z) // 5)  # SCALE-scaled
    return 2 * atanh  # SCALE-scaled logit

# ---- vectorized fixed-point sigmoid (wider piecewise) ----
def sigmoid_vec(x_fp: np.ndarray) -> np.ndarray:
    two = 2 * SCALE
    # piecewise-linear: 0 for x<=-2, 1 for x>=2, linear in-between
    y = np.where(x_fp <= -two, 0,
        np.where(x_fp >= two, SCALE, (x_fp + two) // 4))
    return y.astype(np.int64)

# ========================= Pre-binning (int-only) ===========================
def fixed_linspace_fp(start_fp: int, stop_fp: int, num: int) -> np.ndarray:
    if num <= 1:
        return np.array([start_fp], dtype=np.int64)
    step_fp = (stop_fp - start_fp) // (num - 1)
    out = np.empty(num, dtype=np.int64)
    cur = start_fp
    for i in range(num):
        out[i] = cur
        cur += step_fp
    out[-1] = stop_fp  # exact end
    return out

def digitize_with_edges_int(x_col_fp: np.ndarray, edges_fp: np.ndarray) -> np.ndarray:
    # all int: like np.digitize(x, edges[:-1], right=False)
    return np.searchsorted(edges_fp[:-1], x_col_fp, side='right').astype(np.int64)

def bincount_sum_int(ids: np.ndarray, w: np.ndarray, bins: int) -> np.ndarray:
    # Pure-integer histogram accumulation: out[k] = sum(w[i] for i with ids[i]==k)
    out = np.zeros(bins, dtype=np.int64)
    np.add.at(out, ids, w)
    return out

# =================== Fast fixed-point tree (vectorized) =====================
class XGBoostTreeClassifierFPFast:
    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0, num_bins=128,
                 colsample_bytree=1.0, seed=0, no_repeat_features=True,
                 force_full_depth=False):
        self.max_depth = max_depth
        self.lambda_fp = float_to_fixed(lambda_)
        self.gamma_fp  = float_to_fixed(gamma)
        self.num_bins  = num_bins
        self.colsample_bytree = colsample_bytree
        self.rng = np.random.default_rng(seed)
        self.no_repeat_features = no_repeat_features
        self.force_full_depth = force_full_depth

        self.tree = None
        self.bin_edges_fps = None
        self.bin_ids_per_feat = None

    def fit(self, X_fp: np.ndarray, grad_fp: np.ndarray, hess_fp: np.ndarray):
        n, _ = X_fp.shape
        self._prebin(X_fp)
        idx_all = np.arange(n, dtype=np.int64)
        self.tree = self._fit_node(idx_all, grad_fp, hess_fp, depth=0, used_mask=0)
        return self.tree

    def predict_fp(self, X_fp: np.ndarray) -> np.ndarray:
        out = np.empty(X_fp.shape[0], dtype=np.int64)
        for i in range(X_fp.shape[0]):
            out[i] = self._predict_row_fp(X_fp[i], self.tree)
        return out

    # ---------- internals ----------
    def _prebin(self, X_fp: np.ndarray):
        n, d = X_fp.shape
        self.bin_edges_fps = []
        self.bin_ids_per_feat = []
        for j in range(d):
            col = X_fp[:, j]
            cmin, cmax = int(col.min()), int(col.max())
            if cmin == cmax:
                edges = np.array([cmin] * (self.num_bins + 1), dtype=np.int64)
                ids = np.zeros(n, dtype=np.int64)
            else:
                edges = fixed_linspace_fp(cmin, cmax, self.num_bins + 1)
                ids   = digitize_with_edges_int(col, edges)
            self.bin_edges_fps.append(edges)
            self.bin_ids_per_feat.append(ids)

    def _leaf_value(self, idx: np.ndarray, grad_fp: np.ndarray, hess_fp: np.ndarray) -> int:
        G = int(grad_fp[idx].sum(dtype=np.int64))
        H = int(hess_fp[idx].sum(dtype=np.int64))
        val = fixed_div_scalar(G, H + self.lambda_fp)  # SCALE-scaled
        return fixed_clip(val, float_to_fixed(-1.0), float_to_fixed(1.0))

    def _fit_node(self, idx: np.ndarray, grad_fp: np.ndarray, hess_fp: np.ndarray, depth: int, used_mask: int):
        # Early stop -> leaf (and maybe pad)
        if depth >= self.max_depth or idx.size < 2:
            leaf = self._leaf_value(idx, grad_fp, hess_fp)
            return self._pad_to_depth(leaf, depth, used_mask) if self.force_full_depth else leaf

        d = len(self.bin_edges_fps)

        # Candidate pool honoring "no repeated features in path" for REAL splits
        if self.no_repeat_features:
            avail = [j for j in range(d) if ((used_mask >> j) & 1) == 0]
        else:
            avail = list(range(d))

        if not avail:
            leaf = self._leaf_value(idx, grad_fp, hess_fp)
            return self._pad_to_depth(leaf, depth, used_mask) if self.force_full_depth else leaf

        n_choose = max(1, int(len(avail) * self.colsample_bytree))
        features = self.rng.choice(avail, size=min(n_choose, len(avail)), replace=False)

        best_gain = float_to_fixed(-1_000_000_000)
        best_feat = None
        best_bin  = None

        for j in features:
            ids_j = self.bin_ids_per_feat[j][idx]
            # Skip constants in this node
            if ids_j.max(initial=0) == ids_j.min(initial=0):
                continue

            G = bincount_sum_int(ids_j, grad_fp[idx], self.num_bins + 1)
            H = bincount_sum_int(ids_j, hess_fp[idx], self.num_bins + 1)

            GL = np.cumsum(G[:-1], dtype=np.int64)
            HL = np.cumsum(H[:-1], dtype=np.int64)
            Gtot = int(G.sum(dtype=np.int64))
            Htot = int(H.sum(dtype=np.int64))
            GR = Gtot - GL
            HR = Htot - HL

            ok = (HL != 0) & (HR != 0)
            left   = fixed_div_vec((GL * GL)   // SCALE, HL  + self.lambda_fp)
            right  = fixed_div_vec((GR * GR)   // SCALE, HR  + self.lambda_fp)
            parent = fixed_div_scalar((Gtot * Gtot) // SCALE, Htot + self.lambda_fp)

            gain = ((left + right - parent) // 2).astype(np.int64)  # SCALE
            gain[~ok] = float_to_fixed(-1e9)

            b = int(np.argmax(gain))
            g = int(gain[b])
            if g > best_gain and g > self.gamma_fp:
                best_gain = g
                best_feat = j
                best_bin  = b

        if best_feat is None:
            leaf = self._leaf_value(idx, grad_fp, hess_fp)
            return self._pad_to_depth(leaf, depth, used_mask) if self.force_full_depth else leaf

        # Real split
        ids_best = self.bin_ids_per_feat[best_feat][idx]
        left_mask  = ids_best <= best_bin
        right_mask = ~left_mask
        left_idx  = idx[left_mask]
        right_idx = idx[right_mask]

        split_fp = int(self.bin_edges_fps[best_feat][best_bin])

        # Mark this feature as used for REAL splits
        child_used_mask = used_mask | (1 << int(best_feat))

        left_node  = self._fit_node(left_idx,  grad_fp, hess_fp, depth+1, child_used_mask)
        right_node = self._fit_node(right_idx, grad_fp, hess_fp, depth+1, child_used_mask)
        return (int(best_feat), int(split_fp), left_node, right_node)

    def _choose_dummy_feature(self, used_mask: int) -> int:
        """Prefer an unused feature for padding; fallback to 0 if all used."""
        d = len(self.bin_edges_fps)
        for j in range(d):
            if ((used_mask >> j) & 1) == 0:
                return int(j)
        return 0

    def _pad_to_depth(self, node, depth: int, used_mask: int):
        """
        Wrap `node` in dummy splits until reaching `self.max_depth`.
        Dummy splits DO NOT mark features as used; both children are `node`.
        """
        cur = node
        cur_depth = depth
        while cur_depth < self.max_depth:
            j = self._choose_dummy_feature(used_mask)
            # choose a deterministic threshold (e.g., middle edge)
            edges = self.bin_edges_fps[j]
            split_fp = int(edges[len(edges)//2])
            cur = (int(j), split_fp, cur, cur)
            cur_depth += 1
        return cur

    def _predict_row_fp(self, x_fp_row: np.ndarray, node):
        if not isinstance(node, tuple):
            return node
        feat, split_fp, l, r = node
        if x_fp_row[feat] <= split_fp:
            return self._predict_row_fp(x_fp_row, l)
        else:
            return self._predict_row_fp(x_fp_row, r)

# ================== Booster (vectorized, int-only math) =====================
class GenericFixedPointXGB:
    def __init__(self, n_estimators=50, max_depth=3, learning_rate=0.3,
                 lambda_=1.0, gamma=0.0, num_bins=128, colsample_bytree=1.0, seed=0,
                 no_repeat_features=True, force_full_depth=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate_fp = float_to_fixed(learning_rate)
        self.lambda_ = lambda_
        self.gamma = gamma
        self.num_bins = num_bins
        self.colsample_bytree = colsample_bytree
        self.rng = np.random.default_rng(seed)
        self.no_repeat_features = no_repeat_features
        self.force_full_depth = force_full_depth

        self.trees = []
        self.initial_logit_fp = 0

    def _X_to_fixed(self, X: np.ndarray) -> np.ndarray:
        """
        Convert raw features to fixed-point integers.
        NOTE: This uses rounding on input floats to produce fixed-point integers.
        All subsequent math is strictly integer. If you need float-free ingestion,
        pass integers already multiplied by SCALE.
        """
        return np.rint(X * SCALE).astype(np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Convert labels to fixed-point in {0, SCALE}
        y_fp = np.array([float_to_fixed(float(v)) for v in y], dtype=np.int64)
        X_fp = self._X_to_fixed(X)

        # Prior: fixed-point mean probability, then atanh series to logit
        p_fp = int(y_fp.sum(dtype=np.int64)) // y_fp.size
        self.initial_logit_fp = fixed_logit_from_p(p_fp)

        # Current logits (SCALE-scaled)
        y_pred_fp = np.full(y_fp.shape[0], self.initial_logit_fp, dtype=np.int64)

        for _ in range(self.n_estimators):
            # Fixed-point probabilities
            p_pred_fp = sigmoid_vec(y_pred_fp)  # in [0, SCALE]
            # Grad/Hess in fixed point
            grad_fp = (p_pred_fp - y_fp).astype(np.int64)
            hess_fp = (p_pred_fp * (SCALE - p_pred_fp)) // SCALE  # ∈ [0, SCALE/4]

            # Train one tree (all int math inside)
            tree = XGBoostTreeClassifierFPFast(
                max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma,
                num_bins=self.num_bins, colsample_bytree=self.colsample_bytree,
                seed=int(self.rng.integers(1 << 31)),
                no_repeat_features=self.no_repeat_features,
                force_full_depth=self.force_full_depth
            )
            tree.fit(X_fp, grad_fp, hess_fp)

            # Update logits: y_pred_fp -= eta * update
            update_fp = tree.predict_fp(X_fp)        # SCALE-scaled
            y_pred_fp -= (self.learning_rate_fp * update_fp) // SCALE

            self.trees.append(tree)

    # Fixed-point probability output (no floats at all)
    def predict_proba_fp(self, X: np.ndarray) -> np.ndarray:
        X_fp = self._X_to_fixed(X)
        y_pred_fp = np.full(X_fp.shape[0], self.initial_logit_fp, dtype=np.int64)
        for tree in self.trees:
            update_fp = tree.predict_fp(X_fp)
            y_pred_fp -= (self.learning_rate_fp * update_fp) // SCALE
        proba_fp = sigmoid_vec(y_pred_fp)  # SCALE-scaled ints
        return proba_fp

    # User-facing float probabilities (conversion only)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba_fp = self.predict_proba_fp(X)
        return proba_fp.astype(np.float64) * INV_SCALE

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Classification threshold at 0.5 in fixed-point
        proba_fp = self.predict_proba_fp(X)
        return (proba_fp >= float_to_fixed(0.5)).astype(int)

    def save_model(self, filepath: str):
        data = {
            "initial_logit_fp": int(self.initial_logit_fp),
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate_fp * INV_SCALE),  # for human readability
            "lambda_": float(self.lambda_),
            "gamma": float(self.gamma),
            "num_bins": int(self.num_bins),
            "colsample_bytree": float(self.colsample_bytree),
            "no_repeat_features": bool(self.no_repeat_features),
            "force_full_depth": bool(self.force_full_depth),
            "trees": [tuple_to_list(t.tree) for t in self.trees],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_model(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.initial_logit_fp = int(data["initial_logit_fp"])
        self.n_estimators = int(data["n_estimators"])
        self.max_depth = int(data["max_depth"])
        self.learning_rate_fp = float_to_fixed(float(data["learning_rate"]))
        self.lambda_ = float(data["lambda_"])
        self.gamma = float(data["gamma"])
        self.num_bins = int(data["num_bins"])
        self.colsample_bytree = float(data["colsample_bytree"])
        self.no_repeat_features = bool(data.get("no_repeat_features", True))
        self.force_full_depth = bool(data.get("force_full_depth", False))
        self.trees = []
        for tdata in data["trees"]:
            t = XGBoostTreeClassifierFPFast(
                max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma,
                num_bins=self.num_bins, colsample_bytree=self.colsample_bytree,
                no_repeat_features=self.no_repeat_features,
                force_full_depth=self.force_full_depth
            )
            t.tree = list_to_tuple(tdata)
            self.trees.append(t)

# ---- JSON tuple helpers ----
def tuple_to_list(obj):
    if isinstance(obj, tuple):
        return ["__tuple__"] + [tuple_to_list(item) for item in obj]
    elif isinstance(obj, list):
        return [tuple_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    else:
        return obj

def list_to_tuple(obj):
    if isinstance(obj, list):
        if len(obj) > 0 and obj[0] == "__tuple__":
            return tuple(list_to_tuple(item) for item in obj[1:])
        else:
            return [list_to_tuple(item) for item in obj]
    else:
        return obj


# ======================= Artifacts export helpers ===========================
def _tree_leaf_id_map(node):
    """
    Return mapping from path strings like 'LLR' to dict {'id': int, 'value_fp': int}.
    DFS left-to-right so IDs are stable from 0..(num_leaves-1).
    """
    mapping = {}
    def dfs(n, path, next_id):
        if not isinstance(n, tuple):
            mapping[path] = {"id": next_id, "value_fp": int(n)}
            return next_id + 1
        feat, split_fp, l, r = n
        nid = dfs(l, path + "L", next_id)
        nid = dfs(r, path + "R", nid)
        return nid
    dfs(node, "", 0)
    return mapping

def _tree_internal_nodes(node):
    """Return list of dicts with (node_id pre-order, feature, threshold_fp)."""
    out = []
    def dfs(n, next_id):
        if not isinstance(n, tuple):
            return next_id
        feat, split_fp, l, r = n
        my_id = next_id
        out.append({"node_id": my_id, "feature": int(feat), "threshold_fp": int(split_fp)})
        nid = dfs(l, my_id + 1)
        nid = dfs(r, nid)
        return nid
    dfs(node, 0)
    return out

def _leaf_path_for_row(x_fp_row, node):
    """Follow the tree and return the L/R path string to the reached leaf."""
    path = []
    n = node
    while isinstance(n, tuple):
        feat, split_fp, l, r = n
        if int(x_fp_row[int(feat)]) <= int(split_fp):
            path.append('L')
            n = l
        else:
            path.append('R')
            n = r
    return "".join(path)

def export_model_artifacts(model, X_train, X_test, prefix):
    """
    model: GenericFixedPointXGB
    Writes:
      - {prefix}_leaf_ids_train.csv and _test.csv (rows=data points, cols=trees)
      - {prefix}_weights.csv (tree, leaf_id, value_fp, value_float)
      - {prefix}_model_nodes.csv (tree, node_id, feature, threshold_fp, threshold_float)
    """
    import csv
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    def compute_leaf_ids(X):
        X_fp = model._X_to_fixed(X)
        n, _ = X_fp.shape
        T = len(model.trees)
        arr = np.empty((n, T), dtype=np.int64)
        # Precompute maps per tree
        maps = []
        for t in model.trees:
            maps.append(_tree_leaf_id_map(t.tree))
        for i in range(n):
            row = X_fp[i]
            for t_idx, t in enumerate(model.trees):
                path = _leaf_path_for_row(row, t.tree)
                arr[i, t_idx] = maps[t_idx][path]["id"]
        return arr

    # Leaf IDs
    if X_train is not None:
        arr_tr = compute_leaf_ids(X_train)
        np.savetxt(f"{prefix}_leaf_ids_train.csv", arr_tr, fmt="%d", delimiter=",")
    if X_test is not None:
        arr_te = compute_leaf_ids(X_test)
        np.savetxt(f"{prefix}_leaf_ids_test.csv", arr_te, fmt="%d", delimiter=",")

    # Weights
    with open(f"{prefix}_weights.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree","leaf_id","value_fp","value_float"])
        for t_idx, t in enumerate(model.trees):
            mp = _tree_leaf_id_map(t.tree)
            for path, info in mp.items():
                val_fp = int(info["value_fp"])
                w.writerow([t_idx, info["id"], val_fp, val_fp * INV_SCALE])

    # Internal nodes (feature, threshold)
    with open(f"{prefix}_model_nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree","node_id","feature","threshold_fp","threshold_float"])
        for t_idx, t in enumerate(model.trees):
            nodes = _tree_internal_nodes(t.tree)
            for nd in nodes:
                thr_fp = int(nd["threshold_fp"])
                w.writerow([t_idx, nd["node_id"], nd["feature"], thr_fp, thr_fp * INV_SCALE])
# ============================= Dataset loaders ==============================
def load_credit_dataset(scale_money_by=1000.0):
    data = np.genfromtxt('credit_default.csv', delimiter=',', skip_header=1, filling_values=0)
    X = data[:, 1:-1].astype(np.float64)
    y = (data[:, -1] > 0.5).astype(int)
    if scale_money_by:
        money_idx = np.array([0, *range(11, 17), *range(17, 23)], dtype=np.int64)
        X[:, money_idx] /= float(scale_money_by)  # safe: float I/O layer
    return X, y

def load_covertype_binary():
    X, y = fetch_covtype(return_X_y=True)
    y_bin = (y == 1).astype(int)  # Option A
    return X, y_bin

# ================================ Benchmark =================================
def run_benchmark():
    # lazy loaders so a missing credit_default.csv won’t crash everything
    dataset_loaders = {
        "breast_cancer": lambda: load_breast_cancer(return_X_y=True),
        "credit_card_default": load_credit_dataset,
        "covertype_bin": load_covertype_binary,
    }

    depths = [4, 5]
    trees_list = [50, 100]
    num_runs = 1

    for name, loader in tqdm(dataset_loaders.items(), desc="Datasets", file=sys.stdout):
        try:
            X, y = loader()
        except Exception as e:
            print(f"\n=== Dataset: {name} ===")
            print(f"Skipping due to load error: {e}")
            continue

        print(f"\n=== Dataset: {name} === (n={X.shape[0]}, d={X.shape[1]})")

        for depth in depths:
            for trees in trees_list:
                accs = {"Fixed-Wider": [], "XGB": []}
                times = {k: [] for k in accs}

                for run in range(num_runs):
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=0.2, random_state=42 + run
                    )

                    models = {
                        "Fixed-Wider": GenericFixedPointXGB(
                            n_estimators=trees, max_depth=depth, num_bins=128,
                            colsample_bytree=1.0, lambda_=1.0, gamma=0.0, seed=42,
                            no_repeat_features=True,
                            force_full_depth=True  # ensure exact full depth
                        ),
                        "XGB": xgb.XGBClassifier(
                            n_estimators=trees, max_depth=depth, verbosity=0, eval_metric="logloss"
                        ),
                    }

                    for label, model in models.items():
                        t0 = time.time()
                        model.fit(Xtr, ytr)
                        pred = model.predict(Xte)
                        if label == "Fixed-Wider":
                            pred = np.asarray(pred, dtype=int)
                        acc = np.mean(pred == yte)
                        accs[label].append(acc)
                        times[label].append(time.time() - t0)

                        if label == "Fixed-Wider" and run == 0:
                            os.makedirs("models", exist_ok=True)
                            model.save_model(f"models/model_{name}_depth{depth}_trees{trees}.json")
                            prefix = f"models/{name}_depth{depth}_trees{trees}"
                            export_model_artifacts(model, Xtr, Xte, prefix)

                print(f"{depth:<6} {trees:<6} | " + " ".join(f"{label:<13}" for label in accs))
                print("-" * 80)
                print(" " * 13 + " | " + " ".join(f"{np.mean(accs[k]):<13.4f}" for k in accs))
                print(" " * 13 + " | " + " ".join(f"{np.mean(times[k]):<13.2f}" for k in times))

if __name__ == "__main__":
    run_benchmark()

