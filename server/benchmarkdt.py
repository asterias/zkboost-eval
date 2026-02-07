# benchmark_dt_fp_fulltree.py
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDT
import time
from tqdm import tqdm
import pandas as pd
import sys
import os

# ============================ Fixed-point config ============================
SCALE = 100_000
INV_SCALE = 1 / SCALE  # never used in artifacts; kept for internal sanity when needed

def float_to_fixed(x: float) -> int:
    return int(round(x * SCALE))

def fixed_clip(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def fixed_div_scalar(num: int, den: int) -> int:
    # (num / den) in fixed point (SCALE); returns 0 if den==0
    return (num * SCALE) // den if den != 0 else 0

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
    out[-1] = stop_fp
    return out

def digitize_with_edges_int(x_col_fp: np.ndarray, edges_fp: np.ndarray) -> np.ndarray:
    # integer version of np.digitize(x, edges[:-1], right=False)
    return np.searchsorted(edges_fp[:-1], x_col_fp, side='right').astype(np.int64)

def bincount_sum_int(ids: np.ndarray, w: np.ndarray, bins: int) -> np.ndarray:
    out = np.zeros(bins, dtype=np.int64)
    np.add.at(out, ids, w)
    return out

# ======================= Fixed-point Decision Tree (Gini) ====================
class DecisionTreeClassifierFP:
    """
    Binary CART-style tree with Gini impurity, fully integer.
    - Features pre-binned to 'num_bins' (uniform per feature).
    - Splits picked by maximizing Gini gain.
    - Predictions are majority class at leaves (0/1), exported as fp (0 or SCALE).
    - force_full_depth=True pads with dummy splits to ensure exactly 2^max_depth leaves.
    """
    def __init__(self, max_depth=7, num_bins=128, min_samples_leaf=5, seed=0, force_full_depth=True):
        self.max_depth = int(max_depth)
        self.num_bins = int(num_bins)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.force_full_depth = bool(force_full_depth)

        self.tree = None
        self.bin_edges_fps = None
        self.bin_ids_per_feat = None

    # ---------- public API ----------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_fp = self._X_to_fixed(X)
        self._prebin(X_fp)

        y01 = y.astype(np.int64)
        idx_all = np.arange(X.shape[0], dtype=np.int64)
        core = self._fit_node(idx_all, y01, depth=0)
        self.tree = self._pad_to_depth(core, 0) if self.force_full_depth else core
        self._enumerate_leaves()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_fp = self._X_to_fixed(X)
        out = np.empty(X_fp.shape[0], dtype=np.int64)
        for i in range(X_fp.shape[0]):
            out[i] = self._predict_row(X_fp[i], self.tree)
        return out

    # ---------- internals ----------
    def _X_to_fixed(self, X: np.ndarray) -> np.ndarray:
        return np.rint(X * SCALE).astype(np.int64)

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

    @staticmethod
    def _leaf_majority(y_idx: np.ndarray) -> int:
        # Return majority class (0/1), tie -> 1
        s = int(y_idx.sum(dtype=np.int64))
        n = y_idx.size
        return 1 if s * 2 >= n else 0

    @staticmethod
    def _gini_from_counts(n0: int, n1: int) -> int:
        # Gini = 2 p (1-p), scaled by SCALE
        n = n0 + n1
        if n == 0:
            return 0
        p1_fp = fixed_div_scalar(n1, n)                 # SCALE
        return (2 * p1_fp * (SCALE - p1_fp)) // SCALE   # SCALE

    def _fit_node(self, idx: np.ndarray, y01: np.ndarray, depth: int):
        # stop criteria
        if depth >= self.max_depth or idx.size <= self.min_samples_leaf:
            return self._leaf_majority(y01[idx])

        # pure -> leaf
        s = int(y01[idx].sum(dtype=np.int64))
        if s == 0 or s == idx.size:
            return 1 if s == idx.size else 0

        parent_gini = self._gini_from_counts(idx.size - s, s)

        best_gain = -1
        best_feat = None
        best_bin  = None

        d = len(self.bin_edges_fps)
        for j in range(d):
            ids_j = self.bin_ids_per_feat[j][idx]
            if ids_j.max(initial=0) == ids_j.min(initial=0):
                continue  # constant in this node

            ones_per_bin = bincount_sum_int(ids_j, y01[idx], self.num_bins + 1)
            cnt_per_bin  = bincount_sum_int(ids_j, np.ones_like(ids_j, dtype=np.int64), self.num_bins + 1)

            onesL = np.cumsum(ones_per_bin[:-1], dtype=np.int64)
            cntL  = np.cumsum(cnt_per_bin[:-1], dtype=np.int64)
            onesR = int(ones_per_bin.sum(dtype=np.int64)) - onesL
            cntR  = int(cnt_per_bin.sum(dtype=np.int64)) - cntL

            # child impurities (SCALE)
            giniL = np.array([self._gini_from_counts(int(cL - oL), int(oL)) if cL > 0 else 0
                              for oL, cL in zip(onesL, cntL)], dtype=np.int64)
            giniR = np.array([self._gini_from_counts(int(cR - oR), int(oR)) if cR > 0 else 0
                              for oR, cR in zip(onesR, cntR)], dtype=np.int64)

            n = idx.size
            wL = (cntL * SCALE) // n             # SCALE
            wR = (cntR * SCALE) // n             # SCALE
            child_imp = (wL * giniL + wR * giniR) // SCALE   # SCALE

            gain = parent_gini - child_imp       # vector (SCALE)
            ok = (cntL >= self.min_samples_leaf) & (cntR >= self.min_samples_leaf)
            if not np.any(ok):
                continue
            gain_masked = np.where(ok, gain, -10**12)

            b = int(np.argmax(gain_masked))
            g = int(gain_masked[b])
            if g > best_gain:
                best_gain = g
                best_feat = j
                best_bin  = b

        if best_feat is None or best_gain <= 0:
            return self._leaf_majority(y01[idx])

        # split
        ids_best = self.bin_ids_per_feat[best_feat][idx]
        left_mask  = ids_best <= best_bin
        right_mask = ~left_mask
        left_idx  = idx[left_mask]
        right_idx = idx[right_mask]

        if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
            return self._leaf_majority(y01[idx])

        left_node  = self._fit_node(left_idx,  y01, depth + 1)
        right_node = self._fit_node(right_idx, y01, depth + 1)
        split_fp = int(self.bin_edges_fps[best_feat][best_bin])
        return (int(best_feat), int(split_fp), left_node, right_node)

    def _choose_dummy_feature(self) -> int:
        # Prefer a feature with varying edges; fallback to 0
        for j, edges in enumerate(self.bin_edges_fps):
            if len(edges) > 1 and int(edges[0]) != int(edges[-1]):
                return int(j)
        return 0

    def _pad_to_depth(self, node, depth: int):
        """
        Wrap `node` in dummy splits until reaching `self.max_depth`.
        Dummy splits do NOT alter predictions; both children are `node`.
        """
        cur = node
        cur_depth = depth
        while cur_depth < self.max_depth:
            j = self._choose_dummy_feature()
            edges = self.bin_edges_fps[j]
            split_fp = int(edges[len(edges)//2]) if len(edges) else 0
            cur = (int(j), split_fp, cur, cur)
            cur_depth += 1
        return cur
        
    def _predict_row(self, x_fp_row: np.ndarray, node):
    	if not isinstance(node, tuple):
        	return node
    	feat, thr, l, r = node
    	if x_fp_row[int(feat)] <= int(thr):
        	return self._predict_row(x_fp_row, l)
    	else:
        	return self._predict_row(x_fp_row, r)


    # --------- leaf enumeration & artifact helpers ---------
    def _enumerate_leaves(self):
        # Map path ("LLR...") -> id (0..), and id -> fp value (0 or SCALE)
        path_to_id, id_to_value_fp = {}, {}
        counter = 0
        def dfs(node, path):
            nonlocal counter
            if not isinstance(node, tuple):
                # node is label 0/1; store as SCALE-scaled
                val_fp = SCALE if int(node) == 1 else 0
                path_to_id[path] = counter
                id_to_value_fp[counter] = int(val_fp)
                counter += 1
                return
            _, _, l, r = node
            dfs(l, path + "L"); dfs(r, path + "R")
        dfs(self.tree, "")
        self._leaf_path_to_id = path_to_id
        self._leaf_id_to_value_fp = id_to_value_fp
        return path_to_id, id_to_value_fp

    def _leaf_id_for_row(self, x_fp_row: np.ndarray) -> int:
        node = self.tree
        path = ""
        while isinstance(node, tuple):
            feat, thr, l, r = node
            if x_fp_row[int(feat)] <= int(thr):
                path += "L"; node = l
            else:
                path += "R"; node = r
        if not hasattr(self, "_leaf_path_to_id"):
            self._enumerate_leaves()
        return int(self._leaf_path_to_id[path])

    def export_leaf_values_fp(self):
        if not hasattr(self, "_leaf_id_to_value_fp"):
            self._enumerate_leaves()
        return [{'leaf_id': int(k), 'value_fp': int(v)} for k, v in sorted(self._leaf_id_to_value_fp.items())]

    def export_internal_nodes_fp(self):
        # (node_id, feature, threshold_fp) in preorder
        nodes, counter = [], 0
        def dfs(node):
            nonlocal counter
            if not isinstance(node, tuple): return
            feat, thr, l, r = node
            nodes.append({'node_id': counter, 'feature': int(feat), 'threshold_fp': int(thr)})
            counter += 1
            dfs(l); dfs(r)
        dfs(self.tree)
        return nodes

# ============================= Dataset loaders ==============================
def load_credit_dataset(scale_money_by=1000.0):
    data = np.genfromtxt('credit_default.csv', delimiter=',', skip_header=1, filling_values=0)
    X = data[:, 1:-1].astype(np.float64)
    y = (data[:, -1] > 0.5).astype(int)
    if scale_money_by:
        money_idx = np.array([0, *range(11, 17), *range(17, 23)], dtype=np.int64)
        X[:, money_idx] /= float(scale_money_by)  # safe: float I/O layer
    return X, y

def load_covertype_binary_50k(seed=42):
    X, y = fetch_covtype(return_X_y=True)
    y_bin = (y == 1).astype(int)  # class 1 vs rest
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > 50_000:
        idx = rng.choice(n, size=50_000, replace=False)
        X = X[idx]
        y_bin = y_bin[idx]
    return X, y_bin

def load_adult_dataset():
    """Load UCI Adult from local 'adult.data' and 'adult.test'."""
    cols = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","label"
    ]
    df_train = pd.read_csv("adult.data", header=None, names=cols, na_values="?", skipinitialspace=True)
    df_test  = pd.read_csv("adult.test", header=None, names=cols, na_values="?", skipinitialspace=True, skiprows=1)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df["label"] = df["label"].astype(str).str.strip().str.replace(".", "", regex=False)
    df = df.dropna()
    y = (df["label"] == ">50K").astype(int).to_numpy()

    numeric = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    categorical = [c for c in df.columns if c not in numeric + ["label"]]
    df_cat = pd.get_dummies(df[categorical], drop_first=False)
    X = pd.concat([df[numeric].reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1).to_numpy(dtype=float)
    return X, y

# ================================ Exports (fp-only) =========================
def export_dt_artifacts_fp_only(dt_model: DecisionTreeClassifierFP, X_train: np.ndarray, X_test: np.ndarray, out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # Convert to fixed (consistent with training)
    Xtr_fp = dt_model._X_to_fixed(X_train)
    Xte_fp = dt_model._X_to_fixed(X_test)

    # Per-sample leaf IDs (train/test). Single tree -> 1 column.
    n_tr, n_te = Xtr_fp.shape[0], Xte_fp.shape[0]
    leaf_ids_tr = np.array([dt_model._leaf_id_for_row(Xtr_fp[i]) for i in range(n_tr)], dtype=np.int64).reshape(-1, 1)
    leaf_ids_te = np.array([dt_model._leaf_id_for_row(Xte_fp[i]) for i in range(n_te)], dtype=np.int64).reshape(-1, 1)
    pd.DataFrame(leaf_ids_tr, columns=["tree_0"]).to_csv(f"{out_prefix}_leaf_ids_train.csv", index=False)
    pd.DataFrame(leaf_ids_te, columns=["tree_0"]).to_csv(f"{out_prefix}_leaf_ids_test.csv", index=False)

    # Leaf value list (fp): keep XGBoost-like column naming
    rows_w = []
    for rec in dt_model.export_leaf_values_fp():
        rec2 = dict(rec)
        rec2['tree'] = 0
        rows_w.append(rec2)
    pd.DataFrame(rows_w, columns=['tree','leaf_id','value_fp']).to_csv(f"{out_prefix}_weights.csv", index=False)

    # Internal nodes list (fp only)
    rows_m = []
    for rec in dt_model.export_internal_nodes_fp():
        rec2 = dict(rec); rec2['tree'] = 0
        rows_m.append(rec2)
    pd.DataFrame(rows_m, columns=['tree','node_id','feature','threshold_fp']).to_csv(f"{out_prefix}_model.csv", index=False)

def export_dt_artifact_matrices_fp_only(dt_model: DecisionTreeClassifierFP, X_train: np.ndarray, out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # Leaf-ID matrix: (n_tree x n_train) -> 1 x n_train
    Xtr_fp = dt_model._X_to_fixed(X_train)
    leaf_ids = np.array([dt_model._leaf_id_for_row(x) for x in Xtr_fp], dtype=np.int64)[None, :]
    pd.DataFrame(leaf_ids).to_csv(f"{out_prefix}_leaf_id_matrix.csv", index=False, header=False)

    # Weights matrix (fp): (n_tree x 2^h) -> 1 x 2^h, entries 0 or SCALE
    h = dt_model.max_depth
    n_leaves = 1 << h
    weights_mat_fp = np.zeros((1, n_leaves), dtype=np.int64)
    if not hasattr(dt_model, "_leaf_id_to_value_fp"):
        dt_model._enumerate_leaves()
    for leaf_id, val_fp in dt_model._leaf_id_to_value_fp.items():
        if 0 <= int(leaf_id) < n_leaves:
            weights_mat_fp[0, int(leaf_id)] = int(val_fp)
    pd.DataFrame(weights_mat_fp).to_csv(f"{out_prefix}_weights_matrix_fp.csv", index=False, header=False)

    # Model matrix (fp): features || thresholds, shape 1 x (2*n_internal)
    n_internal = (1 << h) - 1
    model_feat = np.full((1, n_internal), -1, dtype=np.int64)
    model_thr_fp = np.zeros((1, n_internal), dtype=np.int64)
    nodes = []
    def dfs(node):
        if not isinstance(node, tuple): return
        feat, thr, l, r = node
        nodes.append((int(feat), int(thr)))
        dfs(l); dfs(r)
    dfs(dt_model.tree)
    for k in range(min(n_internal, len(nodes))):
        model_feat[0, k] = nodes[k][0]
        model_thr_fp[0, k] = nodes[k][1]
    model_mat_fp = np.concatenate([model_feat, model_thr_fp], axis=1)
    pd.DataFrame(model_mat_fp).to_csv(f"{out_prefix}_model_matrix_fp.csv", index=False, header=False)

# ================================ Benchmark =================================
def run_benchmark():
    dataset_loaders = {
        "breast_cancer": lambda: load_breast_cancer(return_X_y=True),
        "credit_card_default": load_credit_dataset,
        "covertype_50k_bin": load_covertype_binary_50k,
        "adult": load_adult_dataset,
    }

    # Balanced universal defaults
    depths = [3, 5, 7, 9]
    min_leaf_list = [5, 10, 20, 50]
    num_runs = 1
    tol = 0.02  # acceptable |Δ| in accuracy

    print("\n=== Fixed-point Decision Tree (full tree) vs scikit-learn DecisionTree ===")
    for name, loader in tqdm(dataset_loaders.items(), desc="Datasets", file=sys.stdout):
        try:
            X, y = loader()
        except Exception as e:
            print(f"\n=== Dataset: {name} ===")
            print(f"Skipping due to load error: {e}")
            continue

        print(f"\n=== Dataset: {name} === (n={X.shape[0]}, d={X.shape[1]})")
        header = (
            f"{'depth':<6} {'min_leaf':<8} | "
            f"{'FP-DT acc':<10} {'SK-DT acc':<10} {'|Δ|':<8} {'≤0.02?':<7} "
            f"{'FP s':<9} {'SK s':<9} {'Slowdown×':<10}"
        )
        print(header)
        print("-" * len(header))

        for depth in depths:
            for mleaf in min_leaf_list:
                fp_accs, sk_accs = [], []
                fp_times, sk_times = [], []

                for run in range(num_runs):
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42 + run)

                    # Fixed-point DT (full tree enforced)
                    t0 = time.time()
                    fp_dt = DecisionTreeClassifierFP(max_depth=depth, num_bins=128,
                                                     min_samples_leaf=mleaf, seed=42 + run,
                                                     force_full_depth=True)
                    fp_dt.fit(Xtr, ytr)
                    pred_fp = fp_dt.predict(Xte)
                    t_fp = time.time() - t0
                    acc_fp = float(np.mean(pred_fp == yte))
                    fp_accs.append(acc_fp); fp_times.append(t_fp)

                    # Export fp-only artifacts (single tree)
                    out_prefix = os.path.join(ARTIFACTS_DIR, f"{name}_d{depth}_ml{mleaf}")
                    export_dt_artifacts_fp_only(fp_dt, Xtr, Xte, out_prefix=out_prefix)
                    export_dt_artifact_matrices_fp_only(fp_dt, Xtr, out_prefix=out_prefix)

                    # scikit-learn float DT (for comparison only)
                    t0 = time.time()
                    sk_dt = SkDT(criterion="gini", splitter="best", max_depth=depth,
                                 min_samples_leaf=mleaf, random_state=42 + run)
                    sk_dt.fit(Xtr, ytr)
                    pred_sk = sk_dt.predict(Xte)
                    t_sk = time.time() - t0
                    acc_sk = float(np.mean(pred_sk == yte))
                    sk_accs.append(acc_sk); sk_times.append(t_sk)

                acc_fp_mean = float(np.mean(fp_accs))
                acc_sk_mean = float(np.mean(sk_accs))
                diff = abs(acc_fp_mean - acc_sk_mean)
                passmark = "✓" if diff <= tol else "✗"

                fp_s = float(np.mean(fp_times))
                sk_s = float(np.mean(sk_times))
                slowdown = (fp_s / sk_s) if sk_s > 0 else float('inf')

                print(
                    f"{depth:<6} {mleaf:<8} | "
                    f"{acc_fp_mean:<10.4f} {acc_sk_mean:<10.4f} {diff:<8.4f} {passmark:<7} "
                    f"{fp_s:<9.2f} {sk_s:<9.2f} {slowdown:<10.2f}"
                )

if __name__ == "__main__":
    # Optional: tee output like your booster script
    ARTIFACTS_DIR = "artifacts_dt"
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    run_benchmark()

