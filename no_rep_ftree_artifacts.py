#!/usr/bin/env python3
import numpy as np
import time, os, sys, json
from tqdm import tqdm

# Optional imports used in original script
try:
    from sklearn.datasets import load_breast_cancer, fetch_covtype
    from sklearn.model_selection import train_test_split as sk_split
except Exception:
    load_breast_cancer = None
    fetch_covtype = None
    sk_split = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ============================ Fixed-point config ============================
SCALE = 100_000
INV_SCALE = 1 / SCALE

def float_to_fixed(x: float) -> int:
    return int(round(x * SCALE))

def fixed_clip(x_fp: int, lo_fp: int, hi_fp: int) -> int:
    return max(lo_fp, min(hi_fp, x_fp))

def sigmoid_vec(x_fp: np.ndarray) -> np.ndarray:
    two = 2 * SCALE
    return np.where(x_fp <= -two, 0, np.where(x_fp >= two, SCALE, (x_fp + two) // 4)).astype(np.int64)

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
    # returns indices in [0..B] (#edges >= x)
    B = edges_fp.shape[0]
    # vectorized count: for each x, sum(x <= edges)
    return (x_col_fp[:, None] <= edges_fp[None, :]).sum(axis=1).astype(np.int32)

# ================== Single tree (vectorized, int-only math) =================
class XGBoostTreeClassifierFPFast:
    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0, num_bins=128,
                 no_repeat_features=True, force_full_depth=False, colsample_bytree=1.0, rng=None):
        self.max_depth = int(max_depth)
        self.lambda_fp = float_to_fixed(lambda_)
        self.gamma_fp  = float_to_fixed(gamma)
        self.num_bins  = int(num_bins)
        self.no_repeat_features = bool(no_repeat_features)
        self.force_full_depth = bool(force_full_depth)
        self.colsample_bytree = float(colsample_bytree)
        self.rng = np.random.default_rng(0) if rng is None else rng

        self.tree = None  # ('leaf', value_fp, leaf_id) or (feat, split_fp, left, right)
        self.bin_edges_fps = None   # list of per-feature edges (len d)
        self.bin_ids_per_feat = None  # per-feature bin IDs on train, shape (n,)

    # ---------------------- training helpers ----------------------
    def _leaf_value(self, idx, grad_fp, hess_fp) -> int:
        G = int(grad_fp[idx].sum())
        H = int(hess_fp[idx].sum())
        den = H + self.lambda_fp
        if den <= 0: den = 1
        return - (G // den)

    def _pad_to_depth(self, subtree, depth, used_mask):
        # optional: pad so resulting tree has exact max_depth; not required here
        return subtree

    def _fit_node(self, idx, grad_fp, hess_fp, depth, used_mask):
        if depth >= self.max_depth or idx.size < 2:
            return self._leaf_value(idx, grad_fp, hess_fp)

        d = len(self.bin_edges_fps)
        if self.no_repeat_features:
            avail = [j for j in range(d) if ((used_mask >> j) & 1) == 0]
        else:
            avail = list(range(d))
        if not avail:
            return self._leaf_value(idx, grad_fp, hess_fp)

        n_choose = max(1, int(len(avail) * self.colsample_bytree))
        features = self.rng.choice(avail, size=min(n_choose, len(avail)), replace=False)

        best_gain = float_to_fixed(-1_000_000_000)
        best_feat = None
        best_bin  = None

        Gtot = int(grad_fp[idx].sum())
        Htot = int(hess_fp[idx].sum())

        def quad(G, H):
            den = H + self.lambda_fp
            if den <= 0: den = 1
            return (G * G) // den

        base = quad(Gtot, Htot)

        for j in features:
            ids_j = self.bin_ids_per_feat[j][idx]            # in [0..B]
            if ids_j.max(initial=0) == ids_j.min(initial=0): # constant
                continue

            B = self.num_bins
            G_bins = np.zeros(B, dtype=np.int64)
            H_bins = np.zeros(B, dtype=np.int64)
            bi = np.clip(ids_j - 1, 0, B - 1)  # map to [0..B-1]
            np.add.at(G_bins, bi, grad_fp[idx])
            np.add.at(H_bins, bi, hess_fp[idx])

            Gleft = 0; Hleft = 0
            for b in range(B - 1):
                Gleft += int(G_bins[b])
                Hleft += int(H_bins[b])
                Gright = Gtot - Gleft
                Hright = Htot - Hleft
                score = quad(Gleft, Hleft) + quad(Gright, Hright) - base - self.gamma_fp
                if score > best_gain:
                    best_gain = score
                    best_feat = j
                    best_bin  = b

        if best_feat is None or best_gain <= 0:
            return self._leaf_value(idx, grad_fp, hess_fp)

        # partition
        ids_best = self.bin_ids_per_feat[best_feat][idx]
        bi = np.clip(ids_best - 1, 0, self.num_bins - 1)
        L_mask = bi <= best_bin
        L_idx  = idx[L_mask]
        R_idx  = idx[~L_mask]
        if L_idx.size == 0 or R_idx.size == 0:
            return self._leaf_value(idx, grad_fp, hess_fp)

        usedL = used_mask | (1 << best_feat) if self.no_repeat_features else used_mask
        usedR = usedL

        left  = self._fit_node(L_idx, grad_fp, hess_fp, depth + 1, usedL)
        right = self._fit_node(R_idx, grad_fp, hess_fp, depth + 1, usedR)

        threshold_fp = int(self.bin_edges_fps[best_feat][best_bin])
        return (best_feat, threshold_fp, left, right)

    # ---------------------- interface ----------------------
    def fit(self, X_fp: np.ndarray, y: np.ndarray, grad_fp: np.ndarray, hess_fp: np.ndarray):
        n, d = X_fp.shape
        self.bin_edges_fps = []
        self.bin_ids_per_feat = []
        for j in range(d):
            col = X_fp[:, j]
            cmin, cmax = int(col.min()), int(col.max())
            edges = fixed_linspace_fp(cmin, cmax, self.num_bins)
            self.bin_edges_fps.append(edges)
            self.bin_ids_per_feat.append(digitize_with_edges_int(col, edges))

        used0 = 0
        all_idx = np.arange(n, dtype=np.int32)
        self.tree = self._fit_node(all_idx, grad_fp, hess_fp, depth=0, used_mask=used0)
        self._assign_leaf_ids()
        return self

    def _predict_row_fp(self, x_fp_row, node):
        if not isinstance(node, tuple):
            return int(node)
        if isinstance(node[0], str) and node[0] == 'leaf':
            return int(node[1])
        feat, split_fp, l, r = node
        return self._predict_row_fp(x_fp_row, l if x_fp_row[feat] <= split_fp else r)

    def _assign_leaf_ids(self):
        lid = 0
        def wrap(node):
            nonlocal lid
            if not isinstance(node, tuple):
                t = ('leaf', int(node), lid); lid += 1; return t
            f, thr, L, R = node
            return (f, thr, wrap(L), wrap(R))
        if self.tree is not None:
            self.tree = wrap(self.tree)

    def apply_leaf_ids(self, X_fp: np.ndarray) -> np.ndarray:
        out = np.empty(X_fp.shape[0], dtype=np.int64)
        for i in range(X_fp.shape[0]):
            out[i] = self._predict_row_leaf_id(X_fp[i], self.tree)
        return out

    def _predict_row_leaf_id(self, x_fp_row, node):
        if not isinstance(node, tuple):
            return -1
        if isinstance(node[0], str) and node[0] == 'leaf':
            return int(node[2])
        f, thr, L, R = node
        return self._predict_row_leaf_id(x_fp_row, L if x_fp_row[f] <= thr else R)

    # ---------- Export helpers (leaf weights and internal nodes) ----------
    def _gather_leaf_weights(self, node, acc):
        if not isinstance(node, tuple):
            return
        if isinstance(node[0], str) and node[0] == 'leaf':
            acc.append((int(node[2]), int(node[1])))
            return
        f, thr, L, R = node
        self._gather_leaf_weights(L, acc)
        self._gather_leaf_weights(R, acc)

    def _gather_internal_nodes(self, node, acc, nid_counter):
        if not isinstance(node, tuple):
            return
        if isinstance(node[0], str) and node[0] == 'leaf':
            return
        nid = nid_counter[0]; nid_counter[0] += 1
        f, thr, L, R = node
        acc.append((nid, int(f), int(thr)))
        self._gather_internal_nodes(L, acc, nid_counter)
        self._gather_internal_nodes(R, acc, nid_counter)

    def export_leaf_weights(self):
        acc = []
        self._gather_leaf_weights(self.tree, acc)
        acc.sort(key=lambda t: t[0])
        return acc

    def export_internal_model(self):
        acc = []
        self._gather_internal_nodes(self.tree, acc, [0])
        return acc

# ================== Booster (vectorized, int-only math) =====================
class GenericFixedPointXGB:
    def __init__(self, n_estimators=50, max_depth=3, learning_rate=0.3,
                 lambda_=1.0, gamma=0.0, num_bins=128, colsample_bytree=1.0, seed=0,
                 no_repeat_features=True, force_full_depth=False):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate_fp = float_to_fixed(learning_rate)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.num_bins = int(num_bins)
        self.colsample_bytree = float(colsample_bytree)
        self.rng = np.random.default_rng(seed)
        self.no_repeat_features = bool(no_repeat_features)
        self.force_full_depth = bool(force_full_depth)

        self.trees = []
        self.initial_logit_fp = 0

    def _X_to_fixed(self, X: np.ndarray) -> np.ndarray:
        return np.rint(np.asarray(X, dtype=np.float64) * SCALE).astype(np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        X_fp = self._X_to_fixed(X)

        # init margin with a constant logit of class prior using a simple mapping
        p0 = float(y.mean()) if y.size > 0 else 0.5
        p0_fp = float_to_fixed(p0)
        self.initial_logit_fp = 0  # keep zero-init for simplicity

        # gradients/hessians at current margin
        margin_fp = np.zeros(n, dtype=np.int64)
        trees = []
        for t in range(self.n_estimators):
            p = sigmoid_vec(margin_fp)
            g = (p - y * SCALE).astype(np.int64)
            h = (p * (SCALE - p)) // SCALE
            h[h <= 0] = 1

            tree = XGBoostTreeClassifierFPFast(
                max_depth=self.max_depth, num_bins=self.num_bins, lambda_=self.lambda_,
                gamma=self.gamma, no_repeat_features=self.no_repeat_features,
                force_full_depth=self.force_full_depth, colsample_bytree=self.colsample_bytree, rng=self.rng
            )
            tree.fit(X_fp, y, g, h)
            trees.append(tree)

            # update margin
            leaf_vals = np.empty(n, dtype=np.int64)
            for i in range(n):
                leaf_vals[i] = tree._predict_row_fp(X_fp[i], tree.tree)
            margin_fp += (leaf_vals * self.learning_rate_fp) // SCALE

        self.trees = trees
        return self

    def predict_proba_fp(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_fp = self._X_to_fixed(X)
        logits_fp = np.zeros(X_fp.shape[0], dtype=np.int64)
        for tree in self.trees:
            for i in range(X_fp.shape[0]):
                logits_fp[i] += tree._predict_row_fp(X_fp[i], tree.tree)
        return sigmoid_vec(logits_fp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba_fp(X)
        return (p >= SCALE // 2).astype(np.int64)

    # ------------ artifacts -------------
    def apply_leaf_matrix(self, X: np.ndarray) -> np.ndarray:
        X_fp = self._X_to_fixed(X)
        n, T = X_fp.shape[0], len(self.trees)
        out = np.empty((n, T), dtype=np.int64)
        for t, tree in enumerate(self.trees):
            out[:, t] = tree.apply_leaf_ids(X_fp)
        return out

    def export_artifacts(self, out_dir: str, run_name: str):
        import csv
        os.makedirs(out_dir, exist_ok=True)
        lw = os.path.join(out_dir, f"{run_name}_leaf_weights.csv")
        mn = os.path.join(out_dir, f"{run_name}_model_nodes.csv")
        with open(lw, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree","leaf_id","weight_fp","weight"])
            for t, tree in enumerate(self.trees):
                for lid, w_fp in tree.export_leaf_weights():
                    w.writerow([t, lid, w_fp, w_fp * INV_SCALE])
        with open(mn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree","node_id","feature","threshold_fp","threshold"])
            for t, tree in enumerate(self.trees):
                for nid, feat, th_fp in tree.export_internal_model():
                    w.writerow([t, nid, feat, th_fp, th_fp * INV_SCALE])
        return lw, mn

# ============================= Dataset loaders ==============================
def load_credit_dataset():
    data = np.genfromtxt('credit_default.csv', delimiter=',', skip_header=1, filling_values=0)
    X_credit = data[:, 1:-1]
    y_credit = (data[:, -1] > 0.5).astype(int)
    return X_credit, y_credit

def load_breast_cat_onehot():
    # expects pre-encoded one-hot CSV with label last
    import csv
    X, y = [], []
    with open("breast_cat_numeric_unix.csv", "r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        for row in rdr:
            vals = [float(c.strip()) for c in row[:-1]]
            lab  = int(float(row[-1].strip()))
            X.append(vals); y.append(lab)
    return np.asarray(X, float), np.asarray(y, int)

# ================================ Benchmark =================================
def run_benchmark():
    datasets = []
    if load_breast_cancer is not None:
        # Use the one-hot CSV if available; else sklearn breast_cancer for a smoke test
        if os.path.exists("breast_cat_numeric_unix.csv"):
            datasets.append(("breast_cat", load_breast_cat_onehot))
        else:
            datasets.append(("breast_cancer", lambda: load_breast_cancer(return_X_y=True)))
    if os.path.exists("credit_default.csv"):
        datasets.append(("credit", load_credit_dataset))

    depths = [4,5]
    trees_list = [50,100]

    for name, loader in tqdm(datasets, desc="Datasets", file=sys.stdout):
        X, y = loader()
        if sk_split is not None:
            Xtr, Xte, ytr, yte = sk_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
        else:
            n = X.shape[0]
            idx = np.arange(n); np.random.default_rng(42).shuffle(idx)
            nte = int(round(0.2*n))
            te = idx[:nte]; tr = idx[nte:]
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

        models = {"Fixed-Wider": GenericFixedPointXGB(n_estimators=50, max_depth=4, num_bins=128,
                                                      learning_rate=0.3, lambda_=1.0, gamma=0.0,
                                                      no_repeat_features=True, force_full_depth=True)}
        if xgb is not None:
            models["XGB"] = xgb.XGBClassifier(n_estimators=50, max_depth=4, verbosity=0, eval_metric="logloss")

        for depth in depths:
            for trees in trees_list:
                # set per-run params
                models["Fixed-Wider"].n_estimators = trees
                models["Fixed-Wider"].max_depth = depth
                if "XGB" in models:
                    models["XGB"].n_estimators = trees
                    models["XGB"].max_depth = depth

                for label, model in models.items():
                    t0 = time.time()
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xte)
                    secs = time.time() - t0
                    acc = float((pred == yte).mean())
                    print(f"[{name}] {label:12s} depth={depth} trees={trees}  acc={acc:.4f}  time={secs:.2f}s")

                    if label == "Fixed-Wider":
                        # leaf IDs
                        os.makedirs("leaves", exist_ok=True)
                        np.savetxt(f"leaves/{name}_depth{depth}_trees{trees}_train_leaves.csv",
                                   models['Fixed-Wider'].apply_leaf_matrix(Xtr), delimiter=",", fmt="%d")
                        np.savetxt(f"leaves/{name}_depth{depth}_trees{trees}_test_leaves.csv",
                                   models['Fixed-Wider'].apply_leaf_matrix(Xte), delimiter=",", fmt="%d")
                        # artifacts
                        os.makedirs("artifacts", exist_ok=True)
                        run_name = f"{name}_depth{depth}_trees{trees}"
                        models['Fixed-Wider'].export_artifacts("artifacts", run_name)

if __name__ == "__main__":
    run_benchmark()
