import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
from tqdm import tqdm
import pandas as pd
import sys
import os
sys.stdout = os.popen('tee -a benchmark_output.txt', 'w')

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
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
    # z^5 / SCALE^4  (uses Python big ints; safe)
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

# ============ Helper to avoid overflow & keep correct gain scaling ==========
def _quad_over_denom(GL: np.ndarray, HL: np.ndarray, lam_fp: int) -> np.ndarray:
    """
    Returns floor(GL^2 / (HL + lam)), with SCALE-preserving result
    (GL, HL, lam are SCALE-scaled). Uses Python big ints when needed.
    """
    if GL.size == 0:
        return np.array([], dtype=np.int64)
    max_abs = int(np.max(np.abs(GL)))
    # conservative guard to avoid int64 overflow on squaring
    if max_abs <= 3_000_000_000:
        GL64 = GL.astype(np.int64)
        num = GL64 * GL64
        den = (HL.astype(np.int64) + np.int64(lam_fp))
        out = num // den
        return out.astype(np.int64)
    # fallback: big-int path (num_bins-length arrays => tiny perf hit)
    GL_obj = GL.astype(object)
    HL_obj = HL.astype(object)
    lam = int(lam_fp)
    return np.array([(g*g) // (h + lam) for g, h in zip(GL_obj, HL_obj)], dtype=object)

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
        # Initialize leaf maps for stable IDs
        self._enumerate_leaves()
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

        best_gain = float_to_fixed(-1_000_000_000)  # SCALE
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

            # Correct SCALE handling + overflow-safe
            left   = _quad_over_denom(GL, HL, self.lambda_fp)      # SCALE
            right  = _quad_over_denom(GR, HR, self.lambda_fp)      # SCALE
            parent = (Gtot * Gtot) // (Htot + self.lambda_fp)      # SCALE

            gain = ((left + right - parent) // 2)
            neg_big = int(float_to_fixed(-1e9))

            # Normalize dtype for argmax and mask invalids
            if isinstance(gain.dtype, np.dtype) and gain.dtype != object:
                gain = gain.astype(np.int64)
                gain[~ok] = neg_big
            else:
                gain = np.array([int(g) if ok_i else neg_big for g, ok_i in zip(gain, ok)], dtype=np.int64)

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

    # --------- Artifact helpers: leaf IDs, leaf weights, internal nodes ---------
    def _enumerate_leaves(self):
        # Returns (path_to_id, id_to_value). path is a string of 'L'/'R' decisions from root.
        # Leaves are enumerated left-to-right (in-order) with IDs 0..(num_leaves-1).
        path_to_id = {}
        id_to_value = {}
        counter = 0
        def dfs(node, path):
            nonlocal counter
            if not isinstance(node, tuple):
                # leaf
                path_to_id[path] = counter
                id_to_value[counter] = int(node)
                counter += 1
                return
            feat, thr, l, r = node
            dfs(l, path + 'L')
            dfs(r, path + 'R')
        dfs(self.tree, "")
        self._leaf_path_to_id = path_to_id
        self._leaf_id_to_value = id_to_value
        return path_to_id, id_to_value

    def _leaf_id_for_row(self, x_fp_row: np.ndarray) -> int:
        # Traverse the tree to the reached leaf and return its enumerated ID
        node = self.tree
        path = ""
        while isinstance(node, tuple):
            feat, thr, l, r = node
            if x_fp_row[int(feat)] <= int(thr):
                path += "L"
                node = l
            else:
                path += "R"
                node = r
        if not hasattr(self, "_leaf_path_to_id"):
            self._enumerate_leaves()
        return int(self._leaf_path_to_id[path])

    def export_leaf_weights(self):
        # Return list of dicts: {'leaf_id': id, 'value_fp': int}
        if not hasattr(self, "_leaf_id_to_value"):
            self._enumerate_leaves()
        return [{'leaf_id': int(k), 'value_fp': int(v)} for k, v in sorted(self._leaf_id_to_value.items())]

    def export_internal_nodes(self):
        # Return list of dicts: {'node_id': preorder_id, 'feature': j, 'threshold_fp': thr}
        nodes = []
        counter = 0
        def dfs(node):
            nonlocal counter
            if not isinstance(node, tuple):
                return
            feat, thr, l, r = node
            nodes.append({'node_id': counter, 'feature': int(feat), 'threshold_fp': int(thr)})
            counter += 1
            dfs(l); dfs(r)
        dfs(self.tree)
        return nodes

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


    # --------- Artifact helpers on the booster ---------
    def leaf_id_matrix_fp(self, X_fp: np.ndarray) -> np.ndarray:
        # Return an (n_samples, n_trees) array with the leaf ID reached by each sample per tree.
        n = X_fp.shape[0]
        T = len(self.trees)
        out = np.empty((n, T), dtype=np.int64)
        for t_idx, tree in enumerate(self.trees):
            if not hasattr(tree, "_leaf_path_to_id"):
                tree._enumerate_leaves()
            for i in range(n):
                out[i, t_idx] = tree._leaf_id_for_row(X_fp[i])
        return out

    def export_artifacts(self, X_train: np.ndarray, X_test: np.ndarray, out_prefix: str = "artifacts"):
        # Write leaf IDs (train/test), leaf weights, and model nodes to CSV files.
        import pandas as pd
        # Convert input to fixed once, consistent with training
        Xtr_fp = self._X_to_fixed(X_train)
        Xte_fp = self._X_to_fixed(X_test)

        # Leaf ID matrices
        leaf_ids_tr = self.leaf_id_matrix_fp(Xtr_fp)
        leaf_ids_te = self.leaf_id_matrix_fp(Xte_fp)
        df_tr = pd.DataFrame(leaf_ids_tr, columns=[f"tree_{i}" for i in range(leaf_ids_tr.shape[1])])
        df_te = pd.DataFrame(leaf_ids_te, columns=[f"tree_{i}" for i in range(leaf_ids_te.shape[1])])
        df_tr.to_csv(f"{out_prefix}_leaf_ids_train.csv", index=False)
        df_te.to_csv(f"{out_prefix}_leaf_ids_test.csv", index=False)

        # Leaf weights per tree
        rows_w = []
        for t_idx, tree in enumerate(self.trees):
            for rec in tree.export_leaf_weights():
                rec2 = dict(rec)
                rec2['tree'] = int(t_idx)
                rec2['value_float'] = (rec2['value_fp'] * INV_SCALE)
                rows_w.append(rec2)
        pd.DataFrame(rows_w, columns=['tree','leaf_id','value_fp','value_float']).to_csv(f"{out_prefix}_weights.csv", index=False)

        # Internal nodes per tree
        rows_m = []
        for t_idx, tree in enumerate(self.trees):
            for rec in tree.export_internal_nodes():
                rec2 = dict(rec)
                rec2['tree'] = int(t_idx)
                rec2['threshold_float'] = rec2['threshold_fp'] * INV_SCALE
                rows_m.append(rec2)
        pd.DataFrame(rows_m, columns=['tree','node_id','feature','threshold_fp','threshold_float']).to_csv(f"{out_prefix}_model.csv", index=False)

    def export_artifacts_matrices(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, out_prefix: str):
        # Build and save the four matrices requested by the user.
        import pandas as pd
        # ---- Training data matrix: n_data x (n_features+1), last column is y ----
        n_data, n_feat = X_train.shape
        train_mat = np.empty((n_data, n_feat + 1), dtype=np.float64)
        train_mat[:, :n_feat] = X_train
        train_mat[:, n_feat]  = y_train.astype(np.float64)
        pd.DataFrame(train_mat).to_csv(f"{out_prefix}_train_matrix.csv", index=False, header=False)

        # Convert to fixed for tree-dependent matrices
        Xtr_fp = self._X_to_fixed(X_train)
        Xte_fp = self._X_to_fixed(X_test)

        # ---- Leaf ID matrix: n_tree x n_data ----
        leaf_ids = self.leaf_id_matrix_fp(Xtr_fp)   # shape (n_data, n_trees)
        leaf_ids_mat = leaf_ids.T                   # (n_trees, n_data)
        pd.DataFrame(leaf_ids_mat).to_csv(f"{out_prefix}_leaf_id_matrix.csv", index=False, header=False)

        # ---- Weights matrix: n_tree x n_leaves (n_leaves = 2^h) ----
        h = self.max_depth
        n_leaves = 1 << h
        T = len(self.trees)
        weights_mat_fp = np.zeros((T, n_leaves), dtype=np.int64)
        for t_idx, tree in enumerate(self.trees):
            if not hasattr(tree, "_leaf_id_to_value"):
                tree._enumerate_leaves()
            for leaf_id, val in tree._leaf_id_to_value.items():
                if 0 <= int(leaf_id) < n_leaves:
                    weights_mat_fp[t_idx, int(leaf_id)] = int(val)
        pd.DataFrame(weights_mat_fp).to_csv(f"{out_prefix}_weights_matrix_fp.csv", index=False, header=False)
        # Also dump float convenience
        weights_mat_float = weights_mat_fp.astype(np.float64) * INV_SCALE
        pd.DataFrame(weights_mat_float).to_csv(f"{out_prefix}_weights_matrix_float.csv", index=False, header=False)

        # ---- Model matrix: n_tree x (2*n_internal_nodes), first features then thresholds ----
        n_internal = (1 << h) - 1
        model_mat_feat = np.full((T, n_internal), -1, dtype=np.int64)
        model_mat_thr_fp = np.zeros((T, n_internal), dtype=np.int64)
        for t_idx, tree in enumerate(self.trees):
            nodes = []
            def dfs(node):
                if not isinstance(node, tuple):
                    return
                feat, thr, l, r = node
                nodes.append((int(feat), int(thr)))
                dfs(l); dfs(r)
            dfs(tree.tree)
            # Ensure size n_internal; pad if needed
            for k in range(min(n_internal, len(nodes))):
                model_mat_feat[t_idx, k] = nodes[k][0]
                model_mat_thr_fp[t_idx, k] = nodes[k][1]
        # Concatenate features then thresholds along columns
        model_mat_fp = np.concatenate([model_mat_feat, model_mat_thr_fp], axis=1)
        pd.DataFrame(model_mat_fp).to_csv(f"{out_prefix}_model_matrix_fp.csv", index=False, header=False)
        # Also dump float thresholds
        model_mat_thr_float = model_mat_thr_fp.astype(np.float64) * INV_SCALE
        model_mat_float = np.concatenate([model_mat_feat.astype(np.float64), model_mat_thr_float], axis=1)
        pd.DataFrame(model_mat_float).to_csv(f"{out_prefix}_model_matrix_float.csv", index=False, header=False)

    def validate_no_repeat_features(self):
        # Check that no feature repeats along any root->leaf path for REAL splits (ignore dummy splits where left is right)
        problems = []
        def is_dummy(node):
            # a padded dummy node has l is r in our construction
            if not isinstance(node, tuple): return False
            _, _, l, r = node
            return isinstance(l, tuple) or isinstance(r, tuple) and (l is r)
        for t_idx, tree in enumerate(self.trees):
            def dfs(node, used):
                if not isinstance(node, tuple):
                    return
                feat, thr, l, r = node
                # dummy nodes: skip marking used
                dummy = (isinstance(node, tuple) and isinstance(l, (tuple,int)) and isinstance(r, (tuple,int)) and (l is r))
                if not dummy:
                    if int(feat) in used:
                        problems.append((t_idx, feat))
                    used = used | {int(feat)}
                dfs(l, used); dfs(r, used)
            dfs(tree.tree, set())
        return problems
# ============================= Dataset loaders ==============================
def load_credit_dataset():
    # expects a local CSV named 'credit_default.csv'
    data = np.genfromtxt('credit_default.csv', delimiter=',', skip_header=1, filling_values=0)
    X_credit = data[:, 1:-1]
    y_credit = (data[:, -1] > 0.5).astype(int)
    return X_credit, y_credit

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

# ============================ Adult (UCI) loader ============================
def load_adult_dataset():
    """Load the UCI Adult dataset from local files in the current folder:
    - adult.data
    - adult.test
    - adult.names (unused by the loader, optional)
    Returns X (numpy array), y (0/1 numpy array) with one-hot encoded categoricals."""
    cols = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","label"
    ]
    # Train part
    df_train = pd.read_csv("adult.data", header=None, names=cols,
                           na_values="?", skipinitialspace=True)
    # Test part (first line is a header-like comment in UCI files)
    df_test = pd.read_csv("adult.test", header=None, names=cols,
                          na_values="?", skipinitialspace=True, skiprows=1)
    df = pd.concat([df_train, df_test], ignore_index=True)
    # Clean labels: remove trailing '.' and map to 0/1
    df["label"] = df["label"].astype(str).str.strip().str.replace(".", "", regex=False)
    df = df.dropna()
    y = (df["label"] == ">50K").astype(int).to_numpy()

    # Separate numeric and categorical
    numeric = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    categorical = [c for c in df.columns if c not in numeric + ["label"]]

    # One-hot encode categoricals
    df_cat = pd.get_dummies(df[categorical], drop_first=False)
    X = pd.concat([df[numeric].reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1).to_numpy(dtype=float)
    return X, y

# ================================ Benchmark =================================
def run_benchmark():
    # lazy loaders so a missing credit_default.csv won’t crash everything
    dataset_loaders = {
        "breast_cancer": lambda: load_breast_cancer(return_X_y=True),
        "credit_card_default": load_credit_dataset,
        "covertype_50k_bin": load_covertype_binary_50k,
	"adult": load_adult_dataset
    }

    depths = [4,5]
    trees_list = [50,100]
    num_runs = 1
    tol = 0.02  # pass threshold for |acc_fixed - acc_xgb|

    for name, loader in tqdm(dataset_loaders.items(), desc="Datasets", file=sys.stdout):
        try:
            X, y = loader()
        except Exception as e:
            print(f"\n=== Dataset: {name} ===")
            print(f"Skipping due to load error: {e}")
            continue

        print(f"\n=== Dataset: {name} === (n={X.shape[0]}, d={X.shape[1]})")
        header = (
            f"{'depth':<6} {'trees':<6} | "
            f"{'Fixed-Wider acc':<15} {'XGB acc':<10} {'|Δ|':<8} {'≤0.02?':<7} "
            f"{'Fixed s':<9} {'XGB s':<9} {'Slowdown×':<10}"
        )
        print(header)
        print("-" * len(header))

        for depth in depths:
            for trees in trees_list:
                fixed_accs, xgb_accs = [], []
                fixed_times, xgb_times = [], []

                for run in range(num_runs):
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=0.2, random_state=42 + run
                    )

                    models = {
    			"Fixed-Wider": GenericFixedPointXGB(
        		n_estimators=trees, max_depth=depth, num_bins=128,
        		colsample_bytree=1.0, lambda_=1.0, gamma=0.0, seed=42 + run,
        		no_repeat_features=True, force_full_depth=True
    			),
    			"XGB": xgb.XGBClassifier(
        		n_estimators=trees, max_depth=depth, verbosity=0, eval_metric="logloss",
        		random_state=42 + run
    			),
			}


                    # Fixed-Wider
                    t0 = time.time()
                    models["Fixed-Wider"].fit(Xtr, ytr)
                    # Validate no-repeat-features constraint
                    try:
                        probs = models["Fixed-Wider"].validate_no_repeat_features()
                        if probs:
                            print(f"[warn] no-repeat-features violations: {probs[:5]}... total={len(probs)}")
                    except Exception as e:
                        print(f"[warn] validation failed: {e}")
                    pred_fixed = models["Fixed-Wider"].predict(Xte)
                    # Export artifacts for the custom implementation
                    try:
                        models["Fixed-Wider"].export_artifacts(Xtr, Xte, out_prefix=os.path.join(ARTIFACTS_DIR, f"{name}_d{depth}_t{trees}"))
                    except Exception as e:
                        print(f"[warn] export_artifacts failed: {e}")
                    # Export matrix forms
                    try:
                        models["Fixed-Wider"].export_artifacts_matrices(Xtr, ytr, Xte, out_prefix=os.path.join(ARTIFACTS_DIR, f"{name}_d{depth}_t{trees}"))
                    except Exception as e:
                        print(f"[warn] export_artifacts_matrices failed: {e}")
                    t_fixed = time.time() - t0
                    acc_fixed = float(np.mean(pred_fixed == yte))
                    fixed_accs.append(acc_fixed)
                    fixed_times.append(t_fixed)

                    # Real XGB
                    t0 = time.time()
                    models["XGB"].fit(Xtr, ytr)
                    pred_xgb = models["XGB"].predict(Xte)
                    t_xgb = time.time() - t0
                    acc_xgb = float(np.mean(pred_xgb == yte))
                    xgb_accs.append(acc_xgb)
                    xgb_times.append(t_xgb)

                acc_fixed_mean = float(np.mean(fixed_accs))
                acc_xgb_mean   = float(np.mean(xgb_accs))
                diff = abs(acc_fixed_mean - acc_xgb_mean)
                passmark = "✓" if diff <= tol else "✗"

                fixed_s = float(np.mean(fixed_times))
                xgb_s   = float(np.mean(xgb_times))
                slowdown = (fixed_s / xgb_s) if xgb_s > 0 else float('inf')

                print(
                    f"{depth:<6} {trees:<6} | "
                    f"{acc_fixed_mean:<15.4f} {acc_xgb_mean:<10.4f} {diff:<8.4f} {passmark:<7} "
                    f"{fixed_s:<9.2f} {xgb_s:<9.2f} {slowdown:<10.2f}"
                )



if __name__ == "__main__":
    run_benchmark()
