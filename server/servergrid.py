
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
from tqdm import tqdm
import sys
import os
import pandas as pd
import zipfile
import urllib.request
from datetime import datetime

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
    def __init__(self, max_depth=3, lambda_=0.1, gamma=0.0, num_bins=512,
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

            left  = fixed_div_vec(GL*GL, HL + self.lambda_fp)     # SCALE
            right = fixed_div_vec(GR*GR, HR + self.lambda_fp)     # SCALE
            parent = fixed_div_scalar(Gtot*Gtot, Htot + self.lambda_fp)  # SCALE

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
                 lambda_=0.1, gamma=0.0, num_bins=512, colsample_bytree=1.0, seed=0,
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

# ============================= Dataset loaders ==============================
def load_covertype_binary():
    X, y = fetch_covtype(return_X_y=True)
    y_bin = (y == 1).astype(int)  # Map to binary class (1 vs rest)
    return X, y_bin

def load_adult_local():
    """
    Adult / Census Income (binary), ~48k rows.
    Reads from local files in the SAME FOLDER as this script: ./adult.data and ./adult.test
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "adult.data")
    test_path  = os.path.join(script_dir, "adult.test")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Expected 'adult.data' and 'adult.test' next to the script.")

    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    df_train = pd.read_csv(train_path, names=cols, na_values="?", skipinitialspace=True)
    df_test  = pd.read_csv(test_path,  names=cols, na_values="?", skipinitialspace=True, skiprows=1)

    # Normalize labels (strip trailing '.' in test)
    df_test["income"] = df_test["income"].astype(str).str.replace(".", "", regex=False)

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df = df.dropna(axis=0).reset_index(drop=True)

    y = (df["income"].astype(str).str.strip() == ">50K").astype(np.int32).to_numpy()
    X = pd.get_dummies(df.drop(columns=["income"]), drop_first=True, dtype=np.float32).to_numpy(dtype=np.float32)
    return X, y

def load_higgs_subset(n_rows=1_000_000, cache_dir="."):
    """
    HIGGS (binary). Uses the first n_rows lines for a 1M-row subset by default.
    Tries HIGGS.csv.gz (gzip) then HIGGS.zip (zip containing HIGGS.csv).
    If neither exists, downloads the .gz once.
    Returns: X (np.ndarray float32), y (np.ndarray int32)
    """
    os.makedirs(cache_dir, exist_ok=True)
    gz_path = os.path.join(cache_dir, "HIGGS.csv.gz")
    zip_path = os.path.join(cache_dir, "HIGGS.zip")

    if os.path.exists(gz_path):
        df = pd.read_csv(gz_path, compression="gzip", header=None, nrows=n_rows, usecols=list(range(0, 29)))
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            with z.open("HIGGS.csv") as f:
                df = pd.read_csv(f, header=None, nrows=n_rows, usecols=list(range(0, 29)))
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        print(f"[HIGGS] downloading to {gz_path} (this is ~2.6GB compressed)…")
        urllib.request.urlretrieve(url, gz_path)
        print("[HIGGS] download complete.")
        df = pd.read_csv(gz_path, compression="gzip", header=None, nrows=n_rows, usecols=list(range(0, 29)))

    y = df.iloc[:, 0].astype("int32").to_numpy()
    X = df.iloc[:, 1:].astype("float32").to_numpy()
    return X, y

# ================================ Logging ===================================
class TeeLogger:
    """Write stdout both to console and to a log file."""
    def __init__(self, filepath: str):
        self.file = open(filepath, "a", buffering=1)  # line-buffered
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

# ================================ Benchmark =================================
def run_benchmark():
    # Datasets requested for server runs
    dataset_loaders = {
        "adult": load_adult_local,                       # reads ./adult.data + ./adult.test (next to script)
        "higgs1m": lambda: load_higgs_subset(1_000_000), # 1M-row subset
        "covertype_bin": load_covertype_binary,          # sklearn fetch_covtype -> binary (class==1)
    }

    # Grid
    depths = [4, 5, 6]
    trees_list = [50, 100, 500, 800]
    bins_list = [256, 512, 1024]
    lambdas = [0.1, 0.5, 1.0]
    learning_rates = [0.3, 0.1]
    num_runs = 1  # single split for comparability

    for name, loader in tqdm(dataset_loaders.items(), desc="Datasets", file=sys.stdout):
        try:
            X, y = loader()
        except Exception as e:
            print(f"\n=== Dataset: {name} ===")
            print(f"Skipping due to load error: {e}")
            continue

        print(f"\n=== Dataset: {name} === (n={X.shape[0]}, d={X.shape[1]})")

        # Single split reused across all hyperparams for fairness
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        for depth in depths:
            for trees in trees_list:
                print(f"\n-- DATASET={name} depth={depth} trees={trees}")
                print("bins  lambda  eta   |  XGB_acc  XGB_time(s)  ||  FP_acc   FP_time(s)  |  Δacc(XGB-FP)  ok?")
                print("-"*112)

                for num_bins in bins_list:
                    for lam in lambdas:
                        for eta in learning_rates:
                            # XGBoost with matching params (closest equivalents)
                            t0 = time.time()
                            xgb_clf = xgb.XGBClassifier(
                                n_estimators=trees,
                                max_depth=depth,
                                learning_rate=eta,      # match eta
                                reg_lambda=lam,         # match lambda (L2)
                                tree_method="hist",     # histogram algorithm
                                max_bin=num_bins,       # match "bins" concept
                                n_jobs=1,
                                verbosity=0,
                                eval_metric="logloss",
                                objective="binary:logistic",
                            )
                            xgb_clf.fit(Xtr, ytr)
                            x_pred = xgb_clf.predict(Xte)
                            x_acc = np.mean(x_pred == yte)
                            x_time = time.time() - t0

                            # Fixed-point model with the same triple (bins, lambda, eta)
                            t1 = time.time()
                            fp_clf = GenericFixedPointXGB(
                                n_estimators=trees,
                                max_depth=depth,
                                num_bins=num_bins,
                                colsample_bytree=1.0,
                                lambda_=lam,
                                gamma=0.0,
                                seed=42,
                                no_repeat_features=True,
                                force_full_depth=True,
                                learning_rate=eta,
                            )
                            fp_clf.fit(Xtr, ytr)
                            fp_pred = fp_clf.predict(Xte)
                            fp_acc = np.mean(fp_pred == yte)
                            fp_time = time.time() - t1

                            diff = x_acc - fp_acc
                            ok_mark = "✓" if diff <= 0.02 else "✗"

                            print(f"{num_bins:<5} {lam:<7} {eta:<5} |  {x_acc:<7.4f}  {x_time:<12.2f}  ||  {fp_acc:<7.4f}  {fp_time:<10.2f}  |  {diff:<12.4f}  {ok_mark}")

if __name__ == "__main__":
    # set up tee logger to a timestamped file
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results_{stamp}.log"
    tee = TeeLogger(log_file)
    sys.stdout = tee
    print(f"# Log file: {log_file}")
    run_benchmark()
    tee.close()
