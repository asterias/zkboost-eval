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
INV_SCALE = 1 / SCALE  # for readability when printing

def float_to_fixed(x: float) -> int:
    return int(round(x * SCALE))

def fixed_to_float(x: int) -> float:
    return x * INV_SCALE  # never used in train/infer math

def fixed_div_scalar(num: int, den: int) -> int:
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
    p_fp = fixed_clip(p_fp, float_to_fixed(0.001), float_to_fixed(0.999))
    z = 2 * p_fp - SCALE
    atanh = z + (_pow3(z) // 3) + (_pow5(z) // 5)
    return 2 * atanh

# ---- vectorized fixed-point sigmoid (wider piecewise) ----
def sigmoid_vec(x_fp: np.ndarray) -> np.ndarray:
    two = 2 * SCALE
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

# =================== Fast fixed-point tree (vectorized) =====================
class XGBoostTreeClassifierFPFast:
    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0, num_bins=128,
                 colsample_bytree=1.0, seed=0):
        self.max_depth = max_depth
        self.lambda_fp = float_to_fixed(lambda_)
        self.gamma_fp  = float_to_fixed(gamma)
        self.num_bins  = num_bins
        self.colsample_bytree = colsample_bytree
        self.rng = np.random.default_rng(seed)

        self.tree = None
        self.bin_edges_fps = None
        self.bin_ids_per_feat = None

    def fit(self, X_fp: np.ndarray, grad_fp: np.ndarray, hess_fp: np.ndarray):
        n, _ = X_fp.shape
        self._prebin(X_fp)
        idx_all = np.arange(n, dtype=np.int64)
        self.tree = self._fit_node(idx_all, grad_fp, hess_fp, depth=0)
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
        val = fixed_div_scalar(G, H + self.lambda_fp)
        return fixed_clip(val, float_to_fixed(-1.0), float_to_fixed(1.0))

    def _fit_node(self, idx: np.ndarray, grad_fp: np.ndarray, hess_fp: np.ndarray, depth: int):
        if depth >= self.max_depth or idx.size < 2:
            return self._leaf_value(idx, grad_fp, hess_fp)

        d = len(self.bin_edges_fps)
        n_choose = max(1, int(d * self.colsample_bytree))
        features = self.rng.choice(d, size=n_choose, replace=False)

        best_gain = float_to_fixed(-1e9)
        best_feat = None
        best_bin  = None

        for j in features:
            ids_j = self.bin_ids_per_feat[j][idx]
            if ids_j.max(initial=0) == ids_j.min(initial=0):
                continue  # constant in this node

            G = np.bincount(ids_j, weights=grad_fp[idx], minlength=self.num_bins+1).astype(np.int64)
            H = np.bincount(ids_j, weights=hess_fp[idx], minlength=self.num_bins+1).astype(np.int64)

            GL = np.cumsum(G[:-1], dtype=np.int64)
            HL = np.cumsum(H[:-1], dtype=np.int64)
            Gtot = int(G.sum(dtype=np.int64))
            Htot = int(H.sum(dtype=np.int64))
            GR = Gtot - GL
            HR = Htot - HL

            ok = (HL != 0) & (HR != 0)

            left  = fixed_div_vec(GL*GL, HL + self.lambda_fp)
            right = fixed_div_vec(GR*GR, HR + self.lambda_fp)
            parent = fixed_div_scalar(Gtot*Gtot, Htot + self.lambda_fp)

            gain = ((left + right - parent) // 2).astype(np.int64)
            gain[~ok] = float_to_fixed(-1e9)

            b = int(np.argmax(gain))
            g = int(gain[b])
            if g > best_gain and g > self.gamma_fp:
                best_gain = g
                best_feat = j
                best_bin  = b

        if best_feat is None:
            return self._leaf_value(idx, grad_fp, hess_fp)

        ids_best = self.bin_ids_per_feat[best_feat][idx]
        left_mask  = ids_best <= best_bin
        right_mask = ~left_mask
        left_idx  = idx[left_mask]
        right_idx = idx[right_mask]

        split_fp = int(self.bin_edges_fps[best_feat][best_bin])
        left_node  = self._fit_node(left_idx,  grad_fp, hess_fp, depth+1)
        right_node = self._fit_node(right_idx, grad_fp, hess_fp, depth+1)
        return (int(best_feat), int(split_fp), left_node, right_node)

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
                 lambda_=1.0, gamma=0.0, num_bins=128, colsample_bytree=1.0, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate_fp = float_to_fixed(learning_rate)
        self.lambda_ = lambda_
        self.gamma = gamma
        self.num_bins = num_bins
        self.colsample_bytree = colsample_bytree
        self.rng = np.random.default_rng(seed)

        self.trees = []
        self.initial_logit_fp = 0

    def _X_to_fixed(self, X: np.ndarray) -> np.ndarray:
        return np.vectorize(float_to_fixed, otypes=[np.int64])(X)

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_fp = np.array([float_to_fixed(float(v)) for v in y], dtype=np.int64)
        X_fp = self._X_to_fixed(X)

        p_fp = int(y_fp.sum(dtype=np.int64)) // y_fp.size
        self.initial_logit_fp = fixed_logit_from_p(p_fp)

        y_pred_fp = np.full(y_fp.shape[0], self.initial_logit_fp, dtype=np.int64)

        for _ in range(self.n_estimators):
            p_pred_fp = sigmoid_vec(y_pred_fp)
            grad_fp = (p_pred_fp - y_fp).astype(np.int64)
            hess_fp = (p_pred_fp * (SCALE - p_pred_fp)) // SCALE

            tree = XGBoostTreeClassifierFPFast(
                max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma,
                num_bins=self.num_bins, colsample_bytree=self.colsample_bytree,
                seed=int(self.rng.integers(1 << 31))
            )
            tree.fit(X_fp, grad_fp, hess_fp)

            update_fp = tree.predict_fp(X_fp)
            y_pred_fp -= (self.learning_rate_fp * update_fp) // SCALE

            self.trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_fp = self._X_to_fixed(X)
        y_pred_fp = np.full(X_fp.shape[0], self.initial_logit_fp, dtype=np.int64)
        for tree in self.trees:
            update_fp = tree.predict_fp(X_fp)
            y_pred_fp -= (self.learning_rate_fp * update_fp) // SCALE
        proba_fp = sigmoid_vec(y_pred_fp)
        return proba_fp.astype(np.float64) * INV_SCALE  # user-facing; math stayed int

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def save_model(self, filepath: str):
        data = {
            "initial_logit_fp": int(self.initial_logit_fp),
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate_fp * INV_SCALE),
            "lambda_": float(self.lambda_),
            "gamma": float(self.gamma),
            "num_bins": int(self.num_bins),
            "colsample_bytree": float(self.colsample_bytree),
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
        self.trees = []
        for tdata in data["trees"]:
            t = XGBoostTreeClassifierFPFast(
                max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma,
                num_bins=self.num_bins, colsample_bytree=self.colsample_bytree
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

# ============================= Dataset loaders ==============================
def load_credit_dataset():
    data = np.genfromtxt('credit_default.csv', delimiter=',', skip_header=1, filling_values=0)
    X_credit = data[:, 1:-1]
    y_credit = (data[:, -1] > 0.5).astype(int)
    return X_credit, y_credit

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
        #"covertype_bin": load_covertype_binary,
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
                            colsample_bytree=1.0, lambda_=1.0, gamma=0.0, seed=42
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

                print(f"{depth:<6} {trees:<6} | " + " ".join(f"{label:<13}" for label in accs))
                print("-" * 80)
                print(" " * 13 + " | " + " ".join(f"{np.mean(accs[k]):<13.4f}" for k in accs))
                print(" " * 13 + " | " + " ".join(f"{np.mean(times[k]):<13.2f}" for k in times))

if __name__ == "__main__":
    run_benchmark()

