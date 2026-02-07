import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
from tqdm import tqdm
import math
import sys
import json

SCALE = 100000
INV_SCALE = 1 / SCALE

# --- Fixed-point helpers ---
def float_to_fixed(x): return int(round(x * SCALE))
def fixed_to_float(x): return x * INV_SCALE
def fixed_mul(x, y): return (x * y) // SCALE
def fixed_div(x, y): return (x * SCALE) // y if y != 0 else 0

# --- Wider piecewise sigmoid ---
def wider_piecewise_sigmoid(x):
    two_scale = 2 * SCALE
    if x <= -two_scale: return 0
    if x >= two_scale: return SCALE
    return (x + two_scale) // 4

# --- Core fixed-point boosting model ---
class XGBoostTreeClassifier:
    def __init__(self, max_depth=3, lambda_=1, gamma=0, colsample_bytree=1.0, num_bins=64):
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.num_bins = num_bins
        self.tree = None
        self.training_metadata = []

    def fit(self, X, grad, hess, depth=0, node_metadata=None):
        if node_metadata is None:
            node_metadata = {}

        if depth >= self.max_depth or len(X) < 2:
            node_metadata["leaf_value"] = float(np.clip(np.sum(grad) / (np.sum(hess) + self.lambda_), -1, 1))
            self.training_metadata.append(node_metadata)
            return node_metadata["leaf_value"]

        best_gain = -float('inf')
        best_feat, best_split = None, None
        best_bin_edges = None

        n_features = X.shape[1]
        features = np.random.choice(n_features, max(1, int(n_features * self.colsample_bytree)), replace=False)
        node_metadata["features_considered"] = features.tolist()
        node_metadata["histograms"] = {}

        for j in features:
            x = X[:, j]
            if np.min(x) == np.max(x): continue
            bin_edges = np.linspace(np.min(x), np.max(x), self.num_bins + 1)
            bin_ids = np.digitize(x, bin_edges[:-1])
            G = np.bincount(bin_ids, weights=grad, minlength=self.num_bins + 1)
            H = np.bincount(bin_ids, weights=hess, minlength=self.num_bins + 1)

            node_metadata["histograms"][str(j)] = {
                "bin_edges": bin_edges.tolist(),
                "grad_hist": G.tolist(),
                "hess_hist": H.tolist()
            }

            G_L = H_L = 0.0
            G_total, H_total = np.sum(G), np.sum(H)
            for b in range(1, self.num_bins):
                G_L += G[b - 1]
                H_L += H[b - 1]
                G_R = G_total - G_L
                H_R = H_total - H_L
                if H_L == 0 or H_R == 0: continue
                # the gain below needs to be computed in fixed point    
                gain = 0.5 * (G_L**2 / (H_L + self.lambda_) + G_R**2 / (H_R + self.lambda_)) - 0.5 * (G_total**2 / (H_total + self.lambda_))
                if gain > best_gain and gain > self.gamma:
                    best_gain = gain
                    best_feat, best_split = j, b
                    best_bin_edges = bin_edges

        if best_feat is None:
            node_metadata["leaf_value"] = float(np.clip(np.sum(grad) / (np.sum(hess) + self.lambda_), -1, 1))
            self.training_metadata.append(node_metadata)
            return node_metadata["leaf_value"]

        split_val = best_bin_edges[best_split]
        node_metadata["best_feature"] = int(best_feat)
        node_metadata["best_split_val"] = float(split_val)
        self.training_metadata.append(node_metadata)

        left = X[:, best_feat] <= split_val
        right = ~left
        left_tree = self.fit(X[left], grad[left], hess[left], depth + 1)
        right_tree = self.fit(X[right], grad[right], hess[right], depth + 1)
        return (best_feat, split_val, left_tree, right_tree)

    def predict_row(self, x, node):
        if not isinstance(node, tuple): return node
        feat, split, l, r = node
        return self.predict_row(x, l) if x[feat] <= split else self.predict_row(x, r)

    def predict(self, X):
        return np.array([self.predict_row(x, self.tree) for x in X])


# --- Helper functions for JSON serialization ---
def tuple_to_list(obj):
    import numpy as np
    if isinstance(obj, tuple):
        return ["__tuple__"] + [tuple_to_list(item) for item in obj]
    elif isinstance(obj, list):
        return [tuple_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
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

class GenericFixedPointXGB:
    def __init__(self, sigmoid_fn, n_estimators=50, max_depth=3, learning_rate=0.1, lambda_=1, gamma=0):
        self.sigmoid_fn = sigmoid_fn
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.gamma = gamma
        self.scaled_lr = float_to_fixed(learning_rate)
        self.trees = []
        self.initial_logit = 0

    def fit(self, X, y):
        y_fixed = np.array([float_to_fixed(yi) for yi in y], dtype=int)
        y_mean = np.mean(y_fixed)
        p = y_mean / SCALE
        self.initial_logit = float_to_fixed(np.log(p / (1 - p)))
        y_pred = np.full_like(y_fixed, self.initial_logit)

        for _ in range(self.n_estimators):
            p_pred = np.array([self.sigmoid_fn(val) for val in y_pred], dtype=int)
            grad = p_pred - y_fixed
            hess = (p_pred * (SCALE - p_pred)) // SCALE
            tree = XGBoostTreeClassifier(max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma)
            tree.tree = tree.fit(X, grad, hess)
            update = tree.predict(X)
            for i in range(len(y_pred)):
                y_pred[i] -= fixed_mul(self.scaled_lr, float_to_fixed(update[i]))
            self.trees.append(tree)

    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], self.initial_logit)
        for tree in self.trees:
            update = tree.predict(X)
            for i in range(len(y_pred)):
                y_pred[i] -= fixed_mul(self.scaled_lr, float_to_fixed(update[i]))
        return np.array([self.sigmoid_fn(val) for val in y_pred], dtype=int)

    def predict(self, X):
        return (self.predict_proba(X) >= SCALE // 2).astype(int)

    def save_model(self, filepath):
        model_data = {
            "initial_logit": self.initial_logit,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "lambda_": self.lambda_,
            "gamma": self.gamma,
            "trees": [tuple_to_list(tree.tree) for tree in self.trees]
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.initial_logit = model_data["initial_logit"]
        self.n_estimators = model_data["n_estimators"]
        self.max_depth = model_data["max_depth"]
        self.learning_rate = model_data["learning_rate"]
        self.lambda_ = model_data["lambda_"]
        self.gamma = model_data["gamma"]
        self.scaled_lr = float_to_fixed(self.learning_rate)
        self.trees = []
        for tree_data in model_data["trees"]:
            tree = XGBoostTreeClassifier(max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma)
            tree.tree = list_to_tuple(tree_data)
            self.trees.append(tree)

# --- Classifiers ---
FixedWider = lambda **kw: GenericFixedPointXGB(wider_piecewise_sigmoid, **kw)

# --- Credit dataset loader ---
def load_credit_dataset():
    data = np.genfromtxt('credit_default.csv', delimiter=',', skip_header=1, filling_values=0)
    X_credit = data[:, 1:-1]
    y_credit = (data[:, -1] > 0.5).astype(int)
    return X_credit, y_credit

# --- Benchmark ---
def run_benchmark():
    datasets = {
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "credit_card_default": load_credit_dataset(),
    }
    depths = [3, 4]
    trees_list = [5, 10]
    num_runs = 2

    for name, (X, y) in tqdm(datasets.items(), desc="Datasets", file=sys.stdout):
        print(f"\n=== Dataset: {name} ===")
        X = StandardScaler().fit_transform(X)

        for depth in depths:
            for trees in trees_list:
                accs = {"Fixed-Wider": [], "XGB": []}
                times = {k: [] for k in accs}

                for run in range(num_runs):
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42+run)

                    models = {
                        "Fixed-Wider": FixedWider(n_estimators=trees, max_depth=depth),
                        "XGB": xgb.XGBClassifier(n_estimators=trees, max_depth=depth, verbosity=0, eval_metric="logloss"),
                    }

                    for label, model in models.items():
                        t0 = time.time()
                        model.fit(Xtr, ytr)
                        acc = np.mean(model.predict(Xte) == yte)
                        accs[label].append(acc)
                        times[label].append(time.time() - t0)

                        if label == "Fixed-Wider" and run == 0:
                            model.save_model(f"models/model_{name}_depth{depth}_trees{trees}.json")

                print(f"{depth:<6} {trees:<6} | " + " ".join(f"{label:<13}" for label in accs))
                print("-" * 80)
                print(" " * 13 + " | " + " ".join(f"{np.mean(accs[k]):<13.4f}" for k in accs))
                print(" " * 13 + " | " + " ".join(f"{np.mean(times[k]):<13.2f}" for k in times))

run_benchmark()

