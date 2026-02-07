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
def float_to_fixed(x): 
    return int(round(x * SCALE))

def fixed_to_float(x): 
    return x * INV_SCALE

def fixed_mul(x, y): 
    return (x * y) // SCALE

def fixed_div(x, y): 
    return (x * SCALE) // y if y != 0 else 0

def fixed_clip(x, min_val, max_val):
    """Fixed-point clipping"""
    return max(min_val, min(max_val, x))

# --- Fixed-point logarithm approximation ---
def fixed_log_approx(x):
    """
    Approximate ln(x) for fixed-point x using bit manipulation and polynomial approx
    Only works for x > 0. Returns ln(x) in fixed-point format.
    """
    if x <= 0:
        return float_to_fixed(-10)  # Very negative number for log(0)
    
    # Convert to float for approximation, then back to fixed
    x_float = fixed_to_float(x)
    if x_float >= 1.0:
        log_val = math.log(x_float)
    else:
        log_val = math.log(max(x_float, 1e-8))  # Avoid log(0)
    
    return float_to_fixed(log_val)

# --- Wider piecewise sigmoid ---
def wider_piecewise_sigmoid(x):
    two_scale = 2 * SCALE
    if x <= -two_scale: 
        return 0
    if x >= two_scale: 
        return SCALE
    return (x + two_scale) // 4

# --- Fixed-point sum function ---
def fixed_sum(arr):
    """Sum array of fixed-point numbers"""
    return sum(int(x) for x in arr)

# --- Fixed-point mean function ---
def fixed_mean(arr):
    """Mean of array of fixed-point numbers"""
    if len(arr) == 0:
        return 0
    return fixed_sum(arr) // len(arr)

# --- Fixed-point bincount equivalent ---
def fixed_bincount(indices, weights, minlength):
    """Fixed-point equivalent of np.bincount"""
    result = [0] * minlength
    for i, weight in zip(indices, weights):
        if 0 <= i < minlength:
            result[i] += int(weight)
    return result

# --- Fixed-point digitize ---
def fixed_digitize(x_arr, bins):
    """Fixed-point equivalent of np.digitize"""
    result = []
    for x in x_arr:
        bin_idx = 0
        for i, bin_edge in enumerate(bins[:-1]):
            if x > bin_edge:
                bin_idx = i + 1
            else:
                break
        result.append(bin_idx)
    return result

# --- Fixed-point linspace ---
def fixed_linspace(start, stop, num):
    """Fixed-point equivalent of np.linspace"""
    if num <= 1:
        return [start]
    
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

# --- Core fixed-point boosting model ---
class XGBoostTreeClassifier:
    def __init__(self, max_depth=3, lambda_=1, gamma=0, colsample_bytree=1.0, num_bins=64):
        self.max_depth = max_depth
        self.lambda_fixed = float_to_fixed(lambda_)
        self.gamma_fixed = float_to_fixed(gamma)
        self.colsample_bytree = colsample_bytree
        self.num_bins = num_bins
        self.tree = None
        self.training_metadata = []

    def fit(self, X, grad, hess, depth=0, node_metadata=None):
        if node_metadata is None:
            node_metadata = {}

        if depth >= self.max_depth or len(X) < 2:
            # Fixed-point leaf value calculation
            grad_sum = fixed_sum(grad)
            hess_sum = fixed_sum(hess)
            leaf_value = fixed_div(grad_sum, hess_sum + self.lambda_fixed)
            leaf_value = fixed_clip(leaf_value, float_to_fixed(-1), float_to_fixed(1))
            node_metadata["leaf_value"] = fixed_to_float(leaf_value)
            self.training_metadata.append(node_metadata)
            return leaf_value

        best_gain = float_to_fixed(-1000)  # Very negative number
        best_feat, best_split = None, None
        best_bin_edges = None

        n_features = X.shape[1]
        features = np.random.choice(n_features, max(1, int(n_features * self.colsample_bytree)), replace=False)
        node_metadata["features_considered"] = features.tolist()
        node_metadata["histograms"] = {}

        for j in features:
            x = X[:, j]
            if np.min(x) == np.max(x): 
                continue
                
            # Use fixed-point linspace
            bin_edges = fixed_linspace(np.min(x), np.max(x), self.num_bins + 1)
            bin_ids = fixed_digitize(x, bin_edges[:-1])
            
            # Fixed-point histogram accumulation
            G = fixed_bincount(bin_ids, grad, self.num_bins + 1)
            H = fixed_bincount(bin_ids, hess, self.num_bins + 1)

            node_metadata["histograms"][str(j)] = {
                "bin_edges": [float(b) for b in bin_edges],
                "grad_hist": [fixed_to_float(g) for g in G],
                "hess_hist": [fixed_to_float(h) for h in H]
            }

            G_L = H_L = 0
            G_total, H_total = fixed_sum(G), fixed_sum(H)
            
            for b in range(1, self.num_bins):
                G_L += G[b - 1]
                H_L += H[b - 1]
                G_R = G_total - G_L
                H_R = H_total - H_L
                
                if H_L == 0 or H_R == 0: 
                    continue
                
                # FIXED-POINT GAIN CALCULATION - This was the main issue!
                # gain = 0.5 * (G_L²/(H_L + λ) + G_R²/(H_R + λ)) - 0.5 * (G_total²/(H_total + λ))
                
                # Left child gain: G_L² / (H_L + λ)
                gain_left = fixed_div(fixed_mul(G_L, G_L), H_L + self.lambda_fixed)
                
                # Right child gain: G_R² / (H_R + λ) 
                gain_right = fixed_div(fixed_mul(G_R, G_R), H_R + self.lambda_fixed)
                
                # Parent gain: G_total² / (H_total + λ)
                gain_parent = fixed_div(fixed_mul(G_total, G_total), H_total + self.lambda_fixed)
                
                # Total gain = 0.5 * (left + right - parent)
                gain = (gain_left + gain_right - gain_parent) // 2
                
                if gain > best_gain and gain > self.gamma_fixed:
                    best_gain = gain
                    best_feat, best_split = j, b
                    best_bin_edges = bin_edges

        if best_feat is None:
            # Fixed-point leaf value calculation
            grad_sum = fixed_sum(grad)
            hess_sum = fixed_sum(hess)
            leaf_value = fixed_div(grad_sum, hess_sum + self.lambda_fixed)
            leaf_value = fixed_clip(leaf_value, float_to_fixed(-1), float_to_fixed(1))
            node_metadata["leaf_value"] = fixed_to_float(leaf_value)
            self.training_metadata.append(node_metadata)
            return leaf_value

        split_val = best_bin_edges[best_split]
        node_metadata["best_feature"] = int(best_feat)
        node_metadata["best_split_val"] = float(split_val)
        self.training_metadata.append(node_metadata)

        left = X[:, best_feat] <= split_val
        right = ~left
        left_tree = self.fit(X[left], [grad[i] for i in range(len(grad)) if left[i]], 
                           [hess[i] for i in range(len(hess)) if left[i]], depth + 1)
        right_tree = self.fit(X[right], [grad[i] for i in range(len(grad)) if right[i]], 
                            [hess[i] for i in range(len(hess)) if right[i]], depth + 1)
        return (best_feat, split_val, left_tree, right_tree)

    def predict_row(self, x, node):
        if not isinstance(node, tuple): 
            return node  # Return fixed-point value
        feat, split, l, r = node
        return self.predict_row(x, l) if x[feat] <= split else self.predict_row(x, r)

    def predict(self, X):
        # Returns fixed-point predictions
        return [self.predict_row(x, self.tree) for x in X]


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
        # Convert targets to fixed-point
        y_fixed = [float_to_fixed(float(yi)) for yi in y]
        
        # Fixed-point initial logit calculation
        y_mean_fixed = fixed_mean(y_fixed)
        
        # For initial logit: log(p / (1-p)) where p = mean(y)
        # Avoid division by zero and ensure p is in (0,1)
        p_fixed = fixed_clip(y_mean_fixed, float_to_fixed(0.001), float_to_fixed(0.999))
        one_minus_p = SCALE - p_fixed
        
        # log(p / (1-p)) = log(p) - log(1-p)
        log_p = fixed_log_approx(p_fixed)
        log_one_minus_p = fixed_log_approx(one_minus_p)
        self.initial_logit = log_p - log_one_minus_p
        
        # Initialize predictions with initial logit
        y_pred = [self.initial_logit] * len(y_fixed)

        for iteration in range(self.n_estimators):
            # Convert logits to probabilities using fixed-point sigmoid
            p_pred = [self.sigmoid_fn(val) for val in y_pred]
            
            # Fixed-point gradient and hessian calculation
            grad = [p_pred[i] - y_fixed[i] for i in range(len(y_fixed))]
            hess = [fixed_mul(p_pred[i], SCALE - p_pred[i]) for i in range(len(y_fixed))]
            
            # Fit tree to gradients and hessians
            tree = XGBoostTreeClassifier(max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma)
            tree.tree = tree.fit(X, grad, hess)
            
            # Get tree predictions (already in fixed-point)
            update = tree.predict(X)
            
            # Update predictions: y_pred -= learning_rate * update
            for i in range(len(y_pred)):
                y_pred[i] -= fixed_mul(self.scaled_lr, update[i])
            
            self.trees.append(tree)

    def predict_proba(self, X):
        # Initialize with initial logit
        y_pred = [self.initial_logit] * X.shape[0]
        
        # Apply all trees
        for tree in self.trees:
            update = tree.predict(X)
            for i in range(len(y_pred)):
                y_pred[i] -= fixed_mul(self.scaled_lr, update[i])
        
        # Convert to probabilities using sigmoid
        return [self.sigmoid_fn(val) for val in y_pred]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [(p >= SCALE // 2) for p in proba]

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
        # "credit_card_default": load_credit_dataset(),  # Commented out since file might not exist
    }
    depths = [4, 5]
    trees_list = [50, 100]
    num_runs = 3

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
                        pred = model.predict(Xte)
                        
                        # Convert predictions to numpy arrays for comparison
                        if label == "Fixed-Wider":
                            pred = np.array([int(p) for p in pred])
                        
                        acc = np.mean(pred == yte)
                        accs[label].append(acc)
                        times[label].append(time.time() - t0)

                        if label == "Fixed-Wider" and run == 0:
                            import os
                            os.makedirs("models", exist_ok=True)
                            model.save_model(f"models/model_{name}_depth{depth}_trees{trees}.json")

                print(f"{depth:<6} {trees:<6} | " + " ".join(f"{label:<13}" for label in accs))
                print("-" * 80)
                print(" " * 13 + " | " + " ".join(f"{np.mean(accs[k]):<13.4f}" for k in accs))
                print(" " * 13 + " | " + " ".join(f"{np.mean(times[k]):<13.2f}" for k in times))

if __name__ == "__main__":
    run_benchmark()
