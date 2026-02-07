import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons, make_circles, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from decimal import Decimal, getcontext
import xgboost as xgb
import time
from tqdm import tqdm

# --- Stable sigmoid ---
def stable_sigmoid(x):
    x = np.array(x)
    out = np.empty_like(x)
    positive = x >= 0
    negative = ~positive
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1 + exp_x)
    return out

# --- Tree for boosting (optimized histogram binning) ---
class XGBoostTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=10, lambda_=1, gamma=0, colsample_bytree=1.0, num_bins=64):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ = lambda_
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.num_bins = num_bins
        self.tree = None
        self.selected_features = None

    def fit(self, X, grad, hess, depth=0):
        n_samples, n_features = X.shape
        n_selected = max(1, int(self.colsample_bytree * n_features))
        self.selected_features = np.random.choice(n_features, n_selected, replace=False)
        return self._fit(X, grad, hess, depth)

    def _fit(self, X, grad, hess, depth):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            value = np.sum(grad) / (np.sum(hess) + self.lambda_)
            return np.clip(value, -10, 10)

        best_gain = -np.inf
        best_feature = None
        best_split_val = None

        for feature in self.selected_features:
            x = X[:, feature]
            x_min, x_max = np.min(x), np.max(x)
            if x_max == x_min:
                continue

            bin_edges = np.linspace(x_min, x_max, self.num_bins + 1)
            bin_ids = np.digitize(x, bin_edges[:-1], right=False)

            G_hist = np.bincount(bin_ids, weights=grad, minlength=self.num_bins + 2)
            H_hist = np.bincount(bin_ids, weights=hess, minlength=self.num_bins + 2)

            G_L, H_L = 0.0, 0.0
            G_total, H_total = np.sum(G_hist), np.sum(H_hist)

            for b in range(1, self.num_bins):
                G_L += G_hist[b]
                H_L += H_hist[b]
                G_R = G_total - G_L
                H_R = H_total - H_L

                if H_L == 0 or H_R == 0:
                    continue

                gain = 0.5 * (G_L**2 / (H_L + self.lambda_) + G_R**2 / (H_R + self.lambda_))
                gain -= 0.5 * (G_total**2 / (H_total + self.lambda_))

                if gain > best_gain and gain > self.gamma:
                    best_gain = gain
                    best_feature = feature
                    best_split_val = bin_edges[b]

        if best_gain == -np.inf:
            value = np.sum(grad) / (np.sum(hess) + self.lambda_)
            return np.clip(value, -10, 10)

        left = X[:, best_feature] <= best_split_val
        right = ~left

        left_tree = self._fit(X[left], grad[left], hess[left], depth + 1)
        right_tree = self._fit(X[right], grad[right], hess[right], depth + 1)
        return (best_feature, best_split_val, left_tree, right_tree)

    def predict_row(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, split, left, right = node
        return self.predict_row(x, left) if x[feature] <= split else self.predict_row(x, right)

    def predict(self, X):
        return np.array([self.predict_row(x, self.tree) for x in X])

# --- Taylor sigmoid approximation with clamping and stability ---
def taylor_exp(x, terms):
    result = Decimal(1)
    term = Decimal(1)
    for i in range(1, terms + 1):
        term *= x / Decimal(i)
        result += term
    return result

def taylor_sigmoid(x, terms):
    getcontext().prec = 15
    x = Decimal(x)
    x = max(min(x, Decimal(5)), Decimal(-5))
    if x >= 0:
        exp_neg_x = taylor_exp(-x, terms)
        result = Decimal(1) / (Decimal(1) + exp_neg_x)
    else:
        exp_x = taylor_exp(x, terms)
        result = exp_x / (Decimal(1) + exp_x)
    return float(max(Decimal("0.0001"), min(Decimal("0.9999"), result)))

# --- Boosted Classifier (floating point) ---
class SimpleXGBoostClassifier:
    def __init__(self, n_estimators=50, max_depth=3, learning_rate=0.1, lambda_=1, gamma=0, subsample=0.8, colsample_bytree=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []
        self.initial_logit = 0

    def sigmoid(self, x):
        return stable_sigmoid(x)

    def fit(self, X, y):
        eps = 1e-6
        y_mean = np.mean(y)
        self.initial_logit = np.log(y_mean / (1 - y_mean))
        y_pred = np.full_like(y, fill_value=self.initial_logit, dtype=float)
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            n_subsample = max(1, int(self.subsample * n_samples))
            indices = np.random.choice(n_samples, n_subsample, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]

            p = self.sigmoid(y_pred[indices])
            p = np.clip(p, eps, 1 - eps)
            grad = p - y_sub
            hess = p * (1 - p)

            tree = XGBoostTreeClassifier(
                max_depth=self.max_depth,
                lambda_=self.lambda_,
                gamma=self.gamma,
                colsample_bytree=self.colsample_bytree
            )
            tree.tree = tree.fit(X_sub, grad, hess)
            update = tree.predict(X)
            y_pred -= self.learning_rate * update
            self.trees.append(tree)

    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], fill_value=self.initial_logit)
        for tree in self.trees:
            y_pred -= self.learning_rate * tree.predict(X)
        return self.sigmoid(y_pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# --- Fixed-point XGBoost ---
class FixedPointXGBoostClassifier(SimpleXGBoostClassifier):
    def __init__(self, terms=10, **kwargs):
        super().__init__(**kwargs)
        self.terms = terms

    def sigmoid(self, x):
        return np.array([taylor_sigmoid(val, self.terms) for val in x])

# --- Dataset setup ---
datasets = {
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "moons": make_moons(n_samples=1000, noise=0.3, random_state=42),
    "circles": make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42),
    "synthetic": make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42),
    "wine_binary": (lambda: (lambda X, y: (X, (y == 0).astype(int)))(*load_wine(return_X_y=True)))()
}

# --- Fixed Parameters ---
depths = [4, 5]
trees_list = [50,100]
taylor_terms_list = [10]
num_runs = 1

# --- Run experiments ---
for name, (X, y) in tqdm(datasets.items(), desc="Datasets", leave=True):
    print(f"\n=== Dataset: {name} ===")
    X = StandardScaler().fit_transform(X)
    results = []

    for terms, depth, trees in tqdm(
    [(t, d, tr) for t in taylor_terms_list for d in depths for tr in trees_list],
    desc=f"{name} grid", total=len(taylor_terms_list) * len(depths) * len(trees_list)):
                simple_accs, fixed_accs, xgb_accs = [], [], []
                simple_times, fixed_times, xgb_times = [], [], []

                for run in range(num_runs):
                    np.random.seed(42 + run)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + run)

                    # --- Simple model ---
                    simple_model = SimpleXGBoostClassifier(n_estimators=trees, max_depth=depth)
                    t0 = time.time()
                    simple_model.fit(X_train, y_train)
                    simple_accs.append(np.mean(simple_model.predict(X_test) == y_test))
                    simple_times.append(time.time() - t0)

                    # --- Fixed-point model ---
                    fixed_model = FixedPointXGBoostClassifier(terms=terms, n_estimators=trees, max_depth=depth)
                    t1 = time.time()
                    fixed_model.fit(X_train, y_train)
                    fixed_accs.append(np.mean(fixed_model.predict(X_test) == y_test))
                    fixed_times.append(time.time() - t1)

                    # --- Real XGBoost model ---
                    xgb_model = xgb.XGBClassifier(n_estimators=trees, max_depth=depth, verbosity=0, eval_metric="logloss")
                    t2 = time.time()
                    xgb_model.fit(X_train, y_train)
                    xgb_accs.append(np.mean(xgb_model.predict(X_test) == y_test))
                    xgb_times.append(time.time() - t2)

                results.append((
                    terms, depth, trees,
                    np.mean(simple_accs), np.mean(fixed_accs), np.mean(xgb_accs),
                    np.mean(simple_times), np.mean(fixed_times), np.mean(xgb_times)
                ))

    print(f"{'Terms':<7} {'Depth':<6} {'Trees':<6} | {'Simple Acc':<11} {'Fixed Acc':<11} {'XGB Acc':<10} | {'Simple Time':<12} {'Fixed Time':<12} {'XGB Time'}")
    print("-" * 100)
    for r in results:
        print(f"{r[0]:<7} {r[1]:<6} {r[2]:<6} | {r[3]:<11.4f} {r[4]:<11.4f} {r[5]:<10.4f} | {r[6]:<12.2f} {r[7]:<12.2f} {r[8]:.2f}")

