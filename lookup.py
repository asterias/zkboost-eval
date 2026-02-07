import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
from tqdm import tqdm
import math

# --- Fixed-point configuration (5-digit precision) ---
SCALE = 100000
INV_SCALE = 1 / SCALE

def float_to_fixed(x):
    return int(round(x * SCALE))

def fixed_to_float(x_fixed):
    return x_fixed * INV_SCALE

def fixed_mul(x_fixed, y_fixed):
    return (x_fixed * y_fixed) // SCALE

def fixed_div(x_fixed, y_fixed):
    return (x_fixed * SCALE) // y_fixed if y_fixed != 0 else 0

def fixed_exp(x_fixed, terms=10):
    result = SCALE
    term = SCALE
    for i in range(1, terms + 1):
        term = fixed_mul(term, fixed_div(x_fixed, float_to_fixed(i)))
        result += term
    return result

def fixed_sigmoid(x_fixed, terms=10):
    CLAMP_MIN = float_to_fixed(-5)
    CLAMP_MAX = float_to_fixed(5)
    x_fixed = max(CLAMP_MIN, min(CLAMP_MAX, x_fixed))
    if x_fixed >= 0:
        exp_neg = fixed_exp(-x_fixed, terms)
        denom = SCALE + exp_neg
        return fixed_div(SCALE, denom)
    else:
        exp_x = fixed_exp(x_fixed, terms)
        denom = SCALE + exp_x
        return fixed_div(exp_x, denom)

def stable_sigmoid(x):
    x = np.array(x)
    out = np.empty_like(x)
    positive = x >= 0
    negative = ~positive
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1 + exp_x)
    return out

# Create the LUT once
SIGMOID_LUT = {}
SIGMOID_LUT_RESOLUTION = 1000  # i.e., 0.001 step in float space
SIGMOID_LUT_CLAMP_MIN = -500000  # fixed-point -5.0
SIGMOID_LUT_CLAMP_MAX = 500000   # fixed-point 5.0

def init_sigmoid_lut():
    for x_fixed in range(SIGMOID_LUT_CLAMP_MIN, SIGMOID_LUT_CLAMP_MAX + 1, SIGMOID_LUT_RESOLUTION):
        x = x_fixed / SCALE
        sig = int(round((1 / (1 + math.exp(-x))) * SCALE))
        SIGMOID_LUT[x_fixed] = sig

def lut_fixed_sigmoid(x_fixed):
    x_fixed = max(SIGMOID_LUT_CLAMP_MIN, min(SIGMOID_LUT_CLAMP_MAX, x_fixed))
    key = SIGMOID_LUT_RESOLUTION * round(x_fixed / SIGMOID_LUT_RESOLUTION)
    return SIGMOID_LUT.get(key, SCALE // 2)  # fallback to 0.5

# Call once before training
init_sigmoid_lut()

# --- Tree for boosting (histogram binning) ---
class XGBoostTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=10, lambda_=1, gamma=0,
                 colsample_bytree=1.0, num_bins=64):
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
            return np.clip(np.sum(grad) / (np.sum(hess) + self.lambda_), -10, 10)

        best_gain = -np.inf
        best_feature, best_split_val = None, None

        for feature in self.selected_features:
            x = X[:, feature]
            if np.min(x) == np.max(x):
                continue
            bin_edges = np.linspace(np.min(x), np.max(x), self.num_bins + 1)
            bin_ids = np.digitize(x, bin_edges[:-1], right=False)

            G_hist = np.bincount(bin_ids, weights=grad, minlength=self.num_bins + 2)
            H_hist = np.bincount(bin_ids, weights=hess, minlength=self.num_bins + 2)

            G_L, H_L = 0.0, 0.0
            G_total, H_total = np.sum(G_hist), np.sum(H_hist)

            for b in range(1, self.num_bins):
                G_L += G_hist[b]
                H_L += H_hist[b]
                G_R, H_R = G_total - G_L, H_total - H_L
                if H_L == 0 or H_R == 0:
                    continue
                gain = 0.5 * (G_L**2 / (H_L + self.lambda_) + G_R**2 / (H_R + self.lambda_)) \
                       - 0.5 * (G_total**2 / (H_total + self.lambda_))
                if gain > best_gain and gain > self.gamma:
                    best_gain = gain
                    best_feature = feature
                    best_split_val = bin_edges[b]

        if best_feature is None:
            return np.clip(np.sum(grad) / (np.sum(hess) + self.lambda_), -10, 10)

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


# --- Float-based boosted classifier ---
class SimpleXGBoostClassifier:
    def __init__(self, n_estimators=50, max_depth=3, learning_rate=0.1,
                 lambda_=1, gamma=0, subsample=0.8, colsample_bytree=0.8):
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

        for _ in range(self.n_estimators):
            idx = np.random.choice(len(X), int(len(X) * self.subsample), replace=False)
            p = self.sigmoid(y_pred[idx])
            p = np.clip(p, eps, 1 - eps)
            grad = p - y[idx]
            hess = p * (1 - p)
            tree = XGBoostTreeClassifier(max_depth=self.max_depth, lambda_=self.lambda_,
                                         gamma=self.gamma, colsample_bytree=self.colsample_bytree)
            tree.tree = tree.fit(X[idx], grad, hess)
            y_pred -= self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict_proba(self, X):
        y_pred = np.full(X.shape[0], self.initial_logit)
        for tree in self.trees:
            y_pred -= self.learning_rate * tree.predict(X)
        return self.sigmoid(y_pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# --- Fixed-point classifier using integer arithmetic ---
class FixedPointXGBoostClassifier(SimpleXGBoostClassifier):
    def __init__(self, terms=10, **kwargs):
        super().__init__(**kwargs)
        self.terms = terms
        self.scaled_lr = float_to_fixed(self.learning_rate)

    def sigmoid(self, x):
        return np.array([lut_fixed_sigmoid(val) for val in x], dtype=int)


    def fit(self, X, y):
        y_fixed = np.array([float_to_fixed(yi) for yi in y], dtype=int)
        y_mean = np.mean(y_fixed)
        p = y_mean / SCALE
        self.initial_logit = float_to_fixed(np.log(p / (1 - p)))
        y_pred = np.full_like(y_fixed, self.initial_logit)

        for _ in range(self.n_estimators):
            idx = np.random.choice(len(X), int(len(X) * self.subsample), replace=False)
            p_pred = self.sigmoid(y_pred[idx])
            grad = p_pred - y_fixed[idx]
            hess = (p_pred * (SCALE - p_pred)) // SCALE
            tree = XGBoostTreeClassifier(max_depth=self.max_depth, lambda_=self.lambda_,
                                         gamma=self.gamma, colsample_bytree=self.colsample_bytree)
            tree.tree = tree.fit(X[idx], grad, hess)
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
        return self.sigmoid(y_pred)

    def predict(self, X):
        return (self.predict_proba(X) >= SCALE // 2).astype(int)

# --- Dataset setup ---
def load_credit_dataset():
    data = np.genfromtxt('credit_default.csv', delimiter=',', skip_header=1, filling_values=0)
    X_credit = data[:, 1:-1]  # exclude ID and target
    y_credit = data[:, -1].astype(int)
    y_credit = np.where(y_credit > 0.5, 1, 0).astype(int)  # force binary
    return X_credit, y_credit

# Only using credit card and breast cancer datasets
datasets = {
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "credit_card_default": load_credit_dataset()
}

# --- Benchmark parameters ---
depths = [4,5]
trees_list = [50,100]
taylor_terms_list = [10]
num_runs = 5

# --- Run experiments ---
for name, (X, y) in tqdm(datasets.items(), desc="Datasets", leave=True):
    print(f"\n=== Dataset: {name} ===")
    X = StandardScaler().fit_transform(X)
    results = []

    for terms, depth, trees in tqdm(
        [(t, d, tr) for t in taylor_terms_list for d in depths for tr in trees_list],
        desc=f"{name} grid", total=len(taylor_terms_list) * len(depths) * len(trees_list)
    ):
        simple_accs, fixed_accs, xgb_accs = [], [], []
        simple_times, fixed_times, xgb_times = [], [], []

        for run in range(num_runs):
            np.random.seed(42 + run)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + run)

            simple_model = SimpleXGBoostClassifier(n_estimators=trees, max_depth=depth)
            t0 = time.time()
            simple_model.fit(X_train, y_train)
            simple_accs.append(np.mean(simple_model.predict(X_test) == y_test))
            simple_times.append(time.time() - t0)

            fixed_model = FixedPointXGBoostClassifier(terms=terms, n_estimators=trees, max_depth=depth)
            t1 = time.time()
            fixed_model.fit(X_train, y_train)
            fixed_accs.append(np.mean(fixed_model.predict(X_test) == y_test))
            fixed_times.append(time.time() - t1)

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

