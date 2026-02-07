import numpy as np, math, time
import matplotlib.pyplot as plt          # (kept; not used in benchmark)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
# Fixed-point helpers
# ──────────────────────────────────────────────────────────
SCALE = 100_000          # 5-digit precision
INV_SCALE = 1 / SCALE

def float_to_fixed(x):     return int(round(x * SCALE))
def fixed_to_float(xf):    return xf * INV_SCALE
def fixed_mul(xf, yf):     return (xf * yf) // SCALE
def fixed_div(xf, yf):     return (xf * SCALE) // yf if yf else 0

# integer exp / logistic (for reference)
def fixed_exp(xf, terms=10):
    res = SCALE
    term = SCALE
    for i in range(1, terms + 1):
        term = fixed_mul(term, fixed_div(xf, float_to_fixed(i)))
        res += term
    return res

def fixed_sigmoid(xf, terms=10):
    lo, hi = float_to_fixed(-5), float_to_fixed(5)
    xf = max(lo, min(hi, xf))
    if xf >= 0:
        e = fixed_exp(-xf, terms)
        return fixed_div(SCALE, SCALE + e)
    else:
        e = fixed_exp(xf, terms)
        return fixed_div(e, SCALE + e)

# stable float sigmoid
def stable_sigmoid(x):
    x = np.asarray(x)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos]  = 1 / (1 + np.exp(-x[pos]))
    expx      = np.exp(x[~pos])
    out[~pos] = expx / (1 + expx)
    return out

# ──────────────────────────────────────────────────────────
# Integer sigmoid variants
# ──────────────────────────────────────────────────────────
# LUT-based (your original)
SIGMOID_LUT = {}
SIGMOID_LUT_RESOLUTION = 100  # 0.001 granularity (vs 0.01 before)
SIGMOID_LUT_CLAMP_MIN = -500_000
SIGMOID_LUT_CLAMP_MAX = 500_000

def init_sigmoid_lut():
    for xf in range(SIGMOID_LUT_CLAMP_MIN, SIGMOID_LUT_CLAMP_MAX + 1, SIGMOID_LUT_RESOLUTION):
        x = xf / SCALE
        SIGMOID_LUT[xf] = int(round((1 / (1 + math.exp(-x))) * SCALE))

def lut_fixed_sigmoid(xf):
    xf = max(SIGMOID_LUT_CLAMP_MIN, min(SIGMOID_LUT_CLAMP_MAX, xf))
    base = SIGMOID_LUT_RESOLUTION * (xf // SIGMOID_LUT_RESOLUTION)
    rem = xf - base
    s0 = SIGMOID_LUT.get(base, SCALE // 2)
    s1 = SIGMOID_LUT.get(base + SIGMOID_LUT_RESOLUTION, SCALE // 2)
    return s0 + ((s1 - s0) * rem) // SIGMOID_LUT_RESOLUTION

# Call once before training
init_sigmoid_lut()


# NEW wider piece-wise: linear on [-2,2], saturates outside
def wider_piecewise_sigmoid(xf: int) -> int:
    """Improved piecewise: linear in [-2, 2], smoothly clipped"""
    two = 2 * SCALE
    if xf <= -two: return 0
    if xf >= two: return SCALE
    return SCALE * (xf + two) // (4 * SCALE)

# ──────────────────────────────────────────────────────────
# Histogram-tree used by all boosted models (unchanged)
# ──────────────────────────────────────────────────────────
class XGBoostTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=10,
                 lambda_=1, gamma=0, colsample_bytree=1.0, num_bins=64):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_  = lambda_
        self.gamma    = gamma
        self.colsample_bytree = colsample_bytree
        self.num_bins = num_bins
        self.tree     = None

    # --- recursive training ---
    def fit(self, X, grad, hess, depth=0):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return np.clip(np.sum(grad) / (np.sum(hess) + self.lambda_), -10, 10)

        n_features = X.shape[1]
        feats = np.random.choice(n_features,
                                 max(1, int(self.colsample_bytree*n_features)),
                                 replace=False)
        best_gain = -np.inf
        best_feat, best_split = None, None

        for f in feats:
            col = X[:,f]
            if col.min()==col.max(): continue
            bins = np.linspace(col.min(), col.max(), self.num_bins+1)
            ids  = np.digitize(col, bins[:-1], right=False)

            G = np.bincount(ids, weights=grad, minlength=self.num_bins+2)
            H = np.bincount(ids, weights=hess, minlength=self.num_bins+2)
            G_L=H_L=0.0
            Gtot,Htot = G.sum(), H.sum()

            for b in range(1,self.num_bins):
                G_L += G[b];  H_L += H[b]
                G_R, H_R = Gtot-G_L, Htot-H_L
                if H_L==0 or H_R==0: continue
                gain = 0.5*((G_L**2)/(H_L+self.lambda_) +
                             (G_R**2)/(H_R+self.lambda_)) \
                       -0.5*(Gtot**2)/(Htot+self.lambda_)
                if gain > best_gain and gain > self.gamma:
                    best_gain, best_feat, best_split = gain, f, bins[b]

        if best_feat is None:
            return np.clip(np.sum(grad)/(np.sum(hess)+self.lambda_), -10,10)

        left  = X[:,best_feat] <= best_split
        right = ~left
        ltree = self.fit(X[left],  grad[left],  hess[left],  depth+1)
        rtree = self.fit(X[right], grad[right], hess[right], depth+1)
        return (best_feat, best_split, ltree, rtree)

    # --- inference ---
    def _pred_row(self, x, node):
        if not isinstance(node, tuple): return node
        f, sp, l, r = node
        return self._pred_row(x, l) if x[f] <= sp else self._pred_row(x, r)

    def predict(self, X):
        return np.array([self._pred_row(x, self.tree) for x in X])

# ──────────────────────────────────────────────────────────
# Float baseline (unchanged)
# ──────────────────────────────────────────────────────────
class SimpleXGBoostClassifier:
    def __init__(self, n_estimators=50, max_depth=3, learning_rate=0.1,
                 lambda_=1, gamma=0, subsample=0.8, colsample_bytree=0.8):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.learning_rate=learning_rate; self.lambda_=lambda_
        self.gamma=gamma; self.subsample=subsample
        self.colsample_bytree=colsample_bytree
        self.trees=[]; self.initial_logit=0

    def sigmoid(self,x): return stable_sigmoid(x)

    def fit(self,X,y):
        eps=1e-6
        p=y.mean(); self.initial_logit=np.log(p/(1-p))
        y_pred=np.full_like(y,self.initial_logit,dtype=float)
        for _ in range(self.n_estimators):
            idx=np.random.choice(len(X),int(len(X)*self.subsample),replace=False)
            p=self.sigmoid(y_pred[idx]); p=np.clip(p,eps,1-eps)
            grad=p-y[idx]; hess=p*(1-p)
            tree=XGBoostTreeClassifier(max_depth=self.max_depth,
                                       lambda_=self.lambda_,gamma=self.gamma,
                                       colsample_bytree=self.colsample_bytree)
            tree.tree=tree.fit(X[idx],grad,hess)
            y_pred-=self.learning_rate*tree.predict(X)
            self.trees.append(tree)

    def predict_proba(self,X):
        y_pred=np.full(X.shape[0],self.initial_logit)
        for t in self.trees: y_pred-=self.learning_rate*t.predict(X)
        return self.sigmoid(y_pred)

    def predict(self,X): return (self.predict_proba(X)>=0.5).astype(int)

# ──────────────────────────────────────────────────────────
# Generic fixed-point booster (plug-in integer sigmoid)
# ──────────────────────────────────────────────────────────
class GenericFixedPointXGB(SimpleXGBoostClassifier):
    def __init__(self, int_sigmoid, terms=10, **kw):
        super().__init__(**kw)
        self.int_sigmoid=int_sigmoid
        self.terms=terms
        self.scaled_lr=float_to_fixed(self.learning_rate)

    # vectorised integer sigmoid
    def sigmoid(self,x):
        return np.fromiter((self.int_sigmoid(xi) for xi in x),
                           dtype=int,count=len(x))

    def fit(self,X,y):
        y_fix=np.array([float_to_fixed(v) for v in y],dtype=int)
        m=y_fix.mean(); self.initial_logit=float_to_fixed(math.log(m/(SCALE-m)))
        y_pred=np.full_like(y_fix,self.initial_logit)

        for _ in range(self.n_estimators):
            idx=np.random.choice(len(X),int(len(X)*self.subsample),replace=False)
            p_hat=self.sigmoid(y_pred[idx])
            grad=p_hat-y_fix[idx]
            hess=(p_hat*(SCALE-p_hat))//SCALE
            tree=XGBoostTreeClassifier(max_depth=self.max_depth,
                                       lambda_=self.lambda_,gamma=self.gamma,
                                       colsample_bytree=self.colsample_bytree)
            tree.tree=tree.fit(X[idx],grad,hess)
            upd=tree.predict(X)
            for i,u in enumerate(upd):
                y_pred[i]-=fixed_mul(self.scaled_lr,float_to_fixed(u))
            self.trees.append(tree)

    def predict_proba(self,X):
        y_pred=np.full(X.shape[0],self.initial_logit)
        for t in self.trees:
            upd=t.predict(X)
            for i,u in enumerate(upd):
                y_pred[i]-=fixed_mul(self.scaled_lr,float_to_fixed(u))
        return self.sigmoid(y_pred)

    def predict(self,X): return (self.predict_proba(X)>=SCALE//2).astype(int)

# wrappers for convenience
FixedLUT   = lambda **kw: GenericFixedPointXGB(lut_fixed_sigmoid,   **kw)
FixedWider = lambda **kw: GenericFixedPointXGB(wider_piecewise_sigmoid, **kw)

# ──────────────────────────────────────────────────────────
# Integer feature binning (unchanged)
# ──────────────────────────────────────────────────────────
def bin_features_fixed(X,num_bins=64):
    Xb=np.zeros_like(X,dtype=np.int32); s=num_bins-1
    for j in range(X.shape[1]):
        c=X[:,j]; mn, mx=c.min(),c.max()
        if mn==mx: continue
        cint=np.round(c*SCALE).astype(np.int64)
        bins=((cint-round(mn*SCALE))*s)//(round(mx*SCALE)-round(mn*SCALE))
        Xb[:,j]=np.clip(bins,0,s)
    return Xb

# ──────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────
def load_credit_dataset():
    d=np.genfromtxt('credit_default.csv',delimiter=',',skip_header=1,filling_values=0)
    X=d[:,1:-1]; y=(d[:,-1]>0.5).astype(int)
    return X,y

datasets={
    "breast_cancer":      load_breast_cancer(return_X_y=True),
    "credit_card_default":load_credit_dataset()
}

# ──────────────────────────────────────────────────────────
# Benchmark grid
# ──────────────────────────────────────────────────────────
depths  = [4]
trees_l = [50]
runs    = 1
terms   = 10   # kept for completeness

for ds,(X,y) in tqdm(datasets.items(),desc="Datasets",leave=True):
    print(f"\n=== Dataset: {ds} ===")
    Xf=StandardScaler().fit_transform(X)
    results=[]

    for d in depths:
        for nt in trees_l:
            sm_acc=sm_t=[]; fx_acc=fx_t=[]; wd_acc=wd_t=[]; xb_acc=xb_t=[]
            sm_acc,fx_acc,wd_acc,xb_acc=[],[],[],[]
            sm_t,fx_t,wd_t,xb_t=[],[],[],[]

            for r in range(runs):
                Xtr,Xte,ytr,yte=train_test_split(Xf,y,test_size=0.2,
                                                 random_state=42+r)

                # Simple float
                s=SimpleXGBoostClassifier(n_estimators=nt,max_depth=d)
                t0=time.time(); s.fit(Xtr,ytr); sm_t.append(time.time()-t0)
                sm_acc.append((s.predict(Xte)==yte).mean())

                # Binning for fixed models
                Xtrb,Xteb=bin_features_fixed(Xtr),bin_features_fixed(Xte)

                # LUT fixed
                f=FixedLUT(n_estimators=nt,max_depth=d)
                t0=time.time(); f.fit(Xtrb,ytr); fx_t.append(time.time()-t0)
                fx_acc.append((f.predict(Xteb)==yte).mean())

                # Wider fixed
                w=FixedWider(n_estimators=nt,max_depth=d)
                t0=time.time(); w.fit(Xtrb,ytr); wd_t.append(time.time()-t0)
                wd_acc.append((w.predict(Xteb)==yte).mean())

                # Real xgboost
                xg=xgb.XGBClassifier(n_estimators=nt,max_depth=d,
                                     verbosity=0,eval_metric='logloss')
                t0=time.time(); xg.fit(Xtr,ytr); xb_t.append(time.time()-t0)
                xb_acc.append((xg.predict(Xte)==yte).mean())

            results.append((terms,d,nt,
                            np.mean(sm_acc),np.mean(fx_acc),np.mean(wd_acc),np.mean(xb_acc),
                            np.mean(sm_t), np.mean(fx_t), np.mean(wd_t), np.mean(xb_t)))

    # print table
    hdr=("Terms Depth Trees | Simple Acc Fixed Acc Wider Acc XGB Acc | "
         "Simple T  Fixed T  Wider T  XGB T")
    print(hdr); print("-"*len(hdr))
    for r in results:
        print(f"{r[0]:<5}  {r[1]:<5}  {r[2]:<5} | "
              f"{r[3]:<9.4f} {r[4]:<9.4f} {r[5]:<9.4f} {r[6]:<8.4f} | "
              f"{r[7]:<9.2f} {r[8]:<9.2f} {r[9]:<9.2f} {r[10]:<6.2f}")
