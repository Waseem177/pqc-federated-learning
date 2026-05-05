import os, urllib.request
import numpy as np

PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
PIMA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pima_diabetes.csv")

def load_pima():
    if not os.path.exists(PIMA_CACHE):
        print("  Downloading PIMA dataset...")
        urllib.request.urlretrieve(PIMA_URL, PIMA_CACHE)
    data = np.loadtxt(PIMA_CACHE, delimiter=",")
    X, y = data[:, :8].astype(np.float32), data[:, 8].astype(np.float32)
    for col in [1, 2, 3, 4, 5]:
        mask = X[:, col] == 0
        X[mask, col] = np.median(X[~mask, col])
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    return X, y

def load_and_partition(n_nodes=5, test_ratio=0.2, seed=42, verbose=True):
    X, y = load_pima()
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]

    def split(idx):
        rng.shuffle(idx)
        n = max(1, int(len(idx)*test_ratio))
        return idx[n:], idx[:n]

    pos_tr, pos_te = split(pos)
    neg_tr, neg_te = split(neg)
    tr_idx = np.concatenate([pos_tr, neg_tr])
    te_idx = np.concatenate([pos_te, neg_te])
    rng.shuffle(tr_idx)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    pos_tr2 = np.where(y_tr==1)[0]
    neg_tr2 = np.where(y_tr==0)[0]
    rng.shuffle(pos_tr2); rng.shuffle(neg_tr2)
    pos_splits = np.array_split(pos_tr2, n_nodes)
    neg_splits = np.array_split(neg_tr2, n_nodes)

    shards = []
    for i in range(n_nodes):
        idx = np.concatenate([pos_splits[i], neg_splits[i]])
        rng.shuffle(idx)
        Xn, yn = X_tr[idx], y_tr[idx]
        shards.append((Xn, yn, X_te, y_te))
        if verbose:
            print(f"  Node {i}: {len(Xn):4d} train  ({yn.mean()*100:.1f}% pos)  | {len(X_te)} test")
    if verbose:
        print(f"  Dataset: {len(X)} total | {int(y.sum())} positive ({y.mean()*100:.1f}%)")
    return shards

if __name__ == "__main__":
    print("Partitioning PIMA dataset:")
    shards = load_and_partition(n_nodes=5)
