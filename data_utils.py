import os, urllib.request
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── remote URLs ───────────────────────────────────────────────────────────────
_PIMA_URL      = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
_CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
_WDBC_URL      = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

_PIMA_CACHE      = os.path.join(_DIR, "pima_diabetes.csv")
_CLEVELAND_CACHE = os.path.join(_DIR, "cleveland.data")
_WDBC_CACHE      = os.path.join(_DIR, "wdbc.data")


def _fetch(url, path):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


def _normalise(X):
    lo, hi = X.min(0), X.max(0)
    return ((X - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _pad(X, target_dim):
    if X.shape[1] == target_dim:
        return X
    zeros = np.zeros((len(X), target_dim - X.shape[1]), dtype=np.float32)
    return np.concatenate([X, zeros], axis=1)


def _stratified_split(X, y, test_ratio, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos_te = max(1, int(len(pos) * test_ratio))
    n_neg_te = max(1, int(len(neg) * test_ratio))
    te_idx = np.concatenate([pos[:n_pos_te], neg[:n_neg_te]])
    tr_idx = np.concatenate([pos[n_pos_te:], neg[n_neg_te:]])
    rng.shuffle(tr_idx); rng.shuffle(te_idx)
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]


def _stratified_shards(X, y, n_nodes, test_ratio, rng):
    """Split pooled data into n_nodes balanced, stratified hospital shards."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be at least 1")
    if n_nodes > len(X):
        raise ValueError(f"n_nodes={n_nodes} exceeds number of samples={len(X)}")

    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    pos_parts = np.array_split(pos, n_nodes)
    neg_parts = np.array_split(neg, n_nodes)

    shards = []
    for pos_idx, neg_idx in zip(pos_parts, neg_parts):
        idx = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(idx)
        Xtr, ytr, Xte, yte = _stratified_split(X[idx], y[idx], test_ratio, rng)
        shards.append((Xtr, ytr, Xte, yte))
    return shards


# ── dataset loaders ───────────────────────────────────────────────────────────

def _load_pima():
    _fetch(_PIMA_URL, _PIMA_CACHE)
    data = np.loadtxt(_PIMA_CACHE, delimiter=",")
    X, y = data[:, :8].astype(np.float32), data[:, 8].astype(np.float32)
    # impute zero-coded missing values with column median
    for col in [1, 2, 3, 4, 5]:
        mask = X[:, col] == 0
        X[mask, col] = np.median(X[~mask, col])
    return X, y   # 768 × 8,  34.9% positive


def _load_cleveland():
    _fetch(_CLEVELAND_URL, _CLEVELAND_CACHE)
    rows = []
    with open(_CLEVELAND_CACHE) as f:
        for line in f:
            parts = line.strip().split(",")
            if "?" in parts:
                continue          # drop 6 rows with missing ca / thal
            rows.append([float(v) for v in parts])
    data = np.array(rows, dtype=np.float32)
    X = data[:, :13]
    y = (data[:, 13] > 0).astype(np.float32)   # binarise 0-4 → 0/1
    return X, y   # ~297 × 13,  ~54% positive


def _load_wdbc():
    _fetch(_WDBC_URL, _WDBC_CACHE)
    rows, labels = [], []
    with open(_WDBC_CACHE) as f:
        for line in f:
            parts = line.strip().split(",")
            labels.append(1.0 if parts[1] == "M" else 0.0)   # M = malignant
            rows.append([float(v) for v in parts[2:]])        # skip ID + diagnosis
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.float32)
    # 569 × 30,  ~37% positive


# ── public API ────────────────────────────────────────────────────────────────

def _partition_pima_noniid(X, y, n_nodes, test_ratio, rng):
    """
    Non-IID split of Pima dataset across n_nodes hospitals.
    Simulates real-world skew: hospitals are sorted by patient glucose level
    (feature index 1 in raw Pima), then divided into contiguous quantile bands.
    Each hospital has a different glucose distribution → genuine non-IID.
    """
    glucose = X[:, 1]
    order   = np.argsort(glucose)
    X_sorted, y_sorted = X[order], y[order]

    shards = []
    bands  = np.array_split(np.arange(len(X_sorted)), n_nodes)
    for band in bands:
        Xi, yi = X_sorted[band], y_sorted[band]
        Xtr, ytr, Xte, yte = _stratified_split(Xi, yi, test_ratio, rng)
        shards.append((Xtr, ytr, Xte, yte))
    return shards


def load_and_partition(n_nodes=5, test_ratio=0.2, seed=42, verbose=True,
                       dataset="mixed"):
    """
    dataset="mixed"  — heterogeneous: pooled PIMA + Cleveland + WDBC
    dataset="pima"   — homogeneous: Pima diabetes only, non-IID by glucose quartile
    """
    rng = np.random.default_rng(seed)

    # ── Pima-only (homogeneous, 5 diabetes clinics) ───────────────────────────
    if dataset == "pima":
        X, y = _load_pima()
        X    = _normalise(X)
        shards = _partition_pima_noniid(X, y, n_nodes, test_ratio, rng)
        if verbose:
            print(f"  Dataset: Pima Indians Diabetes  "
                  f"(n={len(X)}, non-IID by glucose quartile)")
            for i, (Xtr, ytr, Xte, yte) in enumerate(shards):
                print(f"  Node {i}: {len(Xtr):3d} train "
                      f"({ytr.mean()*100:.1f}% pos) | {len(Xte):3d} test")
            print(f"  Input dim: {X.shape[1]}")
        return shards

    # ── Mixed (heterogeneous, multi-disease) ──────────────────────────────────
    X_pima,  y_pima  = _load_pima()
    X_heart, y_heart = _load_cleveland()
    X_wdbc,  y_wdbc  = _load_wdbc()

    # normalise each hospital independently — simulates local calibration
    X_pima  = _normalise(X_pima)
    X_heart = _normalise(X_heart)
    X_wdbc  = _normalise(X_wdbc)

    # zero-pad all to WDBC dimensionality (largest at 30 features)
    max_dim = max(X_pima.shape[1], X_heart.shape[1], X_wdbc.shape[1])  # 30
    X_pima  = _pad(X_pima,  max_dim)
    X_heart = _pad(X_heart, max_dim)
    X_wdbc  = _pad(X_wdbc,  max_dim)

    X_all = np.concatenate([X_pima, X_heart, X_wdbc], axis=0)
    y_all = np.concatenate([y_pima, y_heart, y_wdbc], axis=0)
    shards = _stratified_shards(X_all, y_all, n_nodes, test_ratio, rng)

    if verbose:
        print(f"  Dataset: pooled PIMA + Cleveland + WDBC "
              f"(stratified into {n_nodes} shards)")
        for i, (Xtr, ytr, Xte, yte) in enumerate(shards):
            print(f"  Node {i}: {len(Xtr):4d} train "
                  f"({ytr.mean()*100:.1f}% pos) | {len(Xte):3d} test")
        print(f"  Input dim: {max_dim}  (PIMA/Heart zero-padded to WDBC size)")
        print(f"  Datasets: PIMA n={len(X_pima)}, "
              f"Cleveland n={len(X_heart)}, WDBC n={len(X_wdbc)}")

    return shards


if __name__ == "__main__":
    shards = load_and_partition(n_nodes=5)
