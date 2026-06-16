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
    Sorted by glucose level (feature 1) into contiguous quantile bands
    → each hospital has a different glucose distribution (genuine non-IID).
    """
    order = np.argsort(X[:, 1])
    X_sorted, y_sorted = X[order], y[order]
    shards = []
    for band in np.array_split(np.arange(len(X_sorted)), n_nodes):
        Xi, yi = X_sorted[band], y_sorted[band]
        shards.append(_stratified_split(Xi, yi, test_ratio, rng))
    return shards


def _load_hetero_shards(test_ratio, rng):
    """
    Node-specialised heterogeneous federation (always 5 nodes):
      Node 0: Pima diabetes shard A  (clean, ~256 samples)
      Node 1: Pima diabetes shard B  (clean, ~256 samples)
      Node 2: Cleveland heart disease (full 297 samples)
      Node 3: WDBC breast cancer     (full 569 samples)
      Node 4: Pima diabetes shard C  (Gaussian noise σ=0.1, lower-quality source)
    All features zero-padded to 30 dimensions (WDBC size).
    Returns (shards, dataset_labels).
    """
    X_pima, y_pima   = _load_pima()
    X_heart, y_heart = _load_cleveland()
    X_wdbc,  y_wdbc  = _load_wdbc()

    X_pima  = _normalise(X_pima)
    X_heart = _normalise(X_heart)
    X_wdbc  = _normalise(X_wdbc)

    max_dim = 30  # WDBC dimensionality

    # Split Pima into 3 equal parts (for nodes 0, 1, 4)
    idx = np.arange(len(X_pima))
    rng.shuffle(idx)
    pima_parts = np.array_split(idx, 3)

    shards, labels = [], []

    # Nodes 0, 1: clean Pima shards
    for part in pima_parts[:2]:
        Xi = _pad(X_pima[part], max_dim)
        yi = y_pima[part]
        shards.append(_stratified_split(Xi, yi, test_ratio, rng))
        labels.append("diabetes")

    # Node 2: Cleveland heart disease
    shards.append(_stratified_split(_pad(X_heart, max_dim), y_heart, test_ratio, rng))
    labels.append("heart")

    # Node 3: WDBC breast cancer
    shards.append(_stratified_split(X_wdbc, y_wdbc, test_ratio, rng))
    labels.append("cancer")

    # Node 4: noisy Pima shard (Gaussian noise simulates lower-quality data)
    Xi = _pad(X_pima[pima_parts[2]], max_dim)
    yi = y_pima[pima_parts[2]]
    noise = rng.normal(0, 0.1, Xi.shape).astype(np.float32)
    shards.append(_stratified_split(np.clip(Xi + noise, 0.0, 1.0), yi, test_ratio, rng))
    labels.append("diabetes_noisy")

    return shards, labels


def load_and_partition(n_nodes=5, test_ratio=0.2, seed=42, verbose=True,
                       dataset="hetero"):
    """
    Returns (shards, dataset_labels).

    dataset="hetero" — node-specialised 5-hospital federation (primary evaluation):
                       nodes 0-1 diabetes, node 2 heart, node 3 cancer,
                       node 4 noisy diabetes; n_nodes must be 5.
    dataset="pima"   — Pima diabetes only, non-IID by glucose quartile (ablation).
    dataset="mixed"  — pooled PIMA+Cleveland+WDBC, randomly distributed
                       (used by scalability experiment for variable n_nodes).
    """
    rng = np.random.default_rng(seed)

    # ── Pima-only ablation ────────────────────────────────────────────────────
    if dataset == "pima":
        X, y = _load_pima()
        X = _normalise(X)
        shards = _partition_pima_noniid(X, y, n_nodes, test_ratio, rng)
        labels = ["diabetes"] * n_nodes
        if verbose:
            print(f"  Dataset: Pima Indians Diabetes  "
                  f"(n={len(X)}, non-IID by glucose quartile)")
            for i, (Xtr, ytr, Xte, yte) in enumerate(shards):
                print(f"  Node {i}: {len(Xtr):3d} train "
                      f"({ytr.mean()*100:.1f}% pos) | {len(Xte):3d} test")
            print(f"  Input dim: {X.shape[1]}")
        return shards, labels

    # ── Node-specialised heterogeneous federation ─────────────────────────────
    if dataset == "hetero":
        if n_nodes != 5:
            raise ValueError("dataset='hetero' requires n_nodes=5")
        shards, labels = _load_hetero_shards(test_ratio, rng)
        if verbose:
            node_names = ["Pima-A", "Pima-B", "Cleveland", "WDBC", "Pima-C (noisy)"]
            print(f"  Dataset: node-specialised heterogeneous federation (5 hospitals)")
            for i, (Xtr, ytr, Xte, yte) in enumerate(shards):
                print(f"  Node {i} [{node_names[i]:16s}]: {len(Xtr):3d} train "
                      f"({ytr.mean()*100:.1f}% pos) | {len(Xte):2d} test  "
                      f"[{labels[i]}]")
            print(f"  Input dim: 30 (Pima/Cleveland zero-padded to WDBC size)")
        return shards, labels

    # ── Mixed pool-and-distribute (scalability experiment, variable n_nodes) ──
    X_pima,  y_pima  = _load_pima()
    X_heart, y_heart = _load_cleveland()
    X_wdbc,  y_wdbc  = _load_wdbc()

    X_pima  = _normalise(X_pima)
    X_heart = _normalise(X_heart)
    X_wdbc  = _normalise(X_wdbc)

    max_dim = max(X_pima.shape[1], X_heart.shape[1], X_wdbc.shape[1])  # 30
    X_pima  = _pad(X_pima,  max_dim)
    X_heart = _pad(X_heart, max_dim)
    X_wdbc  = _pad(X_wdbc,  max_dim)

    X_all = np.concatenate([X_pima, X_heart, X_wdbc], axis=0)
    y_all = np.concatenate([y_pima, y_heart, y_wdbc], axis=0)
    shards = _stratified_shards(X_all, y_all, n_nodes, test_ratio, rng)
    labels = ["mixed"] * n_nodes

    if verbose:
        print(f"  Dataset: pooled PIMA + Cleveland + WDBC "
              f"(stratified into {n_nodes} shards)")
        for i, (Xtr, ytr, Xte, yte) in enumerate(shards):
            print(f"  Node {i}: {len(Xtr):4d} train "
                  f"({ytr.mean()*100:.1f}% pos) | {len(Xte):3d} test")
        print(f"  Input dim: {max_dim}  (PIMA/Heart zero-padded to WDBC size)")
        print(f"  Datasets: PIMA n={len(X_pima)}, "
              f"Cleveland n={len(X_heart)}, WDBC n={len(X_wdbc)}")

    return shards, labels


if __name__ == "__main__":
    shards, labels = load_and_partition(n_nodes=5)
    print(labels)
