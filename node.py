import io, pickle
import numpy as np
from pqc_layer import KEMPublicKey, SigKeyPair, encrypt_and_sign
from classical_layer import classical_encrypt

class FeedForwardNN:
    def __init__(self, input_dim=8, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/input_dim), (input_dim, 64)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2/64), (64, 32)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2/32), (32, 1)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

    def _relu(self, x): return np.maximum(0, x)
    def _sig(self, x): return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

    def forward(self, X):
        self._X = X
        self._z1 = X @ self.W1 + self.b1; self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2; self._a2 = self._relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3; self._out = self._sig(self._z3)
        return self._out

    def backward(self, y, lr=1e-3):
        n = len(y); y = y.reshape(-1,1)
        d3 = (self._out - y)/n
        dW3 = self._a2.T @ d3; db3 = d3.sum(0)
        d2 = (d3 @ self.W3.T) * (self._z2 > 0)
        dW2 = self._a1.T @ d2; db2 = d2.sum(0)
        d1 = (d2 @ self.W2.T) * (self._z1 > 0)
        dW1 = self._X.T @ d1; db1 = d1.sum(0)
        self.W1-=lr*dW1; self.b1-=lr*db1
        self.W2-=lr*dW2; self.b2-=lr*db2
        self.W3-=lr*dW3; self.b3-=lr*db3

    def get_weights(self): return [self.W1.copy(),self.b1.copy(),self.W2.copy(),self.b2.copy(),self.W3.copy(),self.b3.copy()]
    def set_weights(self, w): self.W1,self.b1,self.W2,self.b2,self.W3,self.b3 = [x.copy() for x in w]
    def num_params(self): return sum(x.size for x in self.get_weights())

    def evaluate(self, X, y):
        p = self.forward(X).ravel()
        pred = (p>=0.5).astype(int); yi = y.astype(int)
        acc = (pred==yi).mean()
        eps = 1e-8
        loss = -np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))
        tp=((pred==1)&(yi==1)).sum(); fp=((pred==1)&(yi==0)).sum()
        fn=((pred==0)&(yi==1)).sum(); tn=((pred==0)&(yi==0)).sum()
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        f1=2*prec*rec/max(prec+rec,1e-8)
        sensitivity=float(tp/max(tp+fn,1))   # TPR / recall
        specificity=float(tn/max(tn+fp,1))   # TNR
        th=np.linspace(0,1,100); tprs=[]; fprs=[]
        for t in th:
            pb=(p>=t).astype(int)
            tpr_v=((pb==1)&(yi==1)).sum()/max(((yi==1)).sum(),1)
            fpr_v=((pb==1)&(yi==0)).sum()/max(((yi==0)).sum(),1)
            tprs.append(tpr_v); fprs.append(fpr_v)
        auc=abs(np.trapezoid(tprs,fprs))
        return {"loss":float(loss),"accuracy":float(acc),"f1":float(f1),
                "auc":float(auc),"sensitivity":sensitivity,"specificity":specificity}

def weights_to_bytes(w):
    buf=io.BytesIO(); pickle.dump([x.astype(np.float32) for x in w],buf); return buf.getvalue()

def bytes_to_weights(b): return pickle.loads(b)

DP_DELTA = 1e-5   # (ε, δ)-DP; δ fixed, ε is the privacy budget supplied by caller
DP_CLIP  = 0.10   # L2 sensitivity — clips deltas whose norm exceeds this value

def _dp_noise_sigma(epsilon):
    """Gaussian mechanism: σ = S * sqrt(2*ln(1.25/δ)) / ε"""
    import math
    return DP_CLIP * math.sqrt(2 * math.log(1.25 / DP_DELTA)) / epsilon


class FederatedNode:
    def __init__(self, node_id, X_train, y_train, X_test, y_test,
                 server_kem_pub, server_rsa_pub=None, hmac_key=b"",
                 local_epochs=5, batch_size=32, lr=1e-3, mode="pqc",
                 model_seed=None, epsilon=None):
        self.node_id=node_id; self.X_train=X_train; self.y_train=y_train
        self.X_test=X_test; self.y_test=y_test
        self.server_kem_pub=server_kem_pub; self.server_rsa_pub=server_rsa_pub
        self.hmac_key=hmac_key; self.local_epochs=local_epochs
        self.batch_size=batch_size; self.lr=lr; self.mode=mode
        self.epsilon=epsilon   # None = no DP
        _seed = model_seed if model_seed is not None else node_id*42
        self.model=FeedForwardNN(input_dim=X_train.shape[1], seed=_seed)
        self.sig_kp=SigKeyPair()

    def set_global_weights(self, w): self.model.set_weights(w)

    def train_local(self):
        X,y=self.X_train,self.y_train; n=len(X)
        idx=np.random.permutation(n)
        for _ in range(self.local_epochs):
            for s in range(0,n,self.batch_size):
                b=idx[s:s+self.batch_size]
                self.model.forward(X[b]); self.model.backward(y[b],lr=self.lr)
        return self.model.get_weights()

    def _apply_dp(self, delta):
        """Clip delta to DP_CLIP, then add calibrated Gaussian noise."""
        flat = np.concatenate([l.ravel() for l in delta])
        l2 = float(np.linalg.norm(flat))
        scale = min(1.0, DP_CLIP / (l2 + 1e-12))
        clipped = [l * scale for l in delta]
        sigma = _dp_noise_sigma(self.epsilon)
        return [l + np.random.normal(0, sigma, l.shape).astype(np.float32) for l in clipped]

    def prepare_update(self, global_weights):
        pre=[w.copy() for w in global_weights]
        self.set_global_weights(pre)
        new=self.train_local()
        delta=[n-p for n,p in zip(new,pre)]
        if self.epsilon is not None:
            delta = self._apply_dp(delta)
        delta_bytes=weights_to_bytes(delta)
        metrics=self.model.evaluate(self.X_test,self.y_test)
        if self.mode=="pqc":
            bundle,timing=encrypt_and_sign(self.node_id,delta_bytes,self.server_kem_pub,self.sig_kp)
        else:
            bundle,timing=classical_encrypt(self.node_id,delta_bytes,self.server_rsa_pub,self.hmac_key)
        return bundle,timing,metrics,len(delta_bytes)
