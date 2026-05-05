import numpy as np
from pqc_layer import KEMKeyPair, decrypt_and_verify
from classical_layer import RSAKeyPair, classical_decrypt
from node import bytes_to_weights
import os

class AggregationServer:
    def __init__(self, mode="pqc"):
        self.mode=mode
        self.kem_kp=KEMKeyPair()
        if mode=="classical":
            self.rsa_kp=RSAKeyPair()
            self.hmac_key=os.urandom(32)
        else:
            self.rsa_kp=None; self.hmac_key=b""
        self._global_weights=None
        self._shard_sizes={}

    @property
    def kem_public_key(self): return self.kem_kp.public_key
    @property
    def rsa_public_key(self): return self.rsa_kp.public_key if self.rsa_kp else None
    @property
    def global_weights(self): return self._global_weights

    def init_weights(self, w): self._global_weights=[x.copy() for x in w]
    def register_node(self, node_id, shard_size): self._shard_sizes[node_id]=shard_size

    def aggregate(self, bundles):
        deltas={}; timings=[]; rejected=[]; total_bytes=0
        for b in bundles:
            total_bytes+=b.total_bytes()
            try:
                if self.mode=="pqc":
                    delta_bytes,t=decrypt_and_verify(b,self.kem_kp)
                else:
                    delta_bytes,t=classical_decrypt(b,self.rsa_kp,self.hmac_key)
                deltas[b.node_id]=bytes_to_weights(delta_bytes)
                timings.append(t)
            except Exception as e:
                print(f"  Node {b.node_id} REJECTED: {e}")
                rejected.append(b.node_id)
        if not deltas: raise RuntimeError("All updates rejected")
        total=sum(self._shard_sizes[i] for i in deltas)
        agg=None
        for nid,delta in deltas.items():
            w=self._shard_sizes[nid]/total
            agg=[l*w for l in delta] if agg is None else [a+l*w for a,l in zip(agg,delta)]
        self._global_weights=[g+a for g,a in zip(self._global_weights,agg)]
        return {"timings":timings,"rejected":rejected,"accepted":list(deltas.keys()),"total_bytes":total_bytes}

    def frob(self, prev):
        return float(np.sqrt(sum(np.sum((n-p)**2) for n,p in zip(self._global_weights,prev))))
