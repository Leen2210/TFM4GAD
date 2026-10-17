"""Apakah relasi 'bersih' bisa dipilih dari label yang tersedia pada protokol 100 label?
Simulasi protokol semi-supervised GADBench (20 fraud + 80 normal train; +100 val) dengan 10 seed.
Hitung edge relasi yang KEDUA ujungnya berlabel (satu-satunya edge yang bisa dipakai menaksir H(r))."""
import sys
import numpy as np
import scipy.io as scio
import scipy.sparse as sp

d = scio.loadmat(sys.argv[1])
y = d["label"].ravel().astype(int)
n = len(y)
rels = {}
for r in ["net_rur", "net_rtr", "net_rsr"]:
    a = sp.csr_matrix(d[r])
    a = ((a + a.T) > 0).astype(np.int8)
    a.setdiag(0)
    a.eliminate_zeros()
    rels[r] = sp.triu(a, 1).tocsr()

pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
for setting, mult in [("train saja (100 label)", 1), ("train+val (200 label)", 2)]:
    print(f"\n=== {setting} ===")
    acc = {r: [] for r in rels}
    hs = {r: [] for r in rels}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        lab = np.concatenate([rng.choice(pos, 20 * mult, replace=False), rng.choice(neg, 80 * mult, replace=False)])
        m = np.zeros(n, bool); m[lab] = True
        for r, a in rels.items():
            sub = a[lab][:, lab].tocoo()
            acc[r].append(sub.nnz)
            if sub.nnz:
                hs[r].append(np.mean(y[lab][sub.row] != y[lab][sub.col]))
    for r in rels:
        e = np.array(acc[r])
        h = f"{np.mean(hs[r]):.3f}" if hs[r] else "tak terdefinisi"
        print(f"  {r}: edge berlabel-berlabel per split = min {e.min()}, median {int(np.median(e))}, maks {e.max()} "
              f"| split dengan 0 edge = {(e == 0).sum()}/10 | taksiran H rata-rata = {h}")