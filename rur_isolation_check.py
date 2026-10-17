"""Apakah gain RA bisa dijelaskan hanya oleh INDIKATOR 'review punya/tidak punya tetangga R-U-R'
(praktis: apakah user menulis review lain di dataset), bukan oleh agregasi tetangga yang bersih?

1. AUROC prediktor tunggal: indikator terisolasi di R-U-R, dan log derajat R-U-R.
2. Apakah informasi itu SUDAH ada di 32 fitur mentah? (korelasi tiap kolom dengan indikator/derajat)
"""
import sys
import numpy as np
import scipy.io as scio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

d = scio.loadmat(sys.argv[1])
y = d["label"].ravel().astype(int)
X = d["features"]
X = X.toarray() if sp.issparse(X) else np.asarray(X)
a = sp.csr_matrix(d["net_rur"])
a = ((a + a.T) > 0).astype(np.int8)
a.setdiag(0); a.eliminate_zeros()
deg = np.asarray(a.sum(1)).ravel()
iso = (deg == 0).astype(int)

print(f"n={len(y)} | terisolasi di R-U-R: semua {iso.mean()*100:.2f}% | fraud {iso[y==1].mean()*100:.2f}% | normal {iso[y==0].mean()*100:.2f}%")
print(f"AUROC indikator terisolasi (skor = iso)      : {roc_auc_score(y, iso)*100:.2f}")
print(f"AUROC -log(1+deg_RUR)                         : {roc_auc_score(y, -np.log1p(deg))*100:.2f}")

# oracle label tetangga, HANYA pada node yang punya tetangga R-U-R
m = deg > 0
nb = a @ (y == 1).astype(float)
print(f"AUROC rata-rata label tetangga, node ber-tetangga saja (n={m.sum()}, fraud={y[m].sum()}): "
      f"{roc_auc_score(y[m], nb[m]/deg[m])*100:.2f}")

print("\nKolom fitur mentah paling terkait dengan isolasi/derajat R-U-R:")
rows = []
for j in range(X.shape[1]):
    x = X[:, j].astype(float)
    if x.std() == 0:
        continue
    auc_iso = roc_auc_score(iso, x)
    r_deg = np.corrcoef(x, np.log1p(deg))[0, 1]
    auc_y = roc_auc_score(y, x)
    rows.append((max(auc_iso, 1 - auc_iso), j, auc_iso, r_deg, auc_y, len(np.unique(x))))
rows.sort(reverse=True)
print(f"{'kolom':>5s} {'AUROC->iso':>10s} {'r(log deg)':>10s} {'AUROC->fraud':>12s} {'#nilai unik':>11s}")
for s, j, ai, r, ay, nu in rows[:6]:
    print(f"{j:>5d} {ai*100:>10.2f} {r:>10.3f} {ay*100:>12.2f} {nu:>11d}")