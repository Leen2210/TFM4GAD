"""Diagnostik struktural lintas dataset: apakah kondisi 'relasi bersih yang tenggelam' ada?

Per relasi (graf tak berarah, sederhana, tanpa self-loop — setara rantai praproses baseline):
  - share   : porsi edge relasi terhadap graf gabungan
  - H(r)    : heterofili edge
  - fraud_nb: % fraud yang punya >=1 tetangga
  - oAUC    : AUROC 'oracle' skor = rata-rata label tetangga (node tanpa tetangga = prior).
              Ini BUKAN hasil model; ini ukuran berapa banyak sinyal label yang dibawa
              struktur relasi tsb. Bandingkan oAUC relasi terbaik vs oAUC graf gabungan:
              selisih besar = sinyal bersih yang tenggelam saat digabung.
"""
import sys
import numpy as np
import scipy.io as scio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score


def clean(adj, n):
    a = sp.csr_matrix(adj, shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def stats(name, a, y, eval_idx, merged_nnz, p):
    deg = np.asarray(a.sum(1)).ravel()
    coo = sp.triu(a, 1).tocoo()
    H = float((y[coo.row] != y[coo.col]).mean()) if coo.nnz else float("nan")
    nb_fraud = a @ (y == 1).astype(np.float32)
    score = np.where(deg > 0, nb_fraud / np.maximum(deg, 1), p)
    ye, se, de = y[eval_idx], score[eval_idx], deg[eval_idx]
    oauc = roc_auc_score(ye, se)
    fraud = ye == 1
    print(f"  {name:10s} edges={coo.nnz:>9d} share={coo.nnz / merged_nnz * 100:6.1f}% "
          f"H={H:.4f} node_nb={np.mean(de > 0) * 100:6.2f}% fraud_nb={np.mean(de[fraud] > 0) * 100:6.2f}% "
          f"oAUC={oauc * 100:6.2f}")
    return oauc


def run(title, rels, merged, y, eval_idx):
    n = y.shape[0]
    p = float(y[eval_idx].mean())
    print(f"\n=== {title} | n={n} | dievaluasi={len(eval_idx)} | p={p:.4f} | H_acak=2p(1-p)={2 * p * (1 - p):.4f} ===")
    m = clean(merged, n)
    mnnz = sp.triu(m, 1).nnz
    res = {}
    for r, adj in rels.items():
        res[r] = stats(r, clean(adj, n), y, eval_idx, mnnz, p)
    res["MERGED"] = stats("MERGED", m, y, eval_idx, mnnz, p)
    best = max((k for k in res if k != "MERGED"), key=res.get)
    print(f"  -> relasi terbaik {best}: oAUC {res[best] * 100:.2f} vs gabungan {res['MERGED'] * 100:.2f} "
          f"(selisih {(res[best] - res['MERGED']) * 100:+.2f})")


which = sys.argv[1]
path = sys.argv[2]
if which == "yelp":
    d = scio.loadmat(path)
    y = d["label"].ravel().astype(int)
    run("YelpChi", {"R-U-R": d["net_rur"], "R-T-R": d["net_rtr"], "R-S-R": d["net_rsr"]},
        d["homo"], y, np.arange(len(y)))
elif which == "amazon":
    d = scio.loadmat(path)
    y = d["label"].ravel().astype(int)
    rels = {"U-P-U": d["net_upu"], "U-S-U": d["net_usu"], "U-V-U": d["net_uvu"]}
    print(f"[info] label node 0-3304: {np.bincount(y[:3305])} (dianggap tak berlabel oleh CARE-GNN/SplitGNN)")
    run("Amazon (eval node 3305+)", rels, d["homo"], y, np.arange(3305, len(y)))
    run("Amazon (eval semua node)", rels, d["homo"], y, np.arange(len(y)))
elif which == "comp":
    import dgl
    from dgl.data.utils import load_graphs
    g = load_graphs(path)[0][0]
    n = g.num_nodes()
    y = g.ndata["label"].numpy().astype(int)

    def adj(et):
        s, t = g.edges(etype=et)
        return sp.coo_matrix((np.ones(len(s)), (s.numpy(), t.numpy())), shape=(n, n))
    run("FDCompCN", {"invest": adj("invest_bc2bc"), "provide": adj("provide_bc2bc"), "sale": adj("sale_bc2bc")},
        adj("homo"), y, np.arange(n))