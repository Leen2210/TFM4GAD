"""
Jalankan di dalam container:
    python inspect_yelp_graph.py --data_dir ../../datasets
"""

import argparse
import os

import dgl
import torch
from dgl.data.utils import load_graphs


def sep(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def edge_heterophily(src, dst, y):
    """H(r) = proporsi edge yang menghubungkan node berlabel berbeda."""
    diff = (y[src] != y[dst]).float()
    return diff.mean().item()


def fraud_node_heterophily(src, dst, y, num_nodes):
    """
    Untuk tiap node anomali, hitung rasio tetangga
    yang berlabel berbeda + berapa persen node anomali yang
    rasio heterophily-nya > 50% + berapa yang ga punya tetangga.
    """
    deg = torch.zeros(num_nodes)
    het = torch.zeros(num_nodes)
    deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    het.index_add_(0, src, (y[src] != y[dst]).float())

    fraud = (y == 1)
    f_deg = deg[fraud]
    f_het = het[fraud]

    no_nbr = (f_deg == 0)
    ratio = torch.zeros_like(f_deg)
    ratio[~no_nbr] = f_het[~no_nbr] / f_deg[~no_nbr]

    return {
        "n_fraud": int(fraud.sum()),
        "pct_no_neighbor": 100.0 * no_nbr.float().mean().item(),
        "pct_ratio_gt_50": 100.0 * (ratio > 0.5).float().mean().item(),
        "mean_ratio": ratio.mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../../datasets")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Cek graf homogen yg dipakai TFM4GAD sekarang
    # ------------------------------------------------------------------
    sep("1. datasets/yelp  (graf homogen — jalur baseline TFM4GAD)")
    homo_path = os.path.join(args.data_dir, "yelp")
    g = load_graphs(homo_path)[0][0] #return load_graphs itu (daftar graph, dict label) 

    print("tipe objek                          :", type(g).__name__) #mastiin objeknya DGLGraph
    print("ntypes (daftar tipe nodes)          :", g.ntypes) #daftar tipe nodes, cek ada per relasi ga
    print("canonical_etypes (daftar tipe edge) :", g.canonical_etypes) #daftar tipe edge
    print("num_nodes                           :", g.num_nodes())
    print("num_edges (berarah)                 :", g.num_edges()) #node tidak berarah akan terhitung 2x
    print("ndata keys                          :", list(g.ndata.keys()))
    print("feature shape                       :", g.ndata["feature"].shape)

    y = g.ndata["label"].long().view(-1)
    print("anomaly ratio   : %.4f" % y.float().mean().item())

    # ------------------------------------------------------------------
    # 2. Lihat isi train_mask ada apa saja
    # ------------------------------------------------------------------
    sep("2. struktur train_masks / val_masks / test_masks")
    tm, vm, sm = g.ndata["train_masks"], g.ndata["val_masks"], g.ndata["test_masks"]
    print("shape train/val/test:", tuple(tm.shape), tuple(vm.shape), tuple(sm.shape))
    print("\ncol | n_train | n_anomali_train | n_val | n_test")
    for c in range(tm.shape[1]):
        col = tm[:, c].bool()
        print("%3d | %7d | %15d | %5d | %6d" % (
            c, int(col.sum()), int(y[col].sum()),
            int(vm[:, c].bool().sum()), int(sm[:, c].bool().sum())))

    # ------------------------------------------------------------------
    # 3. Cek versi Hetero
    # ------------------------------------------------------------------
    sep("3. datasets/hetero/yelp (graf multi-relasi)")
    het_path = os.path.join(args.data_dir, "hetero", "yelp")
    if not (os.path.exists(het_path) or os.path.exists(het_path + ".bin")):
        print("TIDAK DITEMUKAN di", het_path)
        return

    gh = load_graphs(het_path)[0][0]
    print("tipe objek      :", type(gh).__name__)
    print("ntypes          :", gh.ntypes)
    print("canonical_etypes:", gh.canonical_etypes)
    print("num_nodes       :", gh.num_nodes())
    print("ndata keys      :", list(gh.ndata.keys()))

    yh = gh.ndata["label"].long().view(-1)

    print("\nper-relasi:")
    for ce in gh.canonical_etypes:
        print("  %-28s edges=%d" % (str(ce), gh.num_edges(ce)))

    # ------------------------------------------------------------------
    # 4. Cek apakah urutan node hetero sama dengan homo
    # ------------------------------------------------------------------
    sep("4. apakah hetero yelp == yelp setelah to_homogeneous?")
    same_n = gh.num_nodes() == g.num_nodes()
    same_feat = torch.allclose(gh.ndata["feature"], g.ndata["feature"]) #bandingkan nilai semua fitur untuk liat identik atau tidak
    same_label = torch.equal(yh, y)
    same_mask = torch.equal(gh.ndata["train_masks"], g.ndata["train_masks"])
    print("jumlah node sama      :", same_n)
    print("matriks fitur sama    :", same_feat)
    print("vektor label sama     :", same_label)
    print("train_masks sama      :", same_mask)

    gh_homo = dgl.to_homogeneous(gh) #ubah hetero jd homogen (relasi duplikat dipertahankan)
    print("edges to_homogeneous  :", gh_homo.num_edges(), "| edges yelp:", g.num_edges())

    # ------------------------------------------------------------------
    # 5. Hitung H(r) per relasi untuk exp 3
    # ------------------------------------------------------------------
    sep("5. heterophily per relasi")
    n = gh.num_nodes()
    print("%-14s | %-9s | %-10s | %-12s | %-12s" % (
        "relasi", "H(r) edge", "n_fraud", "%rasio>50%", "%tanpa tetangga"))
    for ce in gh.canonical_etypes:
        src, dst = gh.edges(etype=ce)
        h = edge_heterophily(src, dst, yh)
        st = fraud_node_heterophily(src, dst, yh, n)
        print("%-14s | %9.4f | %10d | %11.2f%% | %14.2f%%" % (
            ce[1], h, st["n_fraud"], st["pct_ratio_gt_50"], st["pct_no_neighbor"]))

    src, dst = gh_homo.edges()
    print("\nH keseluruhan (graf gabungan): %.4f" % edge_heterophily(src, dst, yh))


if __name__ == "__main__":
    main()