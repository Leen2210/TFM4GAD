import argparse
import os

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs


def sep(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def encode(src, dst, n):
    return src.astype(np.int64) * np.int64(n) + dst.astype(np.int64)


def symmetry_report(src, dst, n):
    fwd = np.sort(encode(src, dst, n))
    bwd = np.sort(encode(dst, src, n))
    exact = fwd.shape == bwd.shape and bool(np.array_equal(fwd, bwd))

    # Berapa edge yang tidak punya pasangan balik (berbasis himpunan unik)
    fwd_u = np.unique(fwd)
    bwd_u = np.unique(bwd)
    missing = np.setdiff1d(fwd_u, bwd_u, assume_unique=True).size

    return {
        "exact_symmetric": exact,
        "n_unique_directed": int(fwd_u.size),
        "n_without_reverse": int(missing),
        "has_duplicates": bool(fwd_u.size != fwd.size),
        "n_duplicate_rows": int(fwd.size - fwd_u.size),
    }


def fraud_stats(src, dst, y, n, use_total_degree):
    """
    Statistik heterophily tingkat node untuk node anomali.
    """
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    diff = (y[src_t] != y[dst_t]).float()
    ones = torch.ones_like(diff)

    deg = torch.zeros(n)
    het = torch.zeros(n)
    deg.index_add_(0, src_t, ones)
    het.index_add_(0, src_t, diff)
    if use_total_degree:
        deg.index_add_(0, dst_t, ones)
        het.index_add_(0, dst_t, diff)

    fraud = (y == 1)
    f_deg, f_het = deg[fraud], het[fraud]
    no_nbr = (f_deg == 0)
    ratio = torch.zeros_like(f_deg)
    ratio[~no_nbr] = f_het[~no_nbr] / f_deg[~no_nbr]

    return {
        "pct_no_neighbor": 100.0 * no_nbr.float().mean().item(),
        "pct_ratio_gt_50": 100.0 * (ratio > 0.5).float().mean().item(),
    }


def edge_heterophily(src, dst, y):
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    return (y[src_t] != y[dst_t]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../../datasets")
    ap.add_argument("--num_layers", type=int, default=2,
                    help="Derajat polinomial Beta Wavelet (icl.conf.yaml yelp = 2)")
    args = ap.parse_args()

    het_path = os.path.join(args.data_dir, "hetero", "yelp")
    gh = load_graphs(het_path)[0][0]
    n = gh.num_nodes()
    y = gh.ndata["label"].long().view(-1)
    n_fraud = int((y == 1).sum())

    print("num_nodes:", n, "| n_fraud:", n_fraud,
          "| etypes:", [ce[1] for ce in gh.canonical_etypes])

    # ------------------------------------------------------------------
    # 1. Simetri penyimpanan per relasi
    # ------------------------------------------------------------------
    
    # mastiin tiap node yg berhubungan itu tersimpan 2 arah di DGL. krn kalau ada yg 1 arah berarti ada yg salah.
    sep("1. Apakah tiap relasi tersimpan DUA ARAH?")
    print("%-10s | %10s | %8s | %10s | %10s | %10s" % (
        "relasi", "edge", "simetris", "unik", "tanpa balik", "duplikat"))
    print("-" * 78)

    raw = {}
    for ce in gh.canonical_etypes:
        s, d = gh.edges(etype=ce)
        s = s.numpy()
        d = d.numpy()
        raw[ce[1]] = (s, d)
        r = symmetry_report(s, d, n)
        print("%-10s | %10d | %8s | %10d | %11d | %10d" % (
            ce[1], s.size, r["exact_symmetric"], r["n_unique_directed"],
            r["n_without_reverse"], r["n_duplicate_rows"]))

    # ------------------------------------------------------------------
    # 2. Efek praproses baseline pada tiap relasi
    # ------------------------------------------------------------------

    # Harus pastikan derajat nya benar krn apakah dipakai untuk hitung normalisasi.
    sep("2. Efek to_bidirected -> remove_self_loop -> add_self_loop per relasi")
    print("%-10s | %10s | %12s | %12s | %12s" % (
        "relasi", "mentah", "self-loop", "bidirected", "final"))
    print("-" * 78)

    processed = {}
    for name, (s, d) in raw.items():
        n_self = int((s == d).sum())
        g_r = dgl.graph((torch.as_tensor(s), torch.as_tensor(d)), num_nodes=n)
        g_b = dgl.to_bidirected(g_r)
        g_f = dgl.add_self_loop(dgl.remove_self_loop(g_b))
        processed[name] = g_f
        print("%-10s | %10d | %12d | %12d | %12d" % (
            name, s.size, n_self, g_b.num_edges(), g_f.num_edges()))

    # ------------------------------------------------------------------
    # 3. Derajat out-only vs derajat total (validasi angka B1)
    # ------------------------------------------------------------------

    # hitung di edge mentah
    sep("3. Statistik fraud: metode LAMA (src saja) vs derajat TOTAL")
    print("%-10s | %-28s | %-28s" % ("relasi", "LAMA (out-degree)", "TOTAL (src+dst)"))
    print("%-10s | %13s %14s | %13s %14s" % (
        "", "%rasio>50", "%tanpa nbr", "%rasio>50", "%tanpa nbr"))
    print("-" * 78)

    for name, (s, d) in raw.items():
        old = fraud_stats(s, d, y, n, use_total_degree=False)
        new = fraud_stats(s, d, y, n, use_total_degree=True)
        print("%-10s | %12.2f%% %13.2f%% | %12.2f%% %13.2f%%" % (
            name, old["pct_ratio_gt_50"], old["pct_no_neighbor"],
            new["pct_ratio_gt_50"], new["pct_no_neighbor"]))

    # ------------------------------------------------------------------
    # 4. H(r) sebelum vs sesudah praproses
    # ------------------------------------------------------------------

    # yg final hrsnya lbh rendah soalnya ketambahan self-loop homofilik
    sep("4. H(r) pada edge mentah vs graf yang benar-benar masuk ke filter")
    print("%-10s | %14s | %14s" % ("relasi", "H(r) mentah", "H(r) final"))
    print("-" * 78)
    for name, (s, d) in raw.items():
        h_raw = edge_heterophily(s, d, y)
        fs, fd = processed[name].edges()
        h_fin = edge_heterophily(fs.numpy(), fd.numpy(), y)
        print("%-10s | %14.4f | %14.4f" % (name, h_raw, h_fin))

    # ------------------------------------------------------------------
    # 5. Populasi kolom nol setelah praproses penuh (terisolasi)
    # ------------------------------------------------------------------
    sep("5. Berapa node menghasilkan blok NOL?")

    print("%-10s | %14s | %14s | %14s" % (
        "relasi", "node isolated", "fraud isolated", "%fraud"))
    print("-" * 78)
    for name, g_f in processed.items():
        deg = g_f.in_degrees()
        iso = (deg == 1)
        iso_fraud = int((iso & (y == 1)).sum())
        print("%-10s | %14d | %14d | %13.2f%%" % (
            name, int(iso.sum()), iso_fraud, 100.0 * iso_fraud / n_fraud))


if __name__ == "__main__":
    main()