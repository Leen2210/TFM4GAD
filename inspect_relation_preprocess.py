"""
inspect_relation_preprocess.py — diagnostik F2, sebelum menulis loader RA-TFM4GAD.

Melengkapi inspect_yelp_graph.py (JANGAN ubah skrip itu; ia bukti untuk angka
yang sudah tercatat di REVISI-PROPOSAL-DAN-LAPORAN.md item B1).

Menjawab empat hal:
  1. Apakah tiap etype tersimpan dua arah? (uji simetri langsung, bukan inferensi)
  2. Berapa edge sebelum vs sesudah to_bidirected / remove_self_loop / add_self_loop?
  3. Apakah pengukuran lama (derajat = out-degree saja) sama dengan derajat total?
     -> ini yang menentukan apakah 81,77% "fraud tanpa tetangga R-U-R" valid.
  4. Berapa node yang benar-benar menghasilkan blok nol setelah praproses penuh?

"""

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
    """
    Uji apakah himpunan edge simetris.

    'exact' membandingkan termasuk multiplisitas: himpunan (src,dst) terurut
    harus identik dengan himpunan (dst,src) terurut. Kalau True, relasi
    benar-benar tersimpan dua arah dan out-degree == in-degree untuk tiap node.
    """
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

    use_total_degree=False meniru inspect_yelp_graph.py (hanya index_add_ pada src).
    use_total_degree=True menghitung derajat dari src DAN dst.
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

    print("\nBaca: 'simetris'=True berarti out-degree == in-degree untuk setiap node,")
    print("sehingga pengukuran lama (berbasis src saja) SAH. 'simetris'=False berarti")
    print("angka %tanpa-tetangga di REVISI item B1 harus dikoreksi (lihat bagian 3).")

    # ------------------------------------------------------------------
    # 2. Efek rantai praproses baseline pada tiap relasi
    # ------------------------------------------------------------------
    sep("2. Efek to_bidirected -> remove_self_loop -> add_self_loop per relasi")
    print("Urutan ini menyalin persis dataloader.py baris load_data().\n")
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

    print("\nJika 'bidirected' == 'mentah', relasi memang sudah dua arah dan")
    print("to_bidirected tidak menambah apa pun. Jika LEBIH KECIL, to_bidirected")
    print("mengoalisasi edge duplikat (jadi graf sederhana) — ini mengubah bobot")
    print("agregasi karena poly_conv menormalkan dengan D^-1/2.")
    print("Jika LEBIH BESAR, ada edge satu arah yang dilengkapi pasangannya.")

    # ------------------------------------------------------------------
    # 3. Derajat out-only vs derajat total (validasi angka B1)
    # ------------------------------------------------------------------
    sep("3. Statistik fraud: metode LAMA (src saja) vs derajat TOTAL")
    print("Dihitung pada edge MENTAH, agar sebanding dengan inspect_yelp_graph.py.\n")
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

    print("\nKolom LAMA harus cocok dengan angka di REVISI item B1")
    print("(rsr 92,62 / 0,10 · rtr 95,34 / 1,09 · rur 1,35 / 81,77).")
    print("Jika kolom TOTAL berbeda, kolom TOTAL yang benar dan B1 perlu dikoreksi.")

    # ------------------------------------------------------------------
    # 4. H(r) sebelum vs sesudah praproses
    # ------------------------------------------------------------------
    sep("4. H(r) pada edge mentah vs graf yang benar-benar masuk ke filter")
    print("%-10s | %14s | %14s" % ("relasi", "H(r) mentah", "H(r) final"))
    print("-" * 78)
    for name, (s, d) in raw.items():
        h_raw = edge_heterophily(s, d, y)
        fs, fd = processed[name].edges()
        h_fin = edge_heterophily(fs.numpy(), fd.numpy(), y)
        print("%-10s | %14.4f | %14.4f" % (name, h_raw, h_fin))

    print("\nH(r) final PASTI lebih rendah: self-loop selalu homofilik dan pada")
    print("relasi jarang jumlahnya mendominasi. Untuk laporan, kutip H(r) MENTAH")
    print("sebagai karakteristik dataset — itu yang sebanding dengan literatur.")

    # ------------------------------------------------------------------
    # 5. Populasi kolom nol setelah praproses penuh
    # ------------------------------------------------------------------
    sep("5. Berapa node menghasilkan blok NOL? (mekanisme hasil negatif §4e)")
    n_zero_cols = 32 * args.num_layers
    n_tot_cols = 32 * (args.num_layers + 1)
    print("Setelah add_self_loop, node tanpa tetangga nyata punya in_degree == 1,")
    print("sehingga A = I dan filter Beta berderajat >= 1 keluar TEPAT nol.")
    print("Dengan num_layers=%d: %d dari %d kolom blok relasi itu bernilai nol.\n"
          % (args.num_layers, n_zero_cols, n_tot_cols))

    print("%-10s | %14s | %14s | %14s" % (
        "relasi", "node isolated", "fraud isolated", "%fraud"))
    print("-" * 78)
    for name, g_f in processed.items():
        deg = g_f.in_degrees()
        iso = (deg == 1)
        iso_fraud = int((iso & (y == 1)).sum())
        print("%-10s | %14d | %14d | %13.2f%%" % (
            name, int(iso.sum()), iso_fraud, 100.0 * iso_fraud / n_fraud))

    sep("RINGKASAN UNTUK KEPUTUSAN F2")
    print("1. Kalau bagian 1 semua True  -> angka B1 sah, lanjut tanpa koreksi.")
    print("2. Kalau bagian 2 kolom 'bidirected' == 'mentah' untuk SEMUA relasi,")
    print("   praproses per relasi aman dan tidak menggeser H(r) yang sudah dicatat.")
    print("3. Angka di bagian 5 adalah prediksi kuantitatif untuk Eksperimen 2 A1.")


if __name__ == "__main__":
    main()