"""
inspect_derived_numbers.py — melengkapi dua skrip diagnostik yang sudah ada.

Dua skrip terdahulu mencetak angka DASAR:
  - inspect_yelp_graph.py            : jumlah edge per relasi, H(r), statistik fraud
  - inspect_relation_preprocess.py   : simetri, efek praproses, node terisolasi

Skrip ini mencetak dua hal yang belum pernah dihitung skrip manapun:

  BAGIAN A  Seluruh angka TURUNAN yang dipakai Bab III tetapi selama ini
            dihitung tangan (pangsa edge, jumlah tak-berarah, tumpang tindih
            lintas relasi, rasio anomali kelompok terisolasi).
            Tumpang tindih 178.695 dihitung lewat irisan himpunan pasangan,
            bukan lewat pengurangan, sehingga menjadi verifikasi independen.

  BAGIAN B  Distribusi RASIO BOBOT NORMALISASI per relasi, yaitu seberapa besar
            kontribusi seorang tetangga teredam ketika normalisasi derajat
            dihitung dari graf gabungan alih-alih dari subgraf relasinya sendiri.
            Angka ini dibutuhkan oleh Bab III bagian Analisis Masalah dan oleh
            gambar ilustrasi peredaman.

CATATAN PENTING
  Angka 69 yang sempat tercatat di berkas revisi BUKAN faktor peredaman.
  69,0 adalah rasio jumlah edge R-S-R terhadap R-U-R. Faktor peredaman
  ditentukan oleh derajat per-node, dan itulah yang dihitung Bagian B.

Pemakaian:
  python inspect_derived_numbers.py --data_dir ../../datasets
  python inspect_derived_numbers.py --data_dir ../../datasets --dump_csv rasio_rur.csv
"""

import argparse
import os

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs

NAMA = {"net_rur": "R-U-R", "net_rtr": "R-T-R", "net_rsr": "R-S-R"}
URUT = ["net_rsr", "net_rtr", "net_rur"]


def sep(judul):
    print("\n" + "=" * 78)
    print(judul)
    print("=" * 78)


def praproses(g):
    """Rantai praproses identik dengan baseline (item C6)."""
    try:
        g = dgl.to_bidirected(g)
    except dgl.DGLError:
        # sebagian versi DGL menolak multigraph; jadikan simple graph dulu
        g = dgl.to_bidirected(dgl.to_simple(g))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g


def pasangan_tak_berarah(src, dst, n):
    """Kode unik untuk tiap pasangan {i, j}, tanpa arah, tanpa duplikat."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    bukan_self = src != dst
    a = np.minimum(src, dst)[bukan_self]
    b = np.maximum(src, dst)[bukan_self]
    return np.unique(a * np.int64(n) + b), int((~bukan_self).sum())


def heterophily_edge(src, dst, y):
    s = torch.as_tensor(np.asarray(src), dtype=torch.long)
    d = torch.as_tensor(np.asarray(dst), dtype=torch.long)
    return (y[s] != y[d]).float().mean().item()


def ringkas(x):
    q = np.percentile(x, [0, 25, 50, 75, 100])
    return dict(min=q[0], q1=q[1], median=q[2], q3=q[3], max=q[4], mean=float(np.mean(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../../datasets")
    ap.add_argument("--num_layers", type=int, default=2,
                    help="Derajat C bank Beta Wavelet (icl.conf.yaml yelp = 2)")
    ap.add_argument("--dump_csv", type=str, default="",
                    help="Opsional: simpan seluruh rasio bobot R-U-R ke CSV")
    args = ap.parse_args()

    gh = load_graphs(os.path.join(args.data_dir, "hetero", "yelp"))[0][0]
    gm = load_graphs(os.path.join(args.data_dir, "yelp"))[0][0]

    n = gh.num_nodes()
    y = gh.ndata["label"].long().view(-1)
    n_fraud = int((y == 1).sum())

    mentah = {}
    for ce in gh.canonical_etypes:
        r = ce[1]
        s, d = gh.edges(etype=ce)
        mentah[r] = (s.numpy(), d.numpy())

    olah = {r: praproses(dgl.graph((torch.as_tensor(mentah[r][0]),
                                    torch.as_tensor(mentah[r][1])),
                                   num_nodes=n)) for r in mentah}
    gm_olah = praproses(gm)

    # ------------------------------------------------------------------ DASAR
    sep("0. ANGKA DASAR (pembanding terhadap skrip terdahulu)")
    print("num_nodes            : %d" % n)
    print("n_fraud              : %d" % n_fraud)
    print("rasio anomali        : %.4f%%   (pembulatan terpublikasi: 14,5%%)" %
          (100.0 * n_fraud / n))
    print("edge berkas homogen  : %d (tersimpan, termasuk self-loop)" % gm.num_edges())
    print()
    print("%-8s | %12s | %12s | %10s | %10s" %
          ("relasi", "edge mentah", "edge akhir", "H(r) mentah", "H(r) akhir"))
    print("-" * 78)
    for r in URUT:
        s, d = mentah[r]
        gp = olah[r]
        sp, dp = gp.edges()
        print("%-8s | %12d | %12d | %11.4f | %10.4f" %
              (NAMA[r], len(s), gp.num_edges(),
               heterophily_edge(s, d, y), heterophily_edge(sp.numpy(), dp.numpy(), y)))

    # -------------------------------------------------------------- BAGIAN A1
    sep("A1. PANGSA EDGE PER RELASI")
    total_berarah = sum(len(mentah[r][0]) for r in mentah)
    print("Total edge berarah ketiga relasi : %d" % total_berarah)
    print()
    print("%-8s | %12s | %8s" % ("relasi", "edge berarah", "pangsa"))
    print("-" * 40)
    for r in URUT:
        m = len(mentah[r][0])
        print("%-8s | %12d | %7.1f%%" % (NAMA[r], m, 100.0 * m / total_berarah))

    # -------------------------------------------------------------- BAGIAN A2
    sep("A2. TUMPANG TINDIH LINTAS RELASI (verifikasi independen 178.695)")
    pasangan, jml_self = {}, {}
    for r in mentah:
        pasangan[r], jml_self[r] = pasangan_tak_berarah(mentah[r][0], mentah[r][1], n)
    for r in URUT:
        print("%-8s : %10d pasangan tak-berarah  (self-loop bawaan: %d)" %
              (NAMA[r], len(pasangan[r]), jml_self[r]))

    jumlah_terpisah = sum(len(pasangan[r]) for r in pasangan)
    gabungan = np.unique(np.concatenate([pasangan[r] for r in pasangan]))
    tumpang = jumlah_terpisah - len(gabungan)

    print()
    print("Jumlah pasangan bila ketiga relasi dijumlahkan : %d" % jumlah_terpisah)
    print("Pasangan unik setelah deduplikasi lintas relasi: %d" % len(gabungan))
    print("Tumpang tindih (muncul di > 1 relasi)          : %d" % tumpang)
    print()
    print("Pembanding dari berkas homogen:")
    print("  edge tersimpan %d  - self-loop %d  = %d berarah = %d pasangan"
          % (gm.num_edges(), n, gm.num_edges() - n, (gm.num_edges() - n) // 2))
    cocok = (gm.num_edges() - n) // 2 == len(gabungan)
    print("  cocok dengan hasil deduplikasi : %s" % ("YA" if cocok else "TIDAK"))
    print("  angka terpublikasi README      : 3846979")

    # -------------------------------------------------------------- BAGIAN A3
    sep("A3. RASIO ANOMALI: KELOMPOK TERISOLASI VERSUS BERTETANGGA")
    print("Terisolasi = in_degree == 1 setelah add_self_loop,")
    print("yaitu tidak punya tetangga selain dirinya sendiri.\n")
    print("%-8s | %10s %8s %8s | %10s %8s %8s | %6s" %
          ("relasi", "n isolasi", "fraud", "rasio", "n punya", "fraud", "rasio", "lipat"))
    print("-" * 78)
    for r in URUT:
        deg = olah[r].in_degrees()
        iso = (deg == 1)
        n_iso = int(iso.sum())
        n_con = n - n_iso
        f_iso = int((iso & (y == 1)).sum())
        f_con = n_fraud - f_iso
        r_iso = 100.0 * f_iso / n_iso if n_iso else float("nan")
        r_con = 100.0 * f_con / n_con if n_con else float("nan")
        lipat = r_iso / r_con if r_con else float("nan")
        print("%-8s | %10d %8d %7.2f%% | %10d %8d %7.2f%% | %5.2fx" %
              (NAMA[r], n_iso, f_iso, r_iso, n_con, f_con, r_con, lipat))
    print("\nRasio anomali seluruh dataset: %.2f%%" % (100.0 * n_fraud / n))

    # -------------------------------------------------------------- BAGIAN A4
    sep("A4. KOLOM BERNILAI NOL PADA NODE TERISOLASI")
    nol = 32 * args.num_layers
    tot = 32 * (args.num_layers + 1)
    print("Dengan C = num_layers = %d, satu blok xnbr berdimensi %d." % (args.num_layers, tot))
    print("Pada node terisolasi, A = I sehingga L = 0 dan filter berderajat >= 1")
    print("keluar tepat nol: %d dari %d kolom." % (nol, tot))

    # ------------------------------------------------------------- BAGIAN B
    sep("B. RASIO BOBOT NORMALISASI — SUBGRAF SENDIRI VERSUS GRAF GABUNGAN")
    print("Untuk tiap edge (i, j) pada relasi r:")
    print("  w_gab(i,j) = 1 / sqrt( d_gab(i) * d_gab(j) )")
    print("  w_sub(i,j) = 1 / sqrt( d_sub(i) * d_sub(j) )")
    print("  rasio      = w_sub / w_gab")
    print("Rasio > 1 berarti kontribusi tetangga itu TEREDAM pada graf gabungan.\n")

    d_gab = gm_olah.in_degrees().double()
    print("%-8s | %8s | %8s | %8s | %8s | %8s | %8s" %
          ("relasi", "min", "Q1", "median", "Q3", "maks", "rata2"))
    print("-" * 78)
    simpan = None
    for r in URUT:
        kode = pasangan[r]
        i = torch.as_tensor(kode // np.int64(n), dtype=torch.long)
        j = torch.as_tensor(kode % np.int64(n), dtype=torch.long)
        d_sub = olah[r].in_degrees().double()
        rasio = torch.sqrt((d_gab[i] * d_gab[j]) / (d_sub[i] * d_sub[j])).numpy()
        s = ringkas(rasio)
        print("%-8s | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f" %
              (NAMA[r], s["min"], s["q1"], s["median"], s["q3"], s["max"], s["mean"]))
        if r == "net_rur":
            simpan = (kode, rasio, s)

    kode, rasio_rur, s_rur = simpan
    print()
    print("ANGKA UNTUK BAB III (relasi R-U-R):")
    print("  median rasio bobot        : %.2f" % s_rur["median"])
    print("  rentang antar-kuartil     : %.2f - %.2f" % (s_rur["q1"], s_rur["q3"]))
    print("  jumlah edge yang diukur   : %d pasangan" % len(rasio_rur))
    print()
    print("PEMBANDING — angka 69 yang KELIRU dipakai sebagai faktor peredaman:")
    print("  rasio jumlah edge R-S-R : R-U-R = %.1f" %
          (len(mentah["net_rsr"][0]) / len(mentah["net_rur"][0])))
    print("  faktor peredaman sebenarnya (median) = %.2f" % s_rur["median"])
    print("  keduanya besaran berbeda dan tidak boleh dipertukarkan.")

    # satu contoh node nyata untuk gambar ilustrasi
    idx = int(np.argmin(np.abs(rasio_rur - s_rur["median"])))
    i_c = int(kode[idx] // n)
    j_c = int(kode[idx] % n)
    print()
    print("CONTOH NODE UNTUK GAMBAR ILUSTRASI (rasio paling dekat median):")
    print("  pasangan node   : i = %d, j = %d" % (i_c, j_c))
    print("  derajat gabungan: d(i) = %d, d(j) = %d" % (int(d_gab[i_c]), int(d_gab[j_c])))
    d_sub_rur = olah["net_rur"].in_degrees().double()
    print("  derajat R-U-R   : d(i) = %d, d(j) = %d" % (int(d_sub_rur[i_c]), int(d_sub_rur[j_c])))
    print("  bobot gabungan  : %.6e" % (1.0 / np.sqrt(float(d_gab[i_c]) * float(d_gab[j_c]))))
    print("  bobot subgraf   : %.6e" % (1.0 / np.sqrt(float(d_sub_rur[i_c]) * float(d_sub_rur[j_c]))))
    print("  rasio           : %.2f" % rasio_rur[idx])

    if args.dump_csv:
        np.savetxt(args.dump_csv,
                   np.column_stack([kode // np.int64(n), kode % np.int64(n), rasio_rur]),
                   delimiter=",", header="node_i,node_j,rasio_bobot", comments="",
                   fmt=["%d", "%d", "%.6f"])
        print("\nSeluruh rasio R-U-R disimpan ke %s" % args.dump_csv)

    sep("SELESAI")
    print("Seluruh angka di atas sebelumnya dihitung tangan atau belum pernah dihitung.")
    print("Salin keluaran ini ke berkas revisi agar Bab III punya jejak yang lengkap.")


if __name__ == "__main__":
    main()