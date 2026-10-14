"""
exp4_splits.py — split bersarang untuk Eksperimen 4 (robustness ukuran in-context set).

Dua kegunaan:
  1. Dijalankan SEKALI dari root repo untuk membangkitkan file split:
         python exp4_splits.py --data_dir ./datasets --out splits/yelp_exp4_nested.json
  2. Diimpor oleh kedua main.py (hanya bila --split_file diberikan) untuk
     mengganti train mask resmi dengan in-context set berukuran 50/100/200.

Definisi, untuk setiap split resmi GADBench s (kolom mask 10-19 file HOMOGEN):
  n=50  : 10 anomali + 40 normal, subsampel acak dari train resmi kolom s
  n=100 : train resmi kolom s, persis
  n=200 : train resmi + val resmi kolom s (40 anomali + 160 normal)
  test  : test resmi kolom s — TIDAK disimpan di file dan TIDAK diubah

Akibatnya 50 ⊂ 100 ⊂ 200 dan test identik di ketiga ukuran: satu-satunya
variabel adalah jumlah node berlabel di in-context set. Augmentasi fitur
tidak memakai label, jadi matriks fitur juga identik di ketiga ukuran.
"""
import argparse
import datetime
import json
import os

import numpy as np

SIZES = (50, 100, 200)
ANOMALY_FRACTION = 0.2    # 20:80, mengikuti protokol semi-supervised GADBench
OFFICIAL_FIRST_COL = 10   # kolom 10-19 = protokol 100 label; kolom 0-9 = fully-supervised


def n_anomalies(n):
    return int(round(n * ANOMALY_FRACTION))


def build_nested_splits(y, train, val, test, seed=0):
    """
    y                 : (N,) label 0/1
    train, val, test  : (N, S) bool, mask resmi kolom 10-19
    Return dict siap ditulis ke JSON. Setiap pemeriksaan yang gagal langsung raise.
    """
    y = np.asarray(y).astype(int).ravel()
    n_nodes, n_splits = train.shape
    rng = np.random.default_rng(seed)
    splits = []
    for s in range(n_splits):
        tr = np.flatnonzero(train[:, s])
        va = np.flatnonzero(val[:, s])
        te = np.flatnonzero(test[:, s])
        tag = f"split {s + 1}"

        # ---- prasyarat pada split resmi ----
        if tr.size != 100 or y[tr].sum() != n_anomalies(100):
            raise RuntimeError(f"{tag}: train resmi bukan 100 node / 20 anomali "
                               f"({tr.size} / {y[tr].sum()}). Apakah kolom 10-19 yang dipakai?")
        if va.size != 100 or y[va].sum() != n_anomalies(100):
            raise RuntimeError(f"{tag}: val resmi bukan 100 node / 20 anomali "
                               f"({va.size} / {y[va].sum()})")
        if (np.intersect1d(tr, va).size or np.intersect1d(tr, te).size
                or np.intersect1d(va, te).size):
            raise RuntimeError(f"{tag}: train/val/test resmi beririsan")

        # ---- tiga ukuran ----
        idx = {100: tr}                                    # train resmi, persis
        anom = rng.choice(tr[y[tr] == 1], size=n_anomalies(50), replace=False)
        norm = rng.choice(tr[y[tr] == 0], size=50 - n_anomalies(50), replace=False)
        idx[50] = np.sort(np.concatenate([anom, norm]))    # subsampel bertingkat
        idx[200] = np.sort(np.concatenate([tr, va]))       # train + val resmi

        # ---- pemeriksaan hasil ----
        for n in SIZES:
            if idx[n].size != n or np.unique(idx[n]).size != n:
                raise RuntimeError(f"{tag}: ukuran {n} berisi {idx[n].size} node / ada duplikat")
            if y[idx[n]].sum() != n_anomalies(n):
                raise RuntimeError(f"{tag}: ukuran {n} berisi {y[idx[n]].sum()} anomali, "
                                   f"seharusnya {n_anomalies(n)}")
            if np.intersect1d(idx[n], te).size:
                raise RuntimeError(f"{tag}: ukuran {n} bocor ke test")
        if not (np.isin(idx[50], idx[100]).all() and np.isin(idx[100], idx[200]).all()):
            raise RuntimeError(f"{tag}: sifat bersarang 50 ⊂ 100 ⊂ 200 tidak terpenuhi")

        splits.append({
            "split": s + 1,
            "source_column": OFFICIAL_FIRST_COL + s,
            "test_size": int(te.size),
            **{str(n): idx[n].tolist() for n in SIZES},
        })

    return {
        "description": "Eksperimen 4: in-context set bersarang 50 ⊂ 100 ⊂ 200 per split resmi "
                       "GADBench; test = test resmi, identik di ketiga ukuran",
        "n_nodes": int(n_nodes),
        "n_splits": int(n_splits),
        "sizes": list(SIZES),
        "anomaly_fraction": ANOMALY_FRACTION,
        "seed": int(seed),
        "rng": f"numpy.random.default_rng (numpy {np.__version__})",
        "splits": splits,
    }


def load_train_masks(path, n_labels, y, train, test, log=print):
    """
    Ganti train mask resmi dengan in-context set berukuran n_labels.

    y, train, test : array numpy dari loader (train/test = mask resmi kolom 10-19)
    Return (N, S) bool numpy. Mask test TIDAK diubah.
    """
    with open(path) as f:
        data = json.load(f)
    y = np.asarray(y).astype(int).ravel()
    train = np.asarray(train, dtype=bool)
    test = np.asarray(test, dtype=bool)
    n_nodes, n_splits = train.shape

    if n_labels not in data["sizes"]:
        raise SystemExit(f"n_labels={n_labels} tidak ada di {path} (pilihan {data['sizes']})")
    if data["n_nodes"] != n_nodes or data["n_splits"] != n_splits or len(data["splits"]) != n_splits:
        raise SystemExit(f"{path} dibuat untuk {data['n_nodes']} node / {data['n_splits']} split, "
                         f"data yang dimuat {n_nodes} node / {n_splits} split")

    out = np.zeros_like(train, dtype=bool)
    for s, rec in enumerate(data["splits"]):
        tag = f"split {s + 1}"
        # Sidik jari: ukuran 100 di file harus sama persis dengan train resmi yang baru
        # dimuat. Gagal bila file milik dataset lain, atau bila n_anomalies di config
        # bukan 20 (loader lalu diam-diam memakai kolom 0-9 fully-supervised).
        if not np.array_equal(np.flatnonzero(train[:, s]), np.asarray(rec["100"])):
            raise SystemExit(f"{tag}: train resmi tidak cocok dengan {path}. "
                             "Periksa n_anomalies=20 di config dan file dataset yang dipakai.")
        if int(test[:, s].sum()) != rec["test_size"]:
            raise SystemExit(f"{tag}: ukuran test {int(test[:, s].sum())} != {rec['test_size']}")
        idx = np.asarray(rec[str(n_labels)], dtype=int)
        if idx.size != n_labels or y[idx].sum() != n_anomalies(n_labels):
            raise SystemExit(f"{tag}: in-context set {idx.size} node / {y[idx].sum()} anomali, "
                             f"seharusnya {n_labels} / {n_anomalies(n_labels)}")
        if test[idx, s].any():
            raise SystemExit(f"{tag}: in-context set beririsan dengan test")
        out[idx, s] = True

    log(f"[exp4] split_file={path} | n_labels={n_labels} ({n_anomalies(n_labels)} anomali) | "
        f"{n_splits} split | test tetap {int(test[:, 0].sum())} node | seed split={data['seed']}")
    return out


def _cli():
    p = argparse.ArgumentParser(description="Bangkitkan split bersarang Eksperimen 4")
    p.add_argument('--data_dir', type=str, default='./datasets')
    p.add_argument('--dataset', type=str, default='yelp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='splits/yelp_exp4_nested.json')
    args = p.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} sudah ada. Split cukup dibangkitkan sekali; "
                         "hapus manual hanya bila memang ingin membuat ulang.")

    from dgl.data.utils import load_graphs
    src = os.path.join(args.data_dir, args.dataset)            # file HOMOGEN
    g = load_graphs(src)[0][0]
    y = g.ndata['label'].numpy()
    c = OFFICIAL_FIRST_COL
    train = g.ndata['train_masks'][:, c:].bool().numpy()
    val = g.ndata['val_masks'][:, c:].bool().numpy()
    test = g.ndata['test_masks'][:, c:].bool().numpy()

    data = build_nested_splits(y, train, val, test, seed=args.seed)
    data["source"] = src
    data["created"] = datetime.date.today().isoformat()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(data, f)

    yi = np.asarray(y).astype(int).ravel()
    for rec in data["splits"]:
        parts = " | ".join(f"n={n}: {len(rec[str(n)])} ({yi[rec[str(n)]].sum()} anomali)"
                           for n in SIZES)
        print(f"split {rec['split']:>2} (kolom {rec['source_column']}): {parts} | test {rec['test_size']}")
    print(f"Semua pemeriksaan lolos. Ditulis ke {args.out} (seed={args.seed}).")


if __name__ == "__main__":
    _cli()