"""
check_exp4_splits.py — memeriksa file split Eksperimen 4 pada DATA ASLI.

Empat bagian:
  A. Audit isi file terhadap mask resmi. Ditulis ulang tanpa memakai fungsi
     exp4_splits.py, supaya bug yang sama tidak lolos di dua tempat.
  B. Determinisme: membangkitkan ulang ke folder sementara dengan seed sama
     (harus identik) dan seed berbeda (harus berbeda).
  C. Pengaman load_train_masks dengan tensor asli dari dataloader baseline.
  D. Pengaman argumen di kedua main.py (berhenti sebelum memuat data).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'LOLOS' if ok else 'GAGAL'}] {name}" + (f" — {detail}" if detail else ""))


def expect_exit(name, fn, must_contain):
    try:
        fn()
    except SystemExit as e:
        check(name, must_contain in str(e), f"berhenti: {str(e)[:90]}")
        return
    except Exception as e:  # noqa: BLE001
        check(name, False, f"error lain, bukan pengaman: {e!r}"[:120])
        return
    check(name, False, "TIDAK berhenti")


def official_masks(data_dir, dataset='yelp'):
    from dgl.data.utils import load_graphs
    g = load_graphs(os.path.join(data_dir, dataset))[0][0]
    y = g.ndata['label'].long().numpy().ravel()
    tr, va, te = (g.ndata[k][:, 10:].bool().numpy()
                  for k in ('train_masks', 'val_masks', 'test_masks'))
    return y, tr, va, te


def audit(data, y, tr, va, te):
    n_nodes, n_splits = tr.shape
    check("jumlah node dan split cocok dengan data asli",
          data['n_nodes'] == n_nodes and data['n_splits'] == n_splits
          and len(data['splits']) == n_splits,
          f"{data['n_nodes']} node, {len(data['splits'])} split")
    for s, rec in enumerate(data['splits']):
        i50, i100, i200 = (np.asarray(rec[k], dtype=int) for k in ('50', '100', '200'))
        o_tr, o_va, o_te = (np.flatnonzero(m[:, s]) for m in (tr, va, te))
        conds = {
            "terurut & unik": all(np.all(np.diff(i) > 0) for i in (i50, i100, i200)),
            "indeks dalam rentang": all(i.min() >= 0 and i.max() < n_nodes for i in (i50, i100, i200)),
            "ukuran 50/100/200": (i50.size, i100.size, i200.size) == (50, 100, 200),
            "anomali 10/20/40": (y[i50].sum(), y[i100].sum(), y[i200].sum()) == (10, 20, 40),
            "100 = train resmi": np.array_equal(i100, o_tr),
            "200 = train ∪ val resmi": np.array_equal(i200, np.union1d(o_tr, o_va)),
            "50 ⊂ 100": np.isin(i50, i100).all(),
            "tidak bocor ke test": all(np.intersect1d(i, o_te).size == 0 for i in (i50, i100, i200)),
            "test 45.754 dan tercatat benar": rec['test_size'] == o_te.size == n_nodes - 200,
            "kolom sumber benar": rec['source_column'] == 10 + s,
        }
        bad = [k for k, v in conds.items() if not v]
        check(f"split {s + 1:>2}", not bad, "gagal: " + ", ".join(bad) if bad else
              f"anomali {y[i50].sum()}/{y[i100].sum()}/{y[i200].sum()}, test {o_te.size}")


def corrupt_with_test_node(data, y, te):
    """Salinan file dengan satu node normal di ukuran 50 split 1 diganti node normal dari test."""
    bad = json.loads(json.dumps(data))
    lst = bad['splits'][0]['50']
    pos = next(i for i, v in enumerate(lst) if y[v] == 0)
    lst[pos] = int(np.flatnonzero(te[:, 0] & (y == 0))[0])
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split_file', default='splits/yelp_exp4_nested.json')
    p.add_argument('--data_dir', default='./datasets')
    args = p.parse_args()
    with open(args.split_file) as f:
        data = json.load(f)
    tmp = tempfile.mkdtemp(prefix='exp4check_')

    print("A. Audit isi file terhadap mask resmi")
    y, tr, va, te = official_masks(args.data_dir)
    audit(data, y, tr, va, te)

    print("B. Determinisme pembangkit")
    for seed, want_same in ((data['seed'], True), (data['seed'] + 1, False)):
        out = os.path.join(tmp, f'seed{seed}.json')
        subprocess.run([sys.executable, os.path.join(ROOT, 'exp4_splits.py'),
                        '--data_dir', args.data_dir, '--seed', str(seed), '--out', out],
                       check=True, capture_output=True)
        with open(out) as f:
            same = json.load(f)['splits'] == data['splits']
        check(f"seed={seed}: hasil {'identik' if same else 'berbeda'} dengan file",
              same == want_same, "harus identik" if want_same else "harus berbeda")

    print("C. Pengaman load_train_masks dengan tensor asli")
    sys.path.insert(0, os.path.join(ROOT, 'TabPFN', 'tfm4gad'))
    sys.path.insert(0, ROOT)
    from dataloader import load_data
    from exp4_splits import load_train_masks, build_nested_splits
    quiet = lambda *_: None  # noqa: E731
    _, _, y_t, tr_t, _, te_t = load_data('yelp', root=args.data_dir, n_anomalies=20)
    ya, tra, tea = y_t.numpy(), tr_t.numpy(), te_t.numpy()
    m100 = load_train_masks(args.split_file, 100, ya, tra, tea, log=quiet)
    check("n=100 identik dengan train resmi dari dataloader", np.array_equal(m100, tra))
    for n in (50, 200):
        m = load_train_masks(args.split_file, n, ya, tra, tea, log=quiet)
        check(f"n={n} lolos semua pemeriksaan", m.sum(0).tolist() == [n] * m.shape[1]
              and not (m & tea).any())
    expect_exit("n_labels tak dikenal (150)",
                lambda: load_train_masks(args.split_file, 150, ya, tra, tea, log=quiet), "tidak ada di")
    _, _, y_f, tr_f, _, te_f = load_data('yelp', root=args.data_dir, n_anomalies=10)
    expect_exit("config fully-supervised (n_anomalies=10)",
                lambda: load_train_masks(args.split_file, 50, y_f.numpy(), tr_f.numpy(),
                                         te_f.numpy(), log=quiet), "tidak cocok")
    if os.path.exists(os.path.join(args.data_dir, 'amazon')):
        _, _, y_a, tr_a, _, te_a = load_data('amazon', root=args.data_dir, n_anomalies=20)
        expect_exit("dataset lain (amazon)",
                    lambda: load_train_masks(args.split_file, 50, y_a.numpy(), tr_a.numpy(),
                                             te_a.numpy(), log=quiet), "dibuat untuk")
    else:
        print("  [LEWATI] dataset lain: datasets/amazon tidak ditemukan")
    bad_path = os.path.join(tmp, 'rusak.json')
    with open(bad_path, 'w') as f:
        json.dump(corrupt_with_test_node(data, y, te), f)
    expect_exit("file split dirusak (1 node test disusupkan)",
                lambda: load_train_masks(bad_path, 50, ya, tra, tea, log=quiet), "beririsan dengan test")
    va2 = va.copy()
    va2[np.flatnonzero(va2[:, 3] & (y == 1))[0], 3] = False
    va2[np.flatnonzero(te[:, 3] & (y == 0))[0], 3] = True
    try:
        build_nested_splits(y, tr, va2, te & ~va2)
        check("prasyarat pembangkit: val bukan 20 anomali", False, "TIDAK berhenti")
    except RuntimeError as e:
        check("prasyarat pembangkit: val bukan 20 anomali", "val resmi" in str(e), str(e)[:90])

    print("D. Pengaman argumen di kedua main.py")
    cases = [
        (['--split_file', 'x.json'], "harus diberikan bersamaan"),
        (['--split_file', 'x.json', '--n_labels', '50'], "Beri --tag"),
        (['--split_file', 'x.json', '--n_labels', '75', '--tag', 'guardtest'], "invalid choice"),
    ]
    for folder in (os.path.join(ROOT, 'TabPFN', 'tfm4gad'), os.path.join(ROOT, 'TabPFN-RA', 'ra-tfm4gad')):
        name = os.path.basename(folder)
        logs_before = set(os.listdir(os.path.join(folder, 'log'))) if os.path.isdir(os.path.join(folder, 'log')) else set()
        for extra, msg in cases:
            r = subprocess.run([sys.executable, 'main.py', '--dataset', 'yelp'] + extra,
                               cwd=folder, capture_output=True, text=True)
            check(f"{name}: {' '.join(extra)}", r.returncode != 0 and msg in (r.stderr + r.stdout),
                  f"exit {r.returncode}")
        logs_after = set(os.listdir(os.path.join(folder, 'log'))) if os.path.isdir(os.path.join(folder, 'log')) else set()
        check(f"{name}: tidak ada log baru yang tercipta", logs_after == logs_before,
              ", ".join(sorted(logs_after - logs_before)))

    print(f"\nRingkasan: {sum(RESULTS)}/{len(RESULTS)} lolos" +
          ("" if all(RESULTS) else "  <-- ADA YANG GAGAL, jangan lanjut ke enam run"))


if __name__ == "__main__":
    main()