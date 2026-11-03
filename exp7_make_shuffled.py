"""Eksperimen 7b dan 7c: salinan data YelpChi dengan SATU relasi diacak, derajat dipertahankan.

Pengacakan pakai double edge swap (networkx): edge (u, v) dan (x, y) ditukar menjadi
(u, x) dan (v, y). Derajat setiap node tetap sama, sedangkan siapa tetangganya menjadi acak.
Jumlah swap = k x jumlah edge tak berarah relasi target (tanpa self-loop).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np

ETYPE = {"rur": "net_rur", "rtr": "net_rtr", "rsr": "net_rsr"}
CACHE_DIRS = ("lap_pe_cache", "pagerank_cache")


# ---------------------------------------------------------------------------
# Logika inti (numpy dan networkx saja, tanpa DGL)
# ---------------------------------------------------------------------------
def undirected_simple_edges(src, dst):
    """Edge tak berarah unik (u < v) tanpa self-loop.

    Setara dengan to_bidirected + remove_self_loop pada praproses subgraf relasi RA.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    u = np.minimum(src[keep], dst[keep])
    v = np.maximum(src[keep], dst[keep])
    if u.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.stack([u, v], axis=1), axis=0)


def degrees(edges, n):
    return np.bincount(edges.ravel(), minlength=n)


def heterophily(edges, y):
    """Proporsi edge yang menghubungkan dua node berlabel berbeda."""
    return float((y[edges[:, 0]] != y[edges[:, 1]]).mean())


def shuffle_preserving_degree(edges, n, k, seed):
    """Double edge swap sebanyak round(k x jumlah edge). Mengembalikan (edge baru, jumlah swap)."""
    nswap = int(round(k * len(edges)))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges.tolist()))
    if nswap > 0:
        nx.double_edge_swap(G, nswap=nswap, max_tries=nswap * 20, seed=seed)
    e = np.array(list(G.edges()), dtype=np.int64).reshape(-1, 2)
    return undirected_simple_edges(e[:, 0], e[:, 1]), nswap


def pct_edges_kept(orig, new, n):
    """Persentase edge asli yang masih ada setelah pengacakan."""
    a = orig[:, 0] * n + orig[:, 1]
    b = new[:, 0] * n + new[:, 1]
    return float(np.intersect1d(a, b).size / a.size * 100)


def check_shuffle(orig, new, n):
    """Guard: jumlah edge sama, derajat identik, tanpa self-loop, tanpa edge ganda."""
    if len(new) != len(orig):
        raise RuntimeError(f"Jumlah edge berubah: {len(orig)} -> {len(new)}")
    if not np.array_equal(degrees(orig, n), degrees(new, n)):
        raise RuntimeError("Derajat berubah setelah pengacakan")
    if (new[:, 0] == new[:, 1]).any():
        raise RuntimeError("Pengacakan menghasilkan self-loop")
    if len(np.unique(new, axis=0)) != len(new):
        raise RuntimeError("Pengacakan menghasilkan edge ganda")


# ---------------------------------------------------------------------------
# Baca dan tulis graf DGL
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_dir", default="datasets",
                    help="Folder data asli (berisi <dataset>, hetero/<dataset>, dan cache)")
    ap.add_argument("--out_root", default="datasets_exp7")
    ap.add_argument("--dataset", default="yelp", choices=["yelp"])
    ap.add_argument("--relation", required=True, choices=list(ETYPE))
    ap.add_argument("--k", type=float, required=True, help="Jumlah swap = k x jumlah edge tak berarah")
    ap.add_argument("--seed", type=int, required=True, help="Seed pengacakan")
    args = ap.parse_args()

    import dgl
    import torch
    from dgl.data.utils import load_graphs, save_graphs

    src_root = Path(args.data_dir).resolve()
    out = Path(args.out_root).resolve() / f"{args.relation}_k{args.k:g}_s{args.seed}"
    if out.exists():
        sys.exit(f"{out} sudah ada. Hapus dulu bila ingin membuat ulang.")
    for name in (args.dataset, *CACHE_DIRS):
        if not (src_root / name).exists():
            sys.exit(f"{src_root / name} tidak ada. Cache harus sudah terbentuk dari run A4.")

    # ---- 1. Baca graf hetero asli ----
    gh = load_graphs(str(src_root / "hetero" / args.dataset))[0][0]
    if len(gh.ntypes) != 1:
        sys.exit(f"Diharapkan satu tipe node, ditemukan {gh.ntypes}")
    ntype = gh.ntypes[0]
    n = gh.num_nodes()
    y = gh.ndata["label"].long().view(-1).numpy()
    targets = [c for c in gh.canonical_etypes if c[1] == ETYPE[args.relation]]
    if len(targets) != 1:
        sys.exit(f"Relasi {ETYPE[args.relation]} tidak ditemukan tepat satu kali: {gh.canonical_etypes}")
    target = targets[0]
    for c in gh.canonical_etypes:
        if len(gh.edges[c].data) > 0:
            print(f"[peringatan] edata pada {c} tidak disalin. Dataloader RA tidak memakainya.")

    # ---- 2. Acak relasi target ----
    s, d = gh.edges(etype=target)
    orig = undirected_simple_edges(s.numpy(), d.numpy())
    t0 = time.time()
    new, nswap = shuffle_preserving_degree(orig, n, args.k, args.seed)
    elapsed = time.time() - t0
    check_shuffle(orig, new, n)
    kept = pct_edges_kept(orig, new, n)

    deg = degrees(orig, n)
    p = float(y.mean())
    q = float(deg[y == 1].sum() / deg.sum())

    # ---- 3. Graf baru: relasi lain disalin persis (tensor dan urutan edge yang sama) ----
    data = {}
    for c in gh.canonical_etypes:
        if c == target:
            u = torch.from_numpy(np.concatenate([new[:, 0], new[:, 1]])).to(gh.idtype)
            v = torch.from_numpy(np.concatenate([new[:, 1], new[:, 0]])).to(gh.idtype)
            data[c] = (u, v)
        else:
            data[c] = gh.edges(etype=c)
    gnew = dgl.heterograph(data, num_nodes_dict={ntype: n}, idtype=gh.idtype)
    for key in gh.ndata.keys():
        gnew.ndata[key] = gh.ndata[key]

    # ---- 4. Tulis folder keluaran ----
    (out / "hetero").mkdir(parents=True)
    save_graphs(str(out / "hetero" / args.dataset), [gnew])
    os.symlink(src_root / args.dataset, out / args.dataset)
    for name in CACHE_DIRS:
        os.symlink(src_root / name, out / name)

    # ---- 5. Baca ulang dan periksa ----
    g2 = load_graphs(str(out / "hetero" / args.dataset))[0][0]
    for c in gh.canonical_etypes:
        a_s, a_d = g2.edges(etype=c)
        if c == target:
            if not np.array_equal(undirected_simple_edges(a_s.numpy(), a_d.numpy()), new):
                raise RuntimeError("Relasi target di file tersimpan tidak sama dengan hasil pengacakan")
        else:
            b_s, b_d = gh.edges(etype=c)
            if not (torch.equal(a_s, b_s) and torch.equal(a_d, b_d)):
                raise RuntimeError(f"Relasi {c} ikut berubah. Hentikan.")
    for key in gh.ndata.keys():
        if not torch.equal(g2.ndata[key], gh.ndata[key]):
            raise RuntimeError(f"ndata['{key}'] berubah. Hentikan.")

    meta = {
        "relation": args.relation,
        "etype": ETYPE[args.relation],
        "k": args.k,
        "seed": args.seed,
        "n_nodes": int(n),
        "n_edges_undirected": int(len(orig)),
        "n_swaps": int(nswap),
        "pct_original_edges_kept": round(kept, 2),
        "H_before": round(heterophily(orig, y), 4),
        "H_after": round(heterophily(new, y), 4),
        "q_anomaly_edge_endpoints": round(q, 4),
        "H_expected_full_shuffle_2q(1-q)": round(2 * q * (1 - q), 4),
        "H_random_label_2p(1-p)": round(2 * p * (1 - p), 4),
        "shuffle_seconds": round(elapsed, 1),
        "networkx": nx.__version__,
        "dgl": dgl.__version__,
        "source": str(src_root),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"OK: {out}")


if __name__ == "__main__":
    main()



