"""Guard Eksperimen 7b dan 7c: bandingkan fitur RA-TFM4GAD (A4, concat) dari data asli dan data acak.

Lolos bila:
  - final_dim = 306 pada keduanya,
  - seluruh kolom di luar blok relasi yang diacak identik bit-per-bit,
  - blok relasi yang diacak memang berubah.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RA_DIR = ROOT / "TabPFN-RA" / "ra-tfm4gad"
sys.path.insert(0, str(RA_DIR))

from augment import augment_node_features_ra, expected_final_dim  
from dataloader import RELATION_ORDER, load_data_hetero  
from misc import get_training_config  


def build_features(data_dir, dataset):
    """Bangun fitur A4 concat persis seperti main.py RA, tanpa inferensi."""
    conf = get_training_config(dataset, config_path=str(RA_DIR / "icl.conf.yaml"))
    data_dir = Path(data_dir).expanduser().resolve()
    aug = conf["augmentation"]
    for key in ["pagerank", "laplacian_pe"]:
        if key in aug and "cache_dir" in aug[key]:
            aug[key]["cache_dir"] = os.path.join(data_dir, aug[key]["cache_dir"])
    g, rel_graphs, x, _y, *_ = load_data_hetero(
        dataset, root=str(data_dir),
        n_anomalies=conf.get("n_anomalies", 20),
        feat_trans=conf.get("feat_trans", "no"),
        relations=list(RELATION_ORDER),
    )
    feats = augment_node_features_ra(
        homo_graph=g, rel_graphs=rel_graphs, base_feat=x, conf=aug, merge_op="concat")
    num_layers = aug["neighbor"].get("num_layers", 2)
    exp_dim = expected_final_dim(
        n_relations=len(RELATION_ORDER), feat_dim=x.shape[1], num_layers=num_layers,
        merge_op="concat", k_lap=aug["laplacian_pe"].get("k", 16),
        n_char=int(aug.get("degree", {}).get("enable", False))
        + int(aug.get("pagerank", {}).get("enable", False)))
    return feats, x.shape[1] * (num_layers + 1), exp_dim


def check_one(fa, width, exp_dim, folder, dataset):
    relation = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))["relation"]
    fb, _, _ = build_features(folder, dataset)

    i = RELATION_ORDER.index(relation)
    lo, hi = i * width, (i + 1) * width
    outside = torch.ones(fa.shape[1], dtype=torch.bool)
    outside[lo:hi] = False

    dims_ok = fa.shape == fb.shape and fa.shape[1] == exp_dim
    same_outside = dims_ok and torch.equal(fa[:, outside], fb[:, outside])
    changed_inside = dims_ok and not torch.equal(fa[:, lo:hi], fb[:, lo:hi])

    print(f"\n== {folder} (relasi diacak: {relation}, kolom {lo}..{hi - 1})")
    print(f"final_dim asli={fa.shape[1]}, acak={fb.shape[1]}, diharapkan={exp_dim}")
    if dims_ok:
        diff = (fa[:, outside] - fb[:, outside]).abs().max().item()
        print(f"kolom di luar blok identik bit-per-bit: {same_outside} (selisih maks {diff:.3e})")
        print(f"blok {relation} berubah: {changed_inside}")
    ok = dims_ok and same_outside and changed_inside
    print("LOLOS" if ok else "GAGAL: jangan jalankan TabPFN pada folder ini sebelum penyebabnya ditemukan.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="datasets")
    ap.add_argument("--shuffled", required=True, nargs="+", help="Satu atau lebih folder keluaran exp7_make_shuffled.py")
    ap.add_argument("--dataset", default="yelp")
    args = ap.parse_args()

    fa, width, exp_dim = build_features(args.original, args.dataset)
    results = [check_one(fa, width, exp_dim, f, args.dataset) for f in args.shuffled]
    print(f"\nRingkasan: {sum(results)} dari {len(results)} folder lolos")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()