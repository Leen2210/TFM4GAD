import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# TabPFN imports
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

# Local imports
from dataloader import load_data
from augment import augment_node_features
from misc import get_training_config, get_logger, set_seed, metrics


# ---------------- Helper ---------------- #
def run_single_test(clf, X_train, y_train, X_test, y_test, batch_size, logger):
    """Run batched inference on test set."""
    n_test = X_test.shape[0]
    probs_all = np.zeros((n_test,), dtype=float)
    for i in tqdm(range(0, n_test, batch_size), desc="Batches", disable=True):
        X_batch = X_test[i:i + batch_size]
        pred_proba = clf.predict_proba(X_batch)
        if pred_proba.ndim == 2 and pred_proba.shape[1] >= 2:
            probs = pred_proba[:, 1]
        else:
            probs = pred_proba.ravel()
        probs_all[i:i + batch_size] = probs
    score = metrics(y_test, probs_all)
    logger.info(f"AUROC={score['AUROC']:.2f}%, AUPRC={score['AUPRC']:.2f}%, RecK={score['RecK']:.2f}%")
    return score


# ---------------- Eksperimen 7a ---------------- #
ETYPE_7A = {"rur": "net_rur", "rtr": "net_rtr", "rsr": "net_rsr"}


def relation_degree_columns(data_dir, dataset, relations, x, y, log=print):
    """Kolom log(1 + derajat) per relasi untuk Eksperimen 7a.

    Struktur relasi dibaca dari <data_dir>/hetero/<dataset>. Derajat dihitung setelah
    to_bidirected dan remove_self_loop, sama dengan praproses subgraf relasi RA-TFM4GAD
    tetapi tanpa self-loop. Urutan kolom mengikuti `relations`.
    """
    import dgl
    from dgl.data.utils import load_graphs

    gh = load_graphs(f'{data_dir}/hetero/{dataset}')[0][0]
    n = x.shape[0]
    if gh.num_nodes() != n:
        raise SystemExit(f"[7a] Jumlah node hetero={gh.num_nodes()} vs homo={n}")
    if not torch.equal(gh.ndata['label'].long().view(-1), y.long().view(-1)):
        raise SystemExit("[7a] Label hetero dan homo berbeda: urutan node tidak sejajar")
    if not torch.allclose(gh.ndata['feature'], x):
        raise SystemExit("[7a] Fitur hetero dan homo berbeda: urutan node tidak sejajar")

    cols = []
    for r in relations:
        src, dst = gh.edges(etype=ETYPE_7A[r])
        g_r = dgl.graph((src, dst), num_nodes=n)
        g_r = dgl.remove_self_loop(dgl.to_bidirected(g_r))
        deg = g_r.in_degrees().float()
        log(f"[7a] {r}: {int((deg > 0).sum())} node bertetangga, "
            f"{g_r.num_edges()} edge berarah tanpa self-loop")
        cols.append(torch.log1p(deg).unsqueeze(-1))
    return torch.cat(cols, dim=1)


# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Graph anomaly detection with TabPFN classifier")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--dataset', type=str, default='amazon',
                        choices=['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
                                 'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial'])
    parser.add_argument('--data_dir', type=str, default="../../datasets",
                        help="Path to dataset")
    parser.add_argument('--config', type=str, default='icl.conf.yaml',
                        help="Path to config YAML")
    parser.add_argument('--model_ckpt', type=str, default='../ckpts/tabpfn-v2-classifier.ckpt',
                        choices=['../ckpts/tabpfn-v2.5-classifier-v2.5_default.ckpt', '../ckpts/tabpfn-v2-classifier.ckpt'])
    parser.add_argument('--model_version', type=str, choices=['v2', 'v2.5'], default='v2.5',
                        help="Which TabPFN version to use")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    
    # ---- Eksperimen 4 (default mati: tanpa argumen ini perilaku lama tidak berubah) ----
    parser.add_argument('--split_file', type=str, default=None,
                        help="File JSON dari exp4_splits.py (Eksperimen 4)")
    parser.add_argument('--n_labels', type=int, default=None, choices=[50, 100, 200],
                        help="Ukuran in-context set; wajib bersama --split_file")
    parser.add_argument('--tag', type=str, default='',
                        help="Bila diisi, log ditulis ke log/<dataset>_<tag>.log")

    # ---- Eksperimen 7a (default mati: tanpa argumen ini perilaku lama tidak berubah) ----
    parser.add_argument('--rel_degree', type=str, default=None,
                        help="Eksperimen 7a: tambahkan kolom log(1 + derajat) per relasi, "
                             "misalnya 'rur' atau 'rur,rtr,rsr'")
    args = parser.parse_args()
    rel_degree = None
    if args.rel_degree:
        rel_degree = [r.strip() for r in args.rel_degree.split(',') if r.strip()]
        if args.dataset != 'yelp':
            raise SystemExit("--rel_degree hanya untuk yelp")
        if not rel_degree or any(r not in ETYPE_7A for r in rel_degree) \
                or len(set(rel_degree)) != len(rel_degree):
            raise SystemExit(f"--rel_degree tidak sah: {args.rel_degree}. Pilihan: {list(ETYPE_7A)}")
        if not args.tag:
            raise SystemExit("Beri --tag agar log Eksperimen 7a tidak tercampur ke log lama")
    if (args.split_file is None) != (args.n_labels is None):
        raise SystemExit("--split_file dan --n_labels harus diberikan bersamaan")
    if args.split_file and not args.tag:
        raise SystemExit("Beri --tag agar log Eksperimen 4 tidak tercampur ke log lama")

    # ---------------- Setup logging and seed ---------------- #
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_name = f'{args.dataset}_{args.tag}.log' if args.tag else f'{args.dataset}1.log'
    log_path = os.path.join('log', log_name)
    logger = get_logger(log_path, log_level=1, name="MainLogger", mode='a')
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # ---------------- Read config ---------------- #
    conf = get_training_config(args.dataset, config_path=args.config)
    data_dir = Path(args.data_dir).expanduser().resolve()
    for key in ["pagerank", "laplacian_pe"]:
        if key in conf.get("augmentation", {}) and "cache_dir" in conf["augmentation"][key]:
            conf["augmentation"][key]["cache_dir"] = os.path.join(data_dir, conf["augmentation"][key]["cache_dir"])
    logger.info(f"Loaded config for dataset={args.dataset}: {conf}")

    # ---------------- Load graph ---------------- #
    g, x, y, train_masks, val_masks, test_masks = load_data(
        args.dataset,
        root=args.data_dir,
        n_anomalies=conf.get('n_anomalies', 20),
        feat_trans=conf.get('feat_trans', 'no')
    )
    n_splits = test_masks.shape[1]
    logger.info(f"Loaded graph with {g.num_nodes()} nodes, splits={n_splits}")

    # ---------------- Eksperimen 7a: hitung kolom derajat sebelum augmentasi ---------------- #
    deg_cols = None
    if rel_degree:
        deg_cols = relation_degree_columns(args.data_dir, args.dataset, rel_degree, x, y,
                                           log=logger.info)

    
    # ---------------- Eksperimen 4: ganti train mask ---------------- #
    if args.split_file:
        sys.path.append(str(Path(__file__).resolve().parents[2]))  # root repo
        from exp4_splits import load_train_masks
        train_masks = torch.from_numpy(load_train_masks(
            args.split_file, args.n_labels, y.cpu().numpy(),
            train_masks.cpu().numpy(), test_masks.cpu().numpy(), log=logger.info))

    # ---------------- Feature augmentation ---------------- #
    if conf.get('augmentation', {}).get('enable', False):
        logger.info("Applying configurable feature augmentation...")
        features = augment_node_features(g, x, conf['augmentation'])
        logger.info(f"Feature augmentation done, initial_dim={x.shape[1]}, final_dim={features.shape[1]}")
    else:
        logger.info(f"Using raw node features, final_dim={x.shape[1]}")
        features = x

    # ---------------- Eksperimen 7a: tempel kolom derajat di akhir ---------------- #
    if deg_cols is not None:
        base_dim = features.shape[1]
        features = torch.cat([features, deg_cols.to(features.dtype)], dim=1)
        logger.info(f"[7a] kolom derajat {rel_degree} ditempel: final_dim {base_dim} -> {features.shape[1]}")
        if base_dim != 114 or features.shape[1] != base_dim + len(rel_degree):
            raise SystemExit(f"[7a] Dimensi tidak sesuai: {base_dim} -> {features.shape[1]}")

    # Convert features/labels to numpy
    features_np = features.cpu().numpy()
    labels_np = y.cpu().numpy().ravel()

    # ---------------- Splits evaluation ---------------- #
    split_scores = []
    for split_id in range(n_splits):
        logger.info(f"\n=== Split {split_id+1}/{n_splits} ===")
        train_idx = np.where(train_masks[:, split_id].cpu().numpy())[0]
        test_idx = np.where(test_masks[:, split_id].cpu().numpy())[0]
        logger.info(f"Train size={len(train_idx)}, Test size={len(test_idx)}")

        X_train, y_train = features_np[train_idx], labels_np[train_idx].astype(int)
        X_test, y_test = features_np[test_idx], labels_np[test_idx].astype(int)

        # Initialize TabPFN classifier per split (fit on current split's train set)
        model_version = ModelVersion.V2_5 if 'v2.5' in args.model_ckpt else ModelVersion.V2
        logger.info(f"Initializing TabPFN classifier, version={model_version}, model_ckpt={args.model_ckpt}")
        clf = TabPFNClassifier.create_default_for_version(model_version, model_path=args.model_ckpt, device=args.device)

        # Fit classifier on current training data
        # TabPFN expects numpy arrays; labels should be 0/1
        clf.fit(X_train, y_train)

        score = run_single_test(
            clf, X_train, y_train, X_test, y_test,
            batch_size=conf.get('batch_size', 2048),
            logger=logger,
        )
        split_scores.append(score)

    # ---------------- Aggregate metrics ---------------- #
    def agg(name):
        vals = np.array([s[name] for s in split_scores])
        return vals.mean(), vals.std()

    auroc_m, auroc_s = agg('AUROC')
    auprc_m, auprc_s = agg('AUPRC')
    reck_m, reck_s = agg('RecK')
    logger.info("\n=== Final Summary ===")
    logger.info(f"AUROC={auroc_m:.2f}±{auroc_s:.2f}, AUPRC={auprc_m:.2f}±{auprc_s:.2f}, RecK={reck_m:.2f}±{reck_s:.2f}\n")

if __name__ == "__main__":
    main()