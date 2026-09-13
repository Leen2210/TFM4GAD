import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from dataloader import load_data_hetero, RELATION_ORDER
from augment import augment_node_features_ra, expected_final_dim, MERGE_OPS
from misc import get_training_config, get_logger, set_seed, metrics


def run_single_test(clf, X_train, y_train, X_test, y_test, batch_size, logger):
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
    logger.info(f"AUROC={score['AUROC']:.2f}%, AUPRC={score['AUPRC']:.2f}%, "
                f"RecK={score['RecK']:.2f}%")
    return score


def main():
    parser = argparse.ArgumentParser(description="RA-TFM4GAD: relation-aware GAD dengan TabPFN")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--dataset', type=str, default='yelp', choices=['yelp'],
                        help="Jalur RA saat ini hanya divalidasi untuk YelpChi")
    parser.add_argument('--data_dir', type=str, default="../../datasets")
    parser.add_argument('--config', type=str, default='icl.conf.yaml')
    parser.add_argument('--model_ckpt', type=str,
                        default='../ckpts/tabpfn-v2.5-classifier-v2.5_default.ckpt')
    parser.add_argument('--seed', type=int, default=0)

    # ---- argumen khusus RA ----
    parser.add_argument('--relations', type=str, default='rur,rtr,rsr',
                        help="Relasi yang dipakai, urut. Eksperimen 2: "
                             "A1='rtr,rsr' A2='rur,rsr' A3='rur,rtr' A4='rur,rtr,rsr'")
    parser.add_argument('--merge_op', type=str, default='concat', choices=list(MERGE_OPS),
                        help="Operator penggabungan antar-relasi (Eksperimen 6)")
    parser.add_argument('--tag', type=str, default='',
                        help="Label tambahan untuk nama berkas log")
    args = parser.parse_args()

    relations = [r.strip() for r in args.relations.split(',') if r.strip()]
    unknown = [r for r in relations if r not in RELATION_ORDER]
    if unknown:
        raise SystemExit(f"Relasi tidak dikenal: {unknown}. Pilihan: {list(RELATION_ORDER)}")
    if len(set(relations)) != len(relations):
        raise SystemExit(f"Ada relasi duplikat: {relations}")

    # ---------------- Logging ---------------- #
    os.makedirs('./log', exist_ok=True)
    run_id = f"{args.dataset}_ra_{'-'.join(relations)}_{args.merge_op}"
    if args.tag:
        run_id += f"_{args.tag}"
    logger = get_logger(os.path.join('log', f'{run_id}.log'),
                        log_level=1, name="RALogger", mode='a')
    logger.info(f"\n########## RUN {run_id} ##########")
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # ---------------- Config ---------------- #
    conf = get_training_config(args.dataset, config_path=args.config)
    data_dir = Path(args.data_dir).expanduser().resolve()
    for key in ["pagerank", "laplacian_pe"]:
        if key in conf.get("augmentation", {}) and "cache_dir" in conf["augmentation"][key]:
            conf["augmentation"][key]["cache_dir"] = os.path.join(
                data_dir, conf["augmentation"][key]["cache_dir"])
    logger.info(f"Config dataset={args.dataset}: {conf}")
    logger.info(f"relations={relations} | merge_op={args.merge_op}")

    # ---------------- Load ---------------- #
    g, rel_graphs, x, y, train_masks, val_masks, test_masks = load_data_hetero(
        args.dataset,
        root=args.data_dir,
        n_anomalies=conf.get('n_anomalies', 20),
        feat_trans=conf.get('feat_trans', 'no'),
        relations=relations,
        verbose_fn=logger.info,
    )
    n_splits = test_masks.shape[1]
    logger.info(f"Graf homogen: {g.num_nodes()} node, {g.num_edges()} edge | splits={n_splits}")

    # ---------------- Augmentasi ---------------- #
    aug_conf = conf['augmentation']
    features = augment_node_features_ra(
        homo_graph=g,
        rel_graphs=rel_graphs,
        base_feat=x,
        conf=aug_conf,
        merge_op=args.merge_op,
        verbose_fn=logger.info,
    )

    # ---------------- Sanity check dimensi ---------------- #
    exp_dim = expected_final_dim(
        n_relations=len(relations),
        feat_dim=x.shape[1],
        num_layers=aug_conf['neighbor'].get('num_layers', 2),
        merge_op=args.merge_op,
        k_lap=aug_conf['laplacian_pe'].get('k', 16),
        n_char=int(aug_conf.get('degree', {}).get('enable', False))
               + int(aug_conf.get('pagerank', {}).get('enable', False)),
    )
    logger.info(f"initial_dim={x.shape[1]}, final_dim={features.shape[1]} "
                f"(diharapkan {exp_dim})")
    if features.shape[1] != exp_dim:
        raise SystemExit(
            f"Dimensi tidak sesuai: {features.shape[1]} != {exp_dim}. "
            "Hentikan — ada kesalahan konstruksi fitur, angka apa pun tidak layak dipercaya.")
    if features.shape[1] > 500:
        logger.info(f"PERINGATAN: {features.shape[1]} kolom MELEBIHI ambang "
                    f"max_features_per_estimator=500 — subsampling fitur akan aktif "
                    f"dan perbandingan antar-operator tidak lagi setara.")

    features_np = features.cpu().numpy()
    labels_np = y.cpu().numpy().ravel()

    # ---------------- Evaluasi per split ---------------- #
    split_scores = []
    for split_id in range(n_splits):
        logger.info(f"\n=== Split {split_id+1}/{n_splits} ===")
        train_idx = np.where(train_masks[:, split_id].cpu().numpy())[0]
        test_idx = np.where(test_masks[:, split_id].cpu().numpy())[0]
        logger.info(f"Train size={len(train_idx)}, Test size={len(test_idx)}")

        X_train, y_train = features_np[train_idx], labels_np[train_idx].astype(int)
        X_test, y_test = features_np[test_idx], labels_np[test_idx].astype(int)

        model_version = ModelVersion.V2_5 if 'v2.5' in args.model_ckpt else ModelVersion.V2
        clf = TabPFNClassifier.create_default_for_version(
            model_version, model_path=args.model_ckpt, device=args.device)
        clf.fit(X_train, y_train)

        split_scores.append(run_single_test(
            clf, X_train, y_train, X_test, y_test,
            batch_size=conf.get('batch_size', 2048), logger=logger))

    def agg(name):
        vals = np.array([s[name] for s in split_scores])
        return vals.mean(), vals.std()

    auroc_m, auroc_s = agg('AUROC')
    auprc_m, auprc_s = agg('AUPRC')
    reck_m, reck_s = agg('RecK')
    logger.info("\n=== Final Summary ===")
    logger.info(f"run_id={run_id}")
    logger.info(f"AUROC={auroc_m:.2f}±{auroc_s:.2f}, AUPRC={auprc_m:.2f}±{auprc_s:.2f}, "
                f"RecK={reck_m:.2f}±{reck_s:.2f}")
    logger.info(f"Baseline internal YelpChi: AUROC=72.20±3.37, AUPRC=32.24±3.99\n")


if __name__ == "__main__":
    main()