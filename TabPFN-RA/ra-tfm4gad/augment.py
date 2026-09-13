import os
import time

import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import scipy.special
import sympy
import torch
from scipy.sparse.linalg import lobpcg, LinearOperator
from sklearn.utils.extmath import randomized_svd


def standard_scale(x, dim=0, eps=1e-6):
    m = x.mean(dim, keepdim=True)
    s = x.std(dim, unbiased=False, keepdim=True) + eps
    x -= m
    x /= s
    return x


def sage_feature_augment(g: dgl.DGLGraph, feat: torch.Tensor,
                             num_layers: int=1, agg: str='mean', sc: bool=False) -> torch.Tensor:
    """
    Feature augmentation using non-parametric SAGE-style polynomial convolution.

    Parameters:
        g: DGLGraph
        feat: Tensor [num_nodes, feat_dim]
        agg: 'sum' / 'mean' / 'max'
        num_layers: number of aggregation hops
        sc: bool, apply standard scaler to aggregated features at each hop
    Returns:
        Tensor [num_nodes, feat_dim * (num_layers + 1)]
    """
    assert agg in ['sum', 'mean', 'max'], "agg must be one of 'sum', 'mean', 'max'"

    results = [feat if not sc else standard_scale(feat)]
    h = feat

    with g.local_scope():  # temporary node data
        for _ in range(num_layers):
            g.ndata['h'] = h
            if agg == 'sum':
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            elif agg == 'mean':
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            elif agg == 'max':
                g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))

            h = g.ndata['neigh']
            if sc:
                h = standard_scale(h)
            results.append(h)

    return torch.cat(results, dim=-1)


def calculate_theta(d):
    """
    Calculate fixed polynomial filter coefficients based on beta distribution.

    Args:
        d (int): maximum polynomial degree.

    Returns:
        list[list[float]]: shape (d+1, d+1), each sublist is a polynomial coefficient set.
    """
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) /
                       (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()  # length = d+1
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))  # reverse order
        thetas.append(inv_coeff)
    return thetas


def poly_conv(graph, feat, theta):
    """
    Perform graph polynomial convolution with fixed coefficients (no learnable parameters).

    Args:
        graph (DGLGraph): input graph.
        feat (Tensor): node feature matrix [num_nodes, feat_dim].
        theta (list[float]): polynomial filter coefficients.

    Returns:
        Tensor: updated node features after polynomial convolution.
    """

    def unn_laplacian(feat, D_invsqrt, graph):
        """
        Apply unnormalized Laplacian: feat - D^-1/2 * A * D^-1/2 * feat
        """
        graph.ndata['h'] = feat * D_invsqrt
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return feat - graph.ndata.pop('h') * D_invsqrt

    with graph.local_scope():
        # Calculate D^-1/2 for symmetric normalization
        D_invsqrt = torch.pow(
            graph.in_degrees().float().clamp(min=1), -0.5
        ).unsqueeze(-1).to(feat.device)

        h = theta[0] * feat
        for k in range(1, len(theta)):
            feat = unn_laplacian(feat, D_invsqrt, graph)
            h += theta[k] * feat
    return h


def bwgnn_feature_augment(graph: dgl.DGLGraph, feat: torch.Tensor, num_layers: int=2) -> torch.Tensor:
    """
    Feature augmentation using non-parametric BWGNN-style polynomial convolution.

    Args:
        graph (DGLGraph): input graph.
        feat (Tensor): node features [num_nodes, feat_dim].
        num_layers (int): maximum polynomial degree (d).

    Returns:
        Tensor [num_nodes, feat_dim * (d+1)]: concatenated convolution results.
    """
    # Generate fixed polynomial coefficients
    thetas = calculate_theta(d=num_layers)

    results = []
    # Apply convolution for each polynomial filter
    for theta in thetas:
        h_conv = poly_conv(graph, feat, theta)
        results.append(h_conv)

    # Concatenate all convolution results along feature dimension
    return torch.cat(results, dim=-1)


def compute_degree_encoding(
        graph: dgl.DGLGraph,
        log: bool = True,
) -> torch.Tensor:
    degrees = 0.5 * (graph.in_degrees() + graph.out_degrees())
    if log:
        degrees = torch.log(1 + degrees)
    return degrees.unsqueeze(-1)


def compute_pagerank(
        graph: dgl.DGLGraph,
        alpha: float = 0.85,
        max_iterations: int = 100,
        tol=1e-6,
        log: bool = True,
        cache_dir: str = "./pagerank_cache"
) -> torch.Tensor:
    """
    Compute PageRank scores with optional caching.
    Uses dgl.function for message passing.
    Caches only the raw (non-log) PageRank scores.

    Parameters:
        graph (dgl.DGLGraph): Input graph.
        alpha (float): Damping factor (> 0.5).
        max_iterations (int): Maximum iterations.
        tol (float): Convergence tolerance.
        log (bool): Apply log-transform to final scores before returning.
        cache_dir (str or None): Directory to save/load cached raw results.
    """
    assert alpha > 0.5, "Please make sure that you provide alpha, not 1-alpha"

    # Prepare cache path if cache_dir is given
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        g_name = getattr(graph, "name", "graph")
        cache_path = os.path.join(cache_dir, f"{g_name}_alpha{alpha}_pagerank.pt")
        if os.path.exists(cache_path):
            # Load raw PV from cache
            pv = torch.load(cache_path, weights_only=False)
            # Apply log if requested
            if log:
                pv = torch.log(tol + pv)
            return pv.unsqueeze(-1)

    n_nodes = graph.num_nodes()
    pv = torch.ones(n_nodes) / n_nodes
    degrees = graph.out_degrees().float().clamp(min=1)  # Prevent division by zero

    # Reset probability vector
    reset_prob = (1 - alpha) / n_nodes

    # -------- PageRank Iteration --------
    with graph.local_scope():
        for _ in range(max_iterations):
            prev_pv = pv.clone()

            graph.ndata['pv'] = pv / degrees
            graph.update_all(fn.copy_u('pv', 'm'), fn.sum('m', 'pv_new'))
            new_pv = graph.ndata.pop('pv_new')

            pv = alpha * new_pv + reset_prob

            # Convergence check
            if torch.abs(pv - prev_pv).sum() < tol:
                break

    # Save raw pv (non-log) to cache
    if cache_path is not None:
        torch.save(pv, cache_path)

    # Apply log transform if requested (for returning)
    if log:
        pv = torch.log(tol + pv)

    # Return column vector
    return pv.unsqueeze(-1)


def get_laplacian_pe(g, k, cache_dir="./lap_pe_cache",
                     approx_threshold=100_000, approx_method="randomized_svd") -> torch.Tensor:
    """
    Compute or load Laplacian Position Embedding (LapPE) for graph `g`.

    If a cache file exists and its dimensionality >= k:
        - Load from cache and return the first k dimensions.
    Else:
        - If num_nodes exceeds `approx_threshold`, use approximate spectral method.
        - Otherwise, compute LapPE using exact sparse eigendecomposition.

    Parameters
    ----------
    g : DGLGraph
        Input graph.
    k : int
        Number of Laplacian PE dimensions required.
    cache_dir : str
        Directory to store cache files.
    approx_threshold : int
        Number of nodes above which approximate method will be used.
    approx_method : str
        Approximation method to use: "randomized_svd" or "lobpcg".

    Returns
    -------
    pe : torch.Tensor
        Tensor of shape [num_nodes, k], Laplacian PE.
    """

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{g.name}_lap_pe.pt")

    # Try to load from cache
    cached_k = -1
    if os.path.exists(cache_path):
        start_time = time.time()
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        cached_k = cached.shape[1]
        if cached_k >= k:
            elapsed = time.time() - start_time
            print(f"[Cache] Loaded LapPE from '{cache_path}' (cached_k={cached_k}) in {elapsed:.2f} seconds")
            return cached[:, :k]
        else:
            print(f"[Cache] Found LapPE cache k={cached_k} but need k={k}, recomputing...")

    num_nodes = g.num_nodes()

    # Get adjacency matrix and normalized Laplacian in sparse format
    A = g.adj_external(scipy_fmt="csr")  # adjacency in CSR
    deg_inv_sqrt = 1.0 / np.sqrt(np.clip(g.in_degrees().numpy(), 1, None))
    D_half = sp.diags(deg_inv_sqrt, dtype=float)
    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = sp.eye(num_nodes) - D_half @ A @ D_half

    # Choose computation method
    if num_nodes > approx_threshold:
        start_time = time.time()
        print(f"[Approx] Using {approx_method} for LapPE (n={num_nodes} > threshold={approx_threshold}, k={k})")

        if approx_method == "randomized_svd":
            # Randomized SVD for smallest eigenvectors of Laplacian
            U, Sigma, VT = randomized_svd(L, n_components=k+1, random_state=0)
            PE_vecs = U[:, 1:k+1]  # skip the trivial eigenvector
        elif approx_method == "lobpcg":
            # Define LinearOperator for Laplacian to avoid dense matrix
            def matvec(x):
                x = x.reshape(-1, 1)
                Ax = A @ (deg_inv_sqrt[:, None] * x)
                return (x - deg_inv_sqrt[:, None] * Ax).ravel()

            Lop = LinearOperator((num_nodes, num_nodes), matvec=matvec)
            X_init = np.random.rand(num_nodes, k+1)
            eigvals, eigvecs = lobpcg(Lop, X_init, largest=False, tol=1e-4, maxiter=200)
            PE_vecs = eigvecs[:, 1:k+1]
        else:
            raise ValueError(f"Unknown approx_method: {approx_method}")

        # Randomly flip signs for stability
        rand_sign = 2 * (np.random.rand(k) > 0.5) - 1.0
        PE = torch.tensor(PE_vecs * rand_sign, dtype=torch.float32)

        elapsed = time.time() - start_time
        print(f"[Approx] LapPE computed in {elapsed:.2f} seconds (method={approx_method})")

    else:
        start_time = time.time()
        print(f"[Exact] Calculating Laplacian PE (n={num_nodes}, k={k}) ...")

        if k + 1 < num_nodes - 1:
            # Use ARPACK for exact eigen-decomposition of sparse matrix
            EigVal, EigVec = sp.linalg.eigs(L, k=k+1, which="SR", ncv=4*k, tol=1e-2)
            topk_indices = EigVal.argsort()[1:]  # skip trivial eigenvalue
        else:
            # Fallback: dense eigendecomposition (only feasible for small graphs!)
            EigVal, EigVec = np.linalg.eig(L.toarray())
            max_freqs = min(num_nodes - 1, k)
            kpartition_indices = np.argpartition(EigVal, max_freqs)[:max_freqs+1]
            topk_eigvals = EigVal[kpartition_indices]
            topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]

        topk_EigVec = EigVec[:, topk_indices].real
        rand_sign = 2 * (np.random.rand(len(topk_indices)) > 0.5) - 1.0
        PE = torch.tensor(topk_EigVec * rand_sign, dtype=torch.float32)

        elapsed = time.time() - start_time
        print(f"[Exact] LapPE computed in {elapsed:.2f} seconds (exact method)")

    # Save to cache
    if k > cached_k:
        torch.save(PE, cache_path)
        print(f"[Cache] Saved LapPE to '{cache_path}' (k={k})")
    return PE


def augment_node_features(
    graph: dgl.DGLGraph,
    base_feat: torch.Tensor,
    conf: dict
) -> torch.Tensor:
    """
    Apply composable node feature augmentation on (graph, base_feat) according to config.

    Parameters
    ----------
    graph : dgl.DGLGraph
        Input graph (should be preprocessed: bidirectional, self-loops if needed).
    base_feat : torch.Tensor
        Initial node feature matrix, shape [num_nodes, feat_dim].
    conf : dict
        Augmentation configuration, for example:
        {
            "neighbor": {
                "enable": True,
                "type": "sage",          # 'sage' or 'bwgnn'
                "num_layers": 2,
                "agg": "mean",           # 'sum', 'mean', 'max' (only for type='sage')
                "sc": False
            },
            "degree": {
                "enable": True,
                "log": True
            },
            "pagerank": {
                "enable": True,
                "alpha": 0.85,
                "max_iterations": 100,
                "tol": 1e-6,
                "log": True,
                "cache_dir": "./pagerank_cache"
            },
            "laplacian_pe": {
                "enable": True,
                "k": 16,
                "cache_dir": "./lap_pe_cache",
                "approx_threshold": 100_000,
                "approx_method": "randomized_svd"
            }
        }

    Returns
    -------
    torch.Tensor
        Augmented feature tensor, shape [num_nodes, augmented_dim]
    """
    # --- store all feature blocks to concatenate later ---
    features = []

    # ---- 1. Neighbor-based augmentation ----
    if conf.get("neighbor", {}).get("enable", False):
        neigh_conf = conf["neighbor"]
        if neigh_conf["type"] == "sage":
            h_aug = sage_feature_augment(
                g=graph,
                feat=base_feat,
                num_layers=neigh_conf.get("num_layers", 1),
                agg=neigh_conf.get("agg", "mean"),
                sc=neigh_conf.get("sc", False)
            )
        elif neigh_conf["type"] == "bwgnn":
            h_aug = bwgnn_feature_augment(
                graph,
                feat=base_feat,
                num_layers=neigh_conf.get("num_layers", 2)
            )
        else:
            h_aug = base_feat
        features.append(h_aug)

    # ---- 2. Degree encoding ----
    if conf.get("degree", {}).get("enable", False):
        deg_conf = conf["degree"]
        deg_feat = compute_degree_encoding(graph, log=deg_conf.get("log", True))
        features.append(deg_feat)

    # ---- 3. PageRank ----
    if conf.get("pagerank", {}).get("enable", False):
        pr_conf = conf["pagerank"]
        pr_feat = compute_pagerank(
            graph,
            alpha=pr_conf.get("alpha", 0.85),
            max_iterations=pr_conf.get("max_iterations", 100),
            tol=pr_conf.get("tol", 1e-6),
            log=pr_conf.get("log", True),
            cache_dir=pr_conf.get("cache_dir", "./pagerank_cache")
        )
        features.append(pr_feat)

    # ---- 4. Laplacian PE ----
    if conf.get("laplacian_pe", {}).get("enable", False):
        lap_conf = conf["laplacian_pe"]
        if not hasattr(graph, "name"):
            graph.name = lap_conf.get("graph_name", "unnamed_graph")
        lap_feat = get_laplacian_pe(
            g=graph,
            k=lap_conf.get("k", 16),
            cache_dir=lap_conf.get("cache_dir", "./lap_pe_cache"),
            approx_threshold=lap_conf.get("approx_threshold", 100_000),
            approx_method=lap_conf.get("approx_method", "randomized_svd")
        )
        features.append(lap_feat)

    # ---- Concatenate all features along last dimension ----
    augmented_feat = torch.cat(
        [f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)
         for f in features],
        dim=-1
    )

    return augmented_feat


"""
Tambahan untuk augment.py — versi RA-TFM4GAD.

TEMPELKAN isi berkas ini di BAGIAN BAWAH augment.py pada fork RA.
Jangan menghapus fungsi helper yang sudah ada: bwgnn_feature_augment,
calculate_theta, poly_conv, compute_degree_encoding, compute_pagerank,
get_laplacian_pe, dan standard_scale semuanya dipakai ulang di sini.

Yang boleh dihapus nanti — setelah jalur RA menghasilkan angka terverifikasi —
hanya entry point lama augment_node_features().
"""

from typing import List, Tuple

import dgl
import torch

MERGE_OPS = ("concat", "mean", "max", "hybrid")


def merge_relation_blocks(blocks: List[torch.Tensor], op: str = "concat") -> torch.Tensor:
    """
    Gabungkan blok agregasi antar-relasi (tahap kedua agregasi).

    Tahap pertama — agregasi tetangga di dalam satu relasi memakai Beta Wavelet —
    dipertahankan identik dengan TFM4GAD. Fungsi ini hanya menangani tahap kedua,
    yang menjadi ruang perancangan Eksperimen 6.

    Seluruh operator non-parametrik: tidak ada bobot yang dilatih, sehingga
    paradigma training-free TFM4GAD tetap terjaga.

    Parameters
    ----------
    blocks : list of [num_nodes, d*(C+1)] — satu per relasi, bentuknya identik
    op     : 'concat' | 'mean' | 'max' | 'hybrid'

    Returns
    -------
    Tensor [num_nodes, D] dengan D bergantung operator:
        concat -> k * d*(C+1)
        mean   -> d*(C+1)
        max    -> d*(C+1)
        hybrid -> (k+1) * d*(C+1)
    """
    if op not in MERGE_OPS:
        raise ValueError(f"merge_op tidak dikenal: {op}. Pilihan: {MERGE_OPS}")
    if len(blocks) == 0:
        raise ValueError("Tidak ada blok relasi untuk digabungkan")

    if len(blocks) == 1:
        # Dengan satu relasi seluruh operator setara; kembalikan apa adanya
        # agar dimensi tidak membengkak tanpa alasan.
        return blocks[0]

    shapes = {tuple(b.shape) for b in blocks}
    if len(shapes) != 1:
        raise RuntimeError(f"Blok relasi berbeda bentuk: {shapes}. "
                           "mean/max/hybrid mensyaratkan bentuk identik.")

    if op == "concat":
        return torch.cat(blocks, dim=-1)

    stacked = torch.stack(blocks, dim=0)  # [k, n, D]
    if op == "mean":
        return stacked.mean(dim=0)
    if op == "max":
        return stacked.max(dim=0).values
    # hybrid: blok terpisah DITAMBAH ringkasan rata-rata secara eksplisit.
    # Kalau hasilnya setara concat, itu bukti attention TabPFNv2.5 sudah
    # menyerap perata-rataan tanpa perlu disediakan.
    return torch.cat(blocks + [stacked.mean(dim=0)], dim=-1)


def bwgnn_relation_augment(
    rel_graphs: List[Tuple[str, dgl.DGLGraph]],
    feat: torch.Tensor,
    num_layers: int = 2,
) -> List[Tuple[str, torch.Tensor]]:
    """
    Jalankan bank filter Beta Wavelet SECARA TERPISAH pada tiap subgraf relasi.

    Memakai bwgnn_feature_augment() yang sama dengan baseline, hanya grafnya
    yang berbeda. Dengan begitu satu-satunya perbedaan terhadap TFM4GAD adalah
    himpunan edge yang diagregasi, bukan mekanisme filternya.
    """
    out = []
    for name, g_r in rel_graphs:
        h = bwgnn_feature_augment(g_r, feat=feat, num_layers=num_layers)  # noqa: F821
        out.append((name, h))
    return out


def augment_node_features_ra(
    homo_graph: dgl.DGLGraph,
    rel_graphs: List[Tuple[str, dgl.DGLGraph]],
    base_feat: torch.Tensor,
    conf: dict,
    merge_op: str = "concat",
    verbose_fn=print,
) -> torch.Tensor:
    """
    Relation-aware graph-to-table flattening.

    Keluaran: [xnbr_gabungan || xchar_degree || xchar_pagerank || xlap]

    Urutan blok SENGAJA identik dengan augment_node_features() milik baseline
    (neighbor -> degree -> pagerank -> laplacian_pe), sehingga satu-satunya
    perbedaan terhadap baseline adalah isi blok xnbr.

    PENTING: xchar dan xlap dihitung dari `homo_graph`, bukan dari subgraf
    relasi. Ini yang menjaga cache lap_pe_cache/ dan pagerank_cache/ terpakai
    dan membuat xlap identik antara baseline dan RA-TFM4GAD (REVISI item E4).
    """
    features = []

    # ---- 1. Agregasi tetangga per relasi ----
    neigh_conf = conf.get("neighbor", {})
    if not neigh_conf.get("enable", False):
        raise RuntimeError("RA-TFM4GAD mensyaratkan neighbor.enable = true")
    if neigh_conf.get("type") != "bwgnn":
        raise RuntimeError(
            f"Jalur RA hanya mendukung type='bwgnn' (YelpChi), "
            f"bukan '{neigh_conf.get('type')}'. Jalur sage menyertakan fitur "
            f"mentah sebagai blok pertama, sehingga strukturnya berbeda.")

    num_layers = neigh_conf.get("num_layers", 2)
    blocks = bwgnn_relation_augment(rel_graphs, base_feat, num_layers=num_layers)
    for name, h in blocks:
        verbose_fn(f"[xnbr] {name:<4}: {tuple(h.shape)}")

    xnbr = merge_relation_blocks([h for _, h in blocks], op=merge_op)
    verbose_fn(f"[xnbr] setelah merge_op='{merge_op}': {tuple(xnbr.shape)}")
    features.append(xnbr)

    # ---- 2. Degree (dari graf homogen) ----
    if conf.get("degree", {}).get("enable", False):
        deg_conf = conf["degree"]
        features.append(compute_degree_encoding(  # noqa: F821
            homo_graph, log=deg_conf.get("log", True)))

    # ---- 3. PageRank (dari graf homogen, cache terpakai) ----
    if conf.get("pagerank", {}).get("enable", False):
        pr_conf = conf["pagerank"]
        features.append(compute_pagerank(  # noqa: F821
            homo_graph,
            alpha=pr_conf.get("alpha", 0.85),
            max_iterations=pr_conf.get("max_iterations", 100),
            tol=pr_conf.get("tol", 1e-6),
            log=pr_conf.get("log", True),
            cache_dir=pr_conf.get("cache_dir", "./pagerank_cache"),
        ))

    # ---- 4. Laplacian PE (dari graf homogen, cache terpakai) ----
    if conf.get("laplacian_pe", {}).get("enable", False):
        lap_conf = conf["laplacian_pe"]
        if not hasattr(homo_graph, "name"):
            raise RuntimeError(
                "homo_graph tidak punya atribut .name — nama cache xlap akan "
                "berubah dan xlap tidak lagi identik dengan baseline.")
        features.append(get_laplacian_pe(  # noqa: F821
            g=homo_graph,
            k=lap_conf.get("k", 16),
            cache_dir=lap_conf.get("cache_dir", "./lap_pe_cache"),
            approx_threshold=lap_conf.get("approx_threshold", 100_000),
            approx_method=lap_conf.get("approx_method", "randomized_svd"),
        ))

    return torch.cat(features, dim=-1)


def expected_final_dim(n_relations: int, feat_dim: int, num_layers: int,
                       merge_op: str, k_lap: int = 16, n_char: int = 2) -> int:
    """Dimensi yang SEHARUSNYA dihasilkan — dipakai sebagai sanity check di main."""
    block = feat_dim * (num_layers + 1)
    if n_relations == 1:
        nbr = block
    elif merge_op == "concat":
        nbr = n_relations * block
    elif merge_op == "hybrid":
        nbr = (n_relations + 1) * block
    else:  # mean / max
        nbr = block
    return nbr + n_char + k_lap
