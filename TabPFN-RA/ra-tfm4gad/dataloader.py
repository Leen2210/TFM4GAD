"""
dataloader.py — versi RA-TFM4GAD.

Menambahkan load_data_hetero() di samping load_data() milik baseline.

Prinsip yang dikunci (lihat REVISI item C3 dan C6):
  - STRUKTUR GRAF per relasi diambil dari datasets/hetero/<dataset>
  - FITUR, LABEL, dan MASK diambil dari datasets/<dataset> (file homogen)
    karena kedua file memuat split yang BERBEDA (irisan nol).
  - Tiap subgraf relasi menjalani rantai praproses yang identik dengan baseline.
  - Graf homogen tetap dikembalikan, karena xchar dan xlap dihitung darinya
    agar cache terpakai dan xlap identik dengan baseline.
"""

import dgl
import torch
from dgl.data.utils import load_graphs

# Urutan relasi DIKUNCI mengikuti proposal: [xnbr_RUR || xnbr_RTR || xnbr_RSR].
# Jangan mengambil urutan dari gh.canonical_etypes — file mengembalikan
# ('net_rsr', 'net_rtr', 'net_rur'), sehingga semantik kolom di laporan
# tidak akan cocok dengan kode.
RELATION_ORDER = ("rur", "rtr", "rsr")
RELATION_ETYPE = {"rur": "net_rur", "rtr": "net_rtr", "rsr": "net_rsr"}


def standard_scale(x, dim=0, eps=1e-6):
    m = x.mean(dim, keepdim=True)
    s = x.std(dim, unbiased=False, keepdim=True) + eps
    x -= m
    x /= s
    return x


def load_data(dataset, root, n_anomalies=20, feat_trans='no'):
    """Jalur baseline TFM4GAD — JANGAN diubah, dipakai sebagai kontrol."""
    g = load_graphs(f'{root}/{dataset}')[0][0]
    x = g.ndata.pop('feature')
    y = g.ndata.pop('label')
    if n_anomalies == 20:
        train_masks = g.ndata.pop('train_masks')[:, 10:].bool()
        val_masks = g.ndata.pop('val_masks')[:, 10:].bool()
        test_masks = g.ndata.pop('test_masks')[:, 10:].bool()
    else:
        train_masks = g.ndata.pop('train_masks')[:, :10].bool()
        val_masks = g.ndata.pop('val_masks')[:, :10].bool()
        test_masks = g.ndata.pop('test_masks')[:, :10].bool()

    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g.name = dataset

    if feat_trans == 'l1':
        x = torch.nn.functional.normalize(x, dim=1, p=1)
    elif feat_trans == 'l2':
        x = torch.nn.functional.normalize(x, dim=1, p=2)
    elif feat_trans == 'sc':
        x = standard_scale(x)
    elif feat_trans == 'row':
        x = x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.0))
    else:
        x = x

    return g, x, y, train_masks, val_masks, test_masks


def _build_relation_subgraph(gh, etype, num_nodes):
    """
    Bangun subgraf satu relasi dengan praproses identik baseline.

    num_nodes WAJIB eksplisit. Tanpa itu, node berindeks tinggi yang tidak
    muncul di daftar edge akan hilang dari graf — pada R-U-R ada 22.123 node
    terisolasi, sehingga penjajaran baris terhadap fitur/label/mask akan
    rusak secara senyap.

    Catatan: sengaja TIDAK menyetel atribut .name. Subgraf relasi hanya
    dipakai untuk xnbr; kalau suatu saat keliru diteruskan ke
    get_laplacian_pe(), ketiadaan .name membuatnya gagal keras alih-alih
    menulis cache xlap baru yang salah.
    """
    src, dst = gh.edges(etype=etype)
    g_r = dgl.graph((src, dst), num_nodes=num_nodes)
    g_r = dgl.to_bidirected(g_r)
    g_r = dgl.remove_self_loop(g_r)
    g_r = dgl.add_self_loop(g_r)
    return g_r


def load_data_hetero(dataset, root, n_anomalies=20, feat_trans='no',
                     relations=RELATION_ORDER, verbose_fn=print):
    """
    Muat data untuk RA-TFM4GAD.

    Returns
    -------
    g_homo      : graf homogen terpraproses (sumber xchar dan xlap), .name = dataset
    rel_graphs  : list[(nama_relasi, DGLGraph)] terurut sesuai `relations`
    x, y        : fitur dan label dari file HOMOGEN
    train/val/test_masks : mask dari file HOMOGEN
    """
    for r in relations:
        if r not in RELATION_ETYPE:
            raise ValueError(f"Relasi tidak dikenal: {r}. Pilihan: {list(RELATION_ETYPE)}")

    # ---------- 1. File homogen: sumber fitur, label, mask, xchar, xlap ----------
    g = load_graphs(f'{root}/{dataset}')[0][0]
    x = g.ndata.pop('feature')
    y = g.ndata.pop('label')
    if n_anomalies == 20:
        train_masks = g.ndata.pop('train_masks')[:, 10:].bool()
        val_masks = g.ndata.pop('val_masks')[:, 10:].bool()
        test_masks = g.ndata.pop('test_masks')[:, 10:].bool()
    else:
        # PERINGATAN: cabang ini adalah protokol FULLY-SUPERVISED (kolom 0-9),
        # bukan setting label scarcity. Lihat REVISI item C2.
        train_masks = g.ndata.pop('train_masks')[:, :10].bool()
        val_masks = g.ndata.pop('val_masks')[:, :10].bool()
        test_masks = g.ndata.pop('test_masks')[:, :10].bool()

    num_nodes = g.num_nodes()

    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g.name = dataset  # menjaga nama cache xlap/PageRank identik dengan baseline

    # ---------- 2. File hetero: sumber struktur per relasi ----------
    gh = load_graphs(f'{root}/hetero/{dataset}')[0][0]

    # ---------- 3. Guard penjajaran node (murah, mencegah kegagalan senyap) ----------
    if gh.num_nodes() != num_nodes:
        raise RuntimeError(
            f"Jumlah node tidak sama: hetero={gh.num_nodes()} vs homo={num_nodes}")
    if not torch.equal(gh.ndata['label'].long().view(-1), y.long().view(-1)):
        raise RuntimeError(
            "Vektor label hetero dan homo BERBEDA — urutan node tidak sejajar. "
            "Hentikan; seluruh perbandingan akan tidak sah.")
    if not torch.allclose(gh.ndata['feature'], x):
        raise RuntimeError(
            "Matriks fitur hetero dan homo BERBEDA — urutan node tidak sejajar.")
    verbose_fn(f"[guard] penjajaran node hetero/homo OK (n={num_nodes})")

    # ---------- 4. Subgraf per relasi ----------
    rel_graphs = []
    for r in relations:
        g_r = _build_relation_subgraph(gh, RELATION_ETYPE[r], num_nodes)
        rel_graphs.append((r, g_r))
        verbose_fn(f"[relasi] {r:<4} ({RELATION_ETYPE[r]}): "
                   f"{g_r.num_edges()} edge setelah praproses")

    # ---------- 5. Transformasi fitur (identik baseline) ----------
    if feat_trans == 'l1':
        x = torch.nn.functional.normalize(x, dim=1, p=1)
    elif feat_trans == 'l2':
        x = torch.nn.functional.normalize(x, dim=1, p=2)
    elif feat_trans == 'sc':
        x = standard_scale(x)
    elif feat_trans == 'row':
        x = x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.0))

    return g, rel_graphs, x, y, train_masks, val_masks, test_masks