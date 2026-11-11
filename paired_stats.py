"""
Untuk setiap perbandingan: selisih per split (metode A - metode B, 10 split resmi),
rata-rata selisih, SD sampel, IK95% (distribusi t, df = 9), jumlah menang/kalah,
dan uji Wilcoxon signed-rank dua sisi (scipy.stats.wilcoxon, pengaturan bawaan).
"""
import re
import sys
from decimal import Decimal, ROUND_FLOOR

import numpy as np
from scipy import stats

LOG = {
    "A4_100": "final_log/Eksperimen 1/yelp_ra_rur-rtr-rsr_concat.log",
    "TFM_100": "final_log/Eksperimen 4/yelp_exp4_n100.log",
    "raw_100": "final_log/Eksperimen 5/yelp_raw_exp5.log",
    "A1_100": "final_log/Eksperimen 2/yelp_ra_rtr-rsr_concat.log",
    "A2_100": "final_log/Eksperimen 2/yelp_ra_rur-rsr_concat.log",
    "A3_100": "final_log/Eksperimen 2/yelp_ra_rur-rtr_concat.log",
    "RSR_100": "final_log/Diagnostik/yelp_ra_rsr_concat.log",
    "RUR_100": "final_log/Diagnostik/yelp_ra_rur_concat.log",
    "mean_100": "final_log/Eksperimen 1/yelp_ra_rur-rtr-rsr_mean.log",
    "max_100": "final_log/Eksperimen 6/yelp_ra_rur-rtr-rsr_max.log",
    "hybrid_100": "final_log/Eksperimen 6/yelp_ra_rur-rtr-rsr_hybrid.log",
    "A4_50": "final_log/Eksperimen 4/yelp_ra_rur-rtr-rsr_concat_exp4_n50.log",
    "A4_200": "final_log/Eksperimen 4/yelp_ra_rur-rtr-rsr_concat_exp4_n200.log",
    "TFM_50": "final_log/Eksperimen 4/yelp_exp4_n50.log",
    "TFM_200": "final_log/Eksperimen 4/yelp_exp4_n200.log",
    "raw_50": "final_log/Eksperimen 4/yelp_exp4_raw_n50.log",
    "raw_200": "final_log/Eksperimen 4/yelp_exp4_raw_n200.log",
    "A2mean_100": "final_log/Eksperimen 6/6b/yelp_ra_rur-rsr_mean_mean2rel_A2.log",
    "A3mean_100": "final_log/Eksperimen 6/6b/yelp_ra_rur-rtr_mean_mean2rel_A3.log",
    "7a1_100": "final_log/Eksperimen 7/yelp_exp7a_rur.log",
    "7a3_100": "final_log/Eksperimen 7/yelp_exp7a_all.log",
    "7c_100": "final_log/Eksperimen 7/yelp_ra_rur-rtr-rsr_concat_exp7c_rtr_k3_s0.log",
}

LOG_7B = {k: [f"final_log/Eksperimen 7/yelp_ra_rur-rtr-rsr_concat_exp7b_k{k}_s{s}.log" for s in (0, 1)]
          for k in ("0.25", "0.5", "1", "3")}

COMPARISONS = (
    [("AUROC", "RA - TFM4GAD", f"A4_{n}", f"TFM_{n}") for n in (50, 100, 200)]
    + [("AUROC", "TFM4GAD - tanpa graf", f"TFM_{n}", f"raw_{n}") for n in (50, 100, 200)]
    + [("AUROC", "RA - tanpa graf", f"A4_{n}", f"raw_{n}") for n in (50, 100, 200)]
    + [("AUPRC", "RA - TFM4GAD", f"A4_{n}", f"TFM_{n}") for n in (50, 100, 200)]
    + [("gap", "Perubahan selisih RA - TFM4GAD, 50 vs 100", 50, None),
       ("gap", "Perubahan selisih RA - TFM4GAD, 200 vs 100", 200, None)]
    + [("AUROC", f"A4 - {k}", "A4_100", f"{k}_100") for k in ("A1", "A2", "A3", "RSR", "RUR")]
    + [("AUPRC", "A4 - A1", "A4_100", "A1_100")]
    + [("AUROC", f"concat - {k}", "A4_100", f"{k}_100") for k in ("hybrid", "max", "mean")]
    + [("AUROC", "max - mean", "max_100", "mean_100"),
       ("AUROC", "mean - TFM4GAD", "mean_100", "TFM_100")]

    + [(m, "D_derajat: 7b penuh - A1", "7b3_100", "A1_100") for m in ("AUROC", "AUPRC")]
    + [(m, "D_tetangga: A4 - 7b penuh", "A4_100", "7b3_100") for m in ("AUROC", "AUPRC")]
    + [(m, "7a-1 - TFM4GAD", "7a1_100", "TFM_100") for m in ("AUROC", "AUPRC")]
    + [("AUROC", "7a-3 - 7a-1", "7a3_100", "7a1_100"),
       ("AUROC", "7b penuh - 7a-3", "7b3_100", "7a3_100")]
    + [(m, "A4 - 7a-3", "A4_100", "7a3_100") for m in ("AUROC", "AUPRC")]
    + [(m, "7c - A4", "7c_100", "A4_100") for m in ("AUROC", "AUPRC")]
    + [(m, "7c - 7b penuh (post-hoc)", "7c_100", "7b3_100") for m in ("AUROC", "AUPRC")]

    + [("AUROC", "Penurunan A2: concat - mean", "A2_100", "A2mean_100"),
       ("AUROC", "Penurunan A3: concat - mean", "A3_100", "A3mean_100")]
    + [(m, f"Penurunan A4 - penurunan {k}", ("A4_100", "mean_100"), (f"{k}_100", f"{k}mean_100"))
       for k in ("A2", "A3") for m in ("AUROC", "AUPRC")]

    + [(m, "7b penuh - TFM4GAD", "7b3_100", "TFM_100") for m in ("AUROC", "AUPRC")]
    + [(m, "7a-3 - TFM4GAD", "7a3_100", "TFM_100") for m in ("AUROC", "AUPRC")]
    + [("AUPRC", "7b penuh - 7a-3", "7b3_100", "7a3_100"),
       ("AUPRC", "7a-3 - 7a-1", "7a3_100", "7a1_100"),
       ("AUPRC", "concat - mean", "A4_100", "mean_100"),
       ("AUPRC", "Penurunan A2: concat - mean", "A2_100", "A2mean_100"),
       ("AUPRC", "Penurunan A3: concat - mean", "A3_100", "A3mean_100")]
)

LINE = re.compile(r"AUROC=([\d.]+)%, AUPRC=([\d.]+)%")


def per_split(path):
    with open(path, encoding="utf-8") as f:
        rows = [tuple(map(float, m.groups())) for m in LINE.finditer(f.read())]
    if len(rows) != 10:
        sys.exit(f"{path}: ditemukan {len(rows)} baris per split, seharusnya 10")
    a = np.array(rows)
    return {"AUROC": a[:, 0], "AUPRC": a[:, 1]}

def bulat2(x):
    x = Decimal(str(round(float(x), 4)))
    return float((x * 100 + Decimal("0.5")).to_integral_value(ROUND_FLOOR) / 100)

def summarize(d):
    n = d.size
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = stats.wilcoxon(d).pvalue
    return mean, sd, mean - half, mean + half, int((d > 0).sum()), int((d < 0).sum()), p


def main():
    D = {k: per_split(v) for k, v in LOG.items()}
    for k, (s0, s1) in LOG_7B.items():
        r0, r1 = per_split(s0), per_split(s1)
        D[f"7b{k}_100"] = {m: (r0[m] + r1[m]) / 2 for m in ("AUROC", "AUPRC")}
    print(f"{'Metrik':6s} {'Perbandingan':42s} {'n':>4s} {'rata2':>7s} {'SD':>6s} "
          f"{'IK95% bawah':>11s} {'atas':>7s} {'menang':>6s} {'kalah':>5s} {'Wilcoxon p':>10s}")
    for mk, label, a, b in COMPARISONS:
        if mk == "gap":
            n = a
            d = ((D[f"A4_{n}"]["AUROC"] - D[f"TFM_{n}"]["AUROC"])
                 - (D["A4_100"]["AUROC"] - D["TFM_100"]["AUROC"]))
            mk, nlab = "AUROC", "-"
        elif isinstance(a, tuple):
            d = (D[a[0]][mk] - D[a[1]][mk]) - (D[b[0]][mk] - D[b[1]][mk])
            nlab = "100"
        else:
            d = D[a][mk] - D[b][mk]
            nlab = a.split("_")[1]
        mean, sd, lo, hi, w, l, p = summarize(np.round(d, 3))
        print(f"{mk:6s} {label:42s} {nlab:>4s} {bulat2(mean):+7.2f} {sd:6.2f} {lo:+11.2f} {hi:+7.2f} "
              f"{w:6d} {l:5d} {p:10.3f}")

    print("\nRata-rata 7b per tingkat (dua seed dirata-ratakan per split dulu)")
    for k in LOG_7B:
        r = D[f"7b{k}_100"]
        print(f"  k = {k:4s}  AUROC {bulat2(r['AUROC'].mean()):.2f} (SD {r['AUROC'].std():.2f})  "
              f"AUPRC {bulat2(r['AUPRC'].mean()):.2f} (SD {r['AUPRC'].std():.2f})")
    print("7c - A4 per split (AUROC):", np.round(D["7c_100"]["AUROC"] - D["A4_100"]["AUROC"], 2))


if __name__ == "__main__":
    main()