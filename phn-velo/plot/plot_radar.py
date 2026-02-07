import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ---- USER INPUT -------------------------------------------
# ============================================================

HPN_JSON = "metrics/mnist_hpn.json"
MML_JSON = "metrics/mnist_mml.json"

# ---- paste your LibMOON table here ----
data_mnist = {
    'Metric': ['HV', 'Spacing', 'Span', 'l_min', 's_lmin', 'PBI', 'IP', 'CA'],
    'EPO':       [0.6458, 0.8357, 0.1337, 0.0209, -0.1581, 0.4041, 0.2649, 1.0888],
    'PMGDA':     [0.6010, 1.3533, 0.1314, 0.0209, -0.1435, 0.4555, 0.2994, 1.0378],
    'Agg-PBI':   [0.6599, 0.9151, 0.1170, 0.0173, -0.1619, 0.4162, 0.2580, 1.8488],
}

LIBMOON_METHODS_TO_SHOW = ["EPO", "PMGDA"]  # choose what to compare

TITLE = "MultiMNIST — Metric Radar"
OUT = "radar_multimnist.png"


# ============================================================
# ---- Load PHN / MML JSON ----------------------------------
# ============================================================

def load_json_metrics(path):
    with open(path) as f:
        raw = json.load(f)
    return {k: v["mean"] for k, v in raw.items()}


hpn = load_json_metrics(HPN_JSON)
mml = load_json_metrics(MML_JSON)


# unify naming
rename_json = {
    "hv": "HV",
    "spacing": "Spacing",
    "span": "Span",
    "lmin": "l_min",
    "soft_lmin": "s_lmin",
    "inner_product": "IP",
    "cross_angle": "CA",
    "pbi": "PBI",
}

hpn = {rename_json.get(k, k): v for k, v in hpn.items()}
mml = {rename_json.get(k, k): v for k, v in mml.items()}


# ============================================================
# ---- Build LibMOON dataframe ------------------------------
# ============================================================

df_lib = pd.DataFrame(data_mnist).set_index("Metric")

df_lib = df_lib[LIBMOON_METHODS_TO_SHOW]


# ============================================================
# ---- Merge all methods ------------------------------------
# ============================================================

df_all = df_lib.copy()

df_all["HPN"] = pd.Series(hpn)
df_all["MML"] = pd.Series(mml)

df_all = df_all.dropna()  # keep common metrics only


# ============================================================
# ---- Metric direction handling ----------------------------
# ============================================================

# smaller is better → invert
inverse_metrics = ["Spacing", "PBI", "CA"]

for m in inverse_metrics:
    if m in df_all.index:
        df_all.loc[m] = df_all.loc[m].max() - df_all.loc[m]


# ============================================================
# ---- Robust normalization (rank-based) --------------------
# ============================================================

df_norm = df_all.rank(pct=True)


# ============================================================
# ---- Radar plot -------------------------------------------
# ============================================================

labels = df_norm.index.tolist()
N = len(labels)

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

colors = {
    "HPN": "#d62728",
    "MML": "#1f77b4",
    "EPO": "#2ca02c",
    "PMGDA": "#9467bd",
}

for method in df_norm.columns:

    vals = df_norm[method].tolist()
    vals += vals[:1]

    ax.plot(
        angles,
        vals,
        linewidth=2.2,
        label=method,
        color=colors.get(method, None),
    )

    ax.fill(
        angles,
        vals,
        alpha=0.12,
        color=colors.get(method, None),
    )


# ---- style ----
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)

ax.set_ylim(0, 1)
ax.set_yticklabels([])
ax.grid(alpha=0.25)

plt.title(TITLE, fontsize=14, pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.20, 1.12), frameon=False)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print("Saved:", OUT)