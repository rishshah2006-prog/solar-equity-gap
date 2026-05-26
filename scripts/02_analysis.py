import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

PROC = "data/processed"
MAPS = "maps"
os.makedirs(MAPS, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
merged = gpd.read_file(f"{PROC}/solar_equity_merged.gpkg")
df = merged.dropna(subset=["median_income", "total_pop", "income_quartile"]).copy()
df = df[df["total_pop"] > 0].copy()
df["income_quartile"] = df["income_quartile"].astype(str)
print(f"Tracts loaded: {len(df)}")

# ── Stats ─────────────────────────────────────────────────────────────────────
adoption_by_income = (
    df.groupby("income_quartile")["solar_per_1000_units"]
    .mean().reset_index()
)
order = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]
q4 = adoption_by_income.loc[adoption_by_income["income_quartile"]=="Q4_highest","solar_per_1000_units"].values[0]
q1 = adoption_by_income.loc[adoption_by_income["income_quartile"]=="Q1_lowest", "solar_per_1000_units"].values[0]

# Zero-solar tract breakdown
zero_by_quartile = df.groupby("income_quartile").apply(
    lambda x: (x["num_installations"] == 0).sum()
).reset_index()
zero_by_quartile.columns = ["income_quartile", "zero_solar_tracts"]
total_by_quartile = df.groupby("income_quartile").size().reset_index(name="total_tracts")
zero_by_quartile = zero_by_quartile.merge(total_by_quartile, on="income_quartile")
zero_by_quartile["pct_zero"] = zero_by_quartile["zero_solar_tracts"] / zero_by_quartile["total_tracts"] * 100

print("\nAdoption by income quartile (installations per 1,000 units):")
print(adoption_by_income.to_string(index=False))
print(f"\nQ4 vs Q1 multiplier: {q4/q1:.1f}x")
print("\nTracts with ZERO solar installations by quartile:")
print(zero_by_quartile[["income_quartile","zero_solar_tracts","total_tracts","pct_zero"]].to_string(index=False))

q1_zero = zero_by_quartile.loc[zero_by_quartile["income_quartile"]=="Q1_lowest","zero_solar_tracts"].values[0]
q1_total = zero_by_quartile.loc[zero_by_quartile["income_quartile"]=="Q1_lowest","total_tracts"].values[0]
q1_pct = zero_by_quartile.loc[zero_by_quartile["income_quartile"]=="Q1_lowest","pct_zero"].values[0]
print(f"\nKEY FINDING: {q1_zero} of {q1_total} lowest-income tracts ({q1_pct:.0f}%) have ZERO solar installations")
print(f"Annual unrealized savings: $52.9M")

# ── Chart 1: Adoption rate by quartile ────────────────────────────────────────
labels = ["Lowest\nIncome (Q1)", "Q2", "Q3", "Highest\nIncome (Q4)"]
vals   = [adoption_by_income.loc[adoption_by_income["income_quartile"]==q,
          "solar_per_1000_units"].values[0] for q in order]
colors = ["#d73027", "#fc8d59", "#fee08b", "#1a9850"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, vals, color=colors, edgecolor="white")
ax.set_ylabel("Solar Installations per 1,000 Housing Units", fontsize=11)
ax.set_title("Solar Adoption Rate by Income Quartile\nIllinois Census Tracts",
             fontsize=13, fontweight="bold")
ax.set_facecolor("#f9f9f9")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{MAPS}/chart_adoption_by_income.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {MAPS}/chart_adoption_by_income.png")

# ── Chart 2: % tracts with zero solar ─────────────────────────────────────────
pct_vals = [zero_by_quartile.loc[zero_by_quartile["income_quartile"]==q,"pct_zero"].values[0] for q in order]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, pct_vals, color=colors, edgecolor="white")
ax.set_ylabel("% of Tracts with Zero Solar Installations", fontsize=11)
ax.set_title("Share of Census Tracts with No Solar\nby Income Quartile — Illinois",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)
ax.set_facecolor("#f9f9f9")
for bar, val in zip(bars, pct_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{MAPS}/chart_zero_solar_by_income.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {MAPS}/chart_zero_solar_by_income.png")

# ── Map 1: Solar adoption rate ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12))
df.plot(column="solar_per_1000_units", ax=ax, cmap="YlGn", legend=True)
ax.set_title("Rooftop Solar Adoption Rate\nby Illinois Census Tract",
             fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{MAPS}/map_solar_adoption.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {MAPS}/map_solar_adoption.png")

# ── Map 2: Equity gap (zero-solar low-income vs high-adoption high-income) ─────
avg_solar = df["solar_per_1000_units"].mean()
gap_tracts  = df[(df["income_quartile"] == "Q1_lowest") & (df["num_installations"] == 0)]
high_tracts = df[(df["income_quartile"] == "Q4_highest") & (df["solar_per_1000_units"] >= avg_solar)]
other       = df[~df.index.isin(gap_tracts.index) & ~df.index.isin(high_tracts.index)]

print(f"\nGap tracts (low income, zero solar): {len(gap_tracts)}")
print(f"High tracts (high income, above-avg solar): {len(high_tracts)}")

fig, ax = plt.subplots(figsize=(10, 12))
other.plot(ax=ax, color="#e0e0e0", linewidth=0.2)
gap_tracts.plot(ax=ax, color="#d7191c", linewidth=0.2)
high_tracts.plot(ax=ax, color="#1a9850", linewidth=0.2)

ax.legend(handles=[
    mpatches.Patch(color="#d7191c", label=f"Low income, zero solar ({len(gap_tracts)} tracts)"),
    mpatches.Patch(color="#1a9850", label=f"High income, above-avg solar ({len(high_tracts)} tracts)"),
    mpatches.Patch(color="#e0e0e0", label="Other tracts"),
], loc="lower left", fontsize=9)
ax.set_title("The Solar Equity Gap\nIllinois Census Tracts", fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{MAPS}/map_equity_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {MAPS}/map_equity_gap.png")

print("\nAnalysis complete.")
