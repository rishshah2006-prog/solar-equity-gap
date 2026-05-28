import pandas as pd
import geopandas as gpd
import os

# Paths
RAW  = "data/raw"
PROC = "data/processed"
os.makedirs(PROC, exist_ok=True)

income = pd.read_csv(f"{RAW}/ACSDT5Y2024.B19013-Data.csv", skiprows=1)
race   = pd.read_csv(f"{RAW}/ACSDT5Y2024.B03002-Data.csv", skiprows=1)
tenure = pd.read_csv(f"{RAW}/ACSDT5Y2024.B25003-Data.csv", skiprows=1)

for df in [income, race, tenure]:
    df["GEOID"] = df["Geography"].str[-11:]

income_col   = income.columns[2]
race_total   = race.columns[2]
race_white   = race.columns[4]
tenure_total = tenure.columns[2]
tenure_owner = tenure.columns[4]

income_clean = income[["GEOID", income_col]].rename(columns={income_col: "median_income"})
race_clean   = race[["GEOID", race_total, race_white]].rename(
    columns={race_total: "total_pop", race_white: "white_alone_nonhisp"})
tenure_clean = tenure[["GEOID", tenure_total, tenure_owner]].rename(
    columns={tenure_total: "total_housing_units", tenure_owner: "owner_occupied"})

census = income_clean.merge(race_clean, on="GEOID").merge(tenure_clean, on="GEOID")

for col in ["median_income", "total_pop", "white_alone_nonhisp",
            "total_housing_units", "owner_occupied"]:
    census[col] = pd.to_numeric(census[col], errors="coerce")

census["pct_nonwhite"]      = 1 - (census["white_alone_nonhisp"] / census["total_pop"])
census["pct_owner_occ"]     = census["owner_occupied"] / census["total_housing_units"]
census["majority_nonwhite"] = census["pct_nonwhite"] > 0.5
census["income_quartile"]   = pd.qcut(
    census["median_income"], q=4,
    labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)
print(f"Census tracts loaded: {len(census)}")


tracts = gpd.read_file(f"{RAW}/tl_2022_17_tract")
tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
tracts = tracts[["GEOID", "COUNTYFP", "geometry"]].to_crs(epsg=4326)
print(f"Shapefile tracts loaded: {len(tracts)}")


solar    = pd.read_csv(f"{RAW}/TTS_LBNL_public_file_29-Sep-2025_all.csv", low_memory=False)
solar_il = solar[solar["state"] == "IL"].copy()
solar_il["zip_code"] = solar_il["zip_code"].astype(str).str.zfill(5)
print(f"Illinois solar installations: {len(solar_il)}")

solar_by_zip = (
    solar_il.groupby("zip_code")["PV_system_size_DC"]
    .agg(total_installed_kw="sum", num_installations="count")
    .reset_index()
)


import glob
zcta_path = glob.glob(f"{RAW}/tl_2022_us_zcta*")[0]
zcta = gpd.read_file(zcta_path).to_crs(epsg=4326)
zcta = zcta.rename(columns={"ZCTA5CE20": "zip_code"})

# Spatial join
il_boundary = tracts.dissolve().geometry
zcta_il = zcta[zcta.intersects(il_boundary.iloc[0])].copy()
print(f"Illinois ZCTAs: {len(zcta_il)}")


zcta_il = zcta_il.merge(solar_by_zip, on="zip_code", how="left")
zcta_il["total_installed_kw"] = zcta_il["total_installed_kw"].fillna(0)
zcta_il["num_installations"]  = zcta_il["num_installations"].fillna(0)

zcta_centroids = zcta_il.copy()
zcta_centroids["geometry"] = zcta_il.centroid

joined = gpd.sjoin(zcta_centroids[["zip_code", "total_installed_kw",
                                    "num_installations", "geometry"]],
                   tracts[["GEOID", "geometry"]],
                   how="left", predicate="within")

solar_by_tract = (
    joined.groupby("GEOID")[["total_installed_kw", "num_installations"]]
    .sum()
    .reset_index()
)

# Merging
merged = tracts.merge(census, on="GEOID", how="left")
merged = merged.merge(solar_by_tract, on="GEOID", how="left")
merged["total_installed_kw"] = merged["total_installed_kw"].fillna(0)
merged["num_installations"]  = merged["num_installations"].fillna(0)
merged["solar_per_1000_units"] = (
    merged["num_installations"] / merged["total_housing_units"] * 1000
)

print(f"\nFinal merged dataset: {len(merged)} tracts")
print(merged[["GEOID", "median_income", "pct_nonwhite",
              "total_installed_kw", "num_installations"]].head(10))

merged.to_file(f"{PROC}/solar_equity_merged.gpkg", driver="GPKG")
print(f"\nSaved to {PROC}/solar_equity_merged.gpkg")
