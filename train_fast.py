"""
Streamlined training: skip model comparison, directly train stacking ensemble.
Target: ~5 min instead of ~15 min.
"""
import warnings, numpy as np, pandas as pd, joblib
from copy import deepcopy
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# ============================================================
# Load & Clean
# ============================================================
print("Loading data...")
df = pd.read_csv("train.csv")
print(f"Loaded: {df.shape}")

# 1. Clean district
df["district"] = df["district"].astype(str).str.strip().str.replace("二手房", "", regex=False)

# 2. Remove suspicious
bedrooms_num = pd.to_numeric(df["bedrooms"], errors="coerce")
suspicious = (bedrooms_num == 1) & (df["area_sqm"] > 200)
n_sus = suspicious.sum()
df = df[~suspicious].reset_index(drop=True)
print(f"Removed {n_sus} suspicious rows")

# 3. area_sqm hard limits
area = pd.to_numeric(df["area_sqm"], errors="coerce")
area = np.clip(area, 10, 1000)
df["area_sqm"] = area

# ============================================================
# PPS target with extreme filtering
# ============================================================
print("Computing PPS target...")
y_house_price = df["house_price"].values.astype(float)
area_sqm = df["area_sqm"].values.astype(float)

pps = y_house_price / area_sqm
pps_low_extreme = np.percentile(pps, 0.1)
pps_high_extreme = np.percentile(pps, 99.9)
valid_mask = (pps >= pps_low_extreme) & (pps <= pps_high_extreme)
n_removed = (~valid_mask).sum()
df = df[valid_mask].reset_index(drop=True)
y_house_price = y_house_price[valid_mask]
area_sqm = area_sqm[valid_mask]
pps = pps[valid_mask]
if n_removed > 0:
    print(f"Removed {n_removed} extreme PPS samples")

pps_lower = np.percentile(pps, 0.5)
pps_upper = np.percentile(pps, 99.5)
pps_clipped = np.clip(pps, pps_lower, pps_upper)
y_log_pps = np.log(pps_clipped)
print(f"PPS range: [{pps_lower:.2f}, {pps_upper:.2f}]")
df["log_pps"] = y_log_pps

# ============================================================
# Feature engineering
# ============================================================
print("Feature engineering...")
target_series = pd.Series(y_log_pps, name="log_pps")

def extract_title_features(title_series):
    t = title_series.fillna("").astype(str).str.lower()
    keywords = {
        "has_subway": "地铁", "has_decorated": "精装|豪装", "has_elevator": "电梯",
        "has_school": "学区|学校", "has_movein": "拎包", "has_mature": "满五|满二|满",
        "has_garden": "花园|景观", "has_quiet": "安静|不临街", "has_corner": "边套|全明",
        "has_luxury": "豪华装修", "has_simple": "简单装修", "has_lianjia": "链家好房",
        "has_full_light": "全明", "has_original": "原装", "has_brand": "品牌",
        "has_view": "景观", "has_traffic": "交通", "has_tax_free": "满五|满二",
    }
    r = pd.DataFrame(index=title_series.index)
    for n, p in keywords.items():
        r[n] = t.str.contains(p, regex=True).astype(int)
    r["title_len"] = t.str.len()
    return r

def extract_orientation_features(ori_series):
    o = ori_series.fillna("").astype(str).str.lower()
    r = pd.DataFrame(index=ori_series.index)
    r["ori_south_north"] = (o.str.contains("南") & o.str.contains("北")).astype(int)
    r["ori_south"] = (o.str.contains("南") & ~o.str.contains("北")).astype(int)
    r["ori_east_west"] = (~o.str.contains("南") & ~o.str.contains("北") &
        (o.str.contains("东")|o.str.contains("西")|o.str.contains("东南")|
         o.str.contains("西南")|o.str.contains("东北")|o.str.contains("西北"))).astype(int)
    r["ori_north"] = (o.str.contains("北") & ~o.str.contains("南")).astype(int)
    r["ori_unknown"] = ((o == "") | o.isna() | o.str.contains("进门", na=False) |
        (~r["ori_south_north"].astype(bool) & ~r["ori_south"].astype(bool) &
         ~r["ori_east_west"].astype(bool) & ~r["ori_north"].astype(bool))).astype(int)
    return r

def compute_ratio_features(df_in):
    r = pd.DataFrame(index=df_in.index)
    beds = pd.to_numeric(df_in["bedrooms"], errors="coerce").fillna(2)
    livs = pd.to_numeric(df_in["livingrooms"], errors="coerce").fillna(1)
    a = pd.to_numeric(df_in["area_sqm"], errors="coerce").fillna(df_in["area_sqm"].median())
    tf = pd.to_numeric(df_in["total_floor"], errors="coerce").fillna(df_in["total_floor"].median())
    by = pd.to_numeric(df_in["build_year"], errors="coerce").fillna(2000)
    r["total_rooms"] = beds + livs
    r["rooms_per_sqm"] = (beds + livs) / (a + 1e-6)
    r["area_per_room"] = a / (beds + livs + 1e-6)
    r["bedroom_ratio"] = beds / (beds + livs + 1e-6)
    ha = 2024 - by
    r["house_age"] = ha.clip(0, 100)
    r["is_new_building"] = (ha <= 5).astype(int)
    r["is_old_building"] = (ha > 20).astype(int)
    floor_map = {"低": 0, "中": 1, "高": 2, "低区": 0, "中区": 1, "高区": 2}
    r["floor_region_ord"] = df_in["floor_region"].astype(str).str.strip().map(floor_map).fillna(1)
    r["floor_height_ratio"] = (r["floor_region_ord"] + 1) / (tf + 1)
    r["area_per_bedroom"] = a / (beds + 1e-6)
    fb = pd.cut(tf, bins=[0, 6, 18, 200], labels=["low_rise", "mid_rise", "high_rise"])
    r["floor_bucket_low"] = (fb == "low_rise").astype(int)
    r["floor_bucket_mid"] = (fb == "mid_rise").astype(int)
    r["floor_bucket_high"] = (fb == "high_rise").astype(int)
    r["floor_ratio_x_total"] = r["floor_height_ratio"] * tf
    r["log_area_sqm"] = np.log(a + 1e-6)
    r["log_total_rooms"] = np.log(r["total_rooms"] + 1e-6)
    r["log_area_per_bedroom"] = np.log(r["area_per_bedroom"] + 1e-6)
    r["rooms_x_log_area"] = r["total_rooms"] * r["log_area_sqm"]
    r["bedroom_ratio_x_area"] = r["bedroom_ratio"] * a
    return r

def detect_villa(df_in):
    fm = df_in["floor_region"].isna()
    tl = pd.to_numeric(df_in["total_floor"], errors="coerce").fillna(99) <= 3
    return pd.DataFrame({"is_villa": (fm & tl).astype(int)}, index=df_in.index)

def bucket_build_year(df_in):
    by = pd.to_numeric(df_in["build_year"], errors="coerce")
    r = pd.DataFrame(index=df_in.index)
    r["by_pre1950"] = ((by < 1950) & by.notna()).astype(int)
    r["by_1950_1990"] = ((by >= 1950) & (by < 1990)).astype(int)
    r["by_1990_2000"] = ((by >= 1990) & (by < 2000)).astype(int)
    r["by_2000_2010"] = ((by >= 2000) & (by < 2010)).astype(int)
    r["by_2010plus"] = (by >= 2010).astype(int)
    r["by_unknown"] = by.isna().astype(int)
    return r

# Static features
title_feats = extract_title_features(df["title"])
ori_feats = extract_orientation_features(df["orientation"])
ratio_feats = compute_ratio_features(df)
villa_feats = detect_villa(df)
by_feats = bucket_build_year(df)

# Numeric
numeric_cols = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
num_data = df[numeric_cols].copy()
for c in numeric_cols:
    num_data[c] = pd.to_numeric(num_data[c], errors="coerce")
medians = num_data.median()
num_data = num_data.fillna(medians)

# Frequency encoding
freq_c = df["community"].value_counts().to_dict()
freq_s = df["subdistrict"].value_counts().to_dict()
community_enc = df["community"].map(freq_c).fillna(1).values
subdistrict_enc = df["subdistrict"].map(freq_s).fillna(1).values

# Target encoding
global_mean = target_series.mean()
def make_target_map(col, smooth):
    agg = df.groupby(col)["log_pps"].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)
    return smoothed.to_dict()

district_map = make_target_map("district", 1)
subdistrict_map = make_target_map("subdistrict", 10)
community_map = make_target_map("community", 50)

district_enc = df["district"].map(district_map).fillna(global_mean).values
subdistrict_target_enc = df["subdistrict"].map(subdistrict_map).fillna(global_mean).values
community_target_enc = df["community"].map(community_map).fillna(global_mean).values

# Interaction features (target-encoding-based)
total_rooms = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(2) + \
              pd.to_numeric(df["livingrooms"], errors="coerce").fillna(1)
house_age = (2024 - pd.to_numeric(df["build_year"], errors="coerce").fillna(2000)).clip(0, 100)
district_rooms_interact = district_enc * total_rooms
by_district_interact = district_enc * house_age

# Feature matrix
X = np.column_stack([
    num_data.values, ratio_feats.values, title_feats.values, ori_feats.values,
    by_feats.values, villa_feats.values,
    community_enc, subdistrict_enc, district_enc, subdistrict_target_enc, community_target_enc,
    district_rooms_interact, by_district_interact
])

# Floor OneHot
floor_series = df["floor_region"].fillna("未知").astype(str)
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
floor_ohe = ohe.fit_transform(floor_series.values.reshape(-1, 1))
X = np.column_stack([X, floor_ohe])
y = y_log_pps
print(f"Feature matrix: {X.shape}")

# ============================================================
# Stacking ensemble (5-fold OOF)
# ============================================================
print("Training stacking ensemble (3-fold OOF)...")
base_gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                     min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE)
base_hgbr = HistGradientBoostingRegressor(max_depth=8, l2_regularization=5.0,
                                          min_samples_leaf=20, learning_rate=0.1, random_state=RANDOM_STATE)

kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
n_models = 2
oof_preds = np.zeros((len(df), n_models))

for fold, (tr_idx, va_idx) in enumerate(kf.split(df)):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    for i, m in enumerate([deepcopy(base_gbr), deepcopy(base_hgbr)]):
        m.fit(Xtr, ytr)
        oof_preds[va_idx, i] = m.predict(Xva)
    print(f"  Fold {fold+1}/3 done")

meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
meta.fit(oof_preds, y)

# Evaluate OOF
pred_log = meta.predict(oof_preds)
pred_price = np.exp(pred_log) * area_sqm
rmse = np.sqrt(mean_squared_error(y_house_price, pred_price))
mae = mean_absolute_error(y_house_price, pred_price)
print(f"\n  Stacking OOF: RMSE={rmse:.2f}, MAE={mae:.2f}")

# ============================================================
# Retrain on full data and save
# ============================================================
print("\nRetraining on full data and saving model...")
full_gbr = deepcopy(base_gbr).fit(X, y)
full_hgbr = deepcopy(base_hgbr).fit(X, y)

full_meta_feats = np.column_stack([m.predict(X) for m in [full_gbr, full_hgbr]])
meta.fit(full_meta_feats, y)

final_model = {
    "type": "stacking",
    "base_models": {"gbr": full_gbr, "histgbr": full_hgbr},
    "meta_learner": meta,
    "model_names": ["gbr", "histgbr"],
}

joblib.dump(final_model, "model_stacking.pkl")
joblib.dump(final_model, "model.pkl")

# Verify
pred_log_final = meta.predict(full_meta_feats)
pred_price_final = np.exp(pred_log_final) * area_sqm
rmse_final = np.sqrt(mean_squared_error(y_house_price, pred_price_final))
mae_final = mean_absolute_error(y_house_price, pred_price_final)
print(f"  Train RMSE: {rmse_final:.2f}")
print(f"  Train MAE: {mae_final:.2f}")
print(f"  Models saved to: model_stacking.pkl, model.pkl")
print("\nDone!")
