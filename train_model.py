"""
上海房价预测 —— 本地训练脚本（PPS 优化版）
===========================================
核心改动:
  - 目标变量改为 price_per_sqm (PPS)，取 log 后训练
  - 推理时还原: house_price = exp(pred_log_PPS) * area_sqm
  - 特征工程全面升级（18 关键词、朝向分组、Villa 识别、build_year 分桶等）

功能:
  1. 读取 train.csv，执行探索性数据分析 (EDA)
  2. 特征工程（与 predict.py 保持一致）
  3. 训练多种回归模型并对比（使用 log(PPS) 目标，house_price 评估）
  4. 交叉验证评估
  5. 保存模型（用于 predict.py 加载）

运行: python train_model.py
依赖: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# 0. 配置
# ============================================================
DATA_PATH = "train.csv"
MODEL_SAVE_PATH = "model.pkl"
RANDOM_STATE = 42


# ============================================================
# 1. 数据读取与探索
# ============================================================

def load_and_explore(path: str) -> pd.DataFrame:
    """加载数据并打印基本信息"""
    print("=" * 60)
    print("1. 数据加载与探索")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"\n数据集形状: {df.shape}")
    print(f"\n前 5 行:\n{df.head()}")
    print(f"\n字段信息:")
    print(df.info())
    print(f"\n基本统计量:\n{df.describe()}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")

    # 原始目标变量分布 (house_price)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df["house_price"], bins=80, kde=True)
    plt.title("house_price 分布")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df["house_price"])
    plt.title("house_price 箱线图")
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png", dpi=150)
    plt.close()
    print("\n[已保存] eda_target_distribution.png")

    # 类别字段频次
    cat_cols = ["district", "floor_region", "orientation"]
    for col in cat_cols:
        print(f"\n{col} 前 10 类:\n{df[col].value_counts().head(10)}")

    # 数值相关性
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Id" in numeric_cols:
        numeric_cols.remove("Id")
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("数值特征相关性热力图")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150)
    plt.close()
    print("\n[已保存] eda_correlation.png")

    # district 清洗前检查
    print(f"\ndistrict 样例 (清洗前): {df['district'].unique()[:10]}")

    return df


def analyze_pps(df: pd.DataFrame):
    """分析 PPS 目标变量分布"""
    print("\n" + "=" * 60)
    print("1b. PPS (price_per_sqm) 分布分析")
    print("=" * 60)

    pps = df["house_price"] / df["area_sqm"]
    pps_cap = np.percentile(pps, 99.5)
    pps_clipped = np.clip(pps, None, pps_cap)
    pps_log = np.log(pps_clipped)

    print(f"  PPS 偏度 (原始): {pps.skew():.2f}")
    print(f"  PPS 偏度 (裁剪+log): {pd.Series(pps_log).skew():.2f}")
    print(f"  PPS 裁剪上限 (99.5%): {pps_cap:.2f} 万元/平米")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(pps, bins=80, alpha=0.7)
    axes[0].set_title("PPS 原始分布 (skew={:.2f})".format(pps.skew()))

    axes[1].hist(pps_clipped, bins=80, alpha=0.7)
    axes[1].set_title(f"PPS 裁剪 99.5% (skew={pd.Series(pps_clipped).skew():.2f})")

    axes[2].hist(pps_log, bins=80, alpha=0.7)
    axes[2].set_title(f"log(PPS) 分布 (skew={pd.Series(pps_log).skew():.2f})")

    plt.tight_layout()
    plt.savefig("eda_pps_distribution.png", dpi=150)
    plt.close()
    print("[已保存] eda_pps_distribution.png")

    # 与 house_price 偏度对比
    hp_skew = df["house_price"].skew()
    print(f"  house_price 偏度: {hp_skew:.2f}")
    print(f"  PPS 偏度 (原始): {pps.skew():.2f}")
    print(f"  log(PPS) 偏度: {pd.Series(pps_log).skew():.2f}")
    print(f"  改善幅度: {abs(hp_skew) - abs(pd.Series(pps_log).skew()):.2f}")


# ============================================================
# 2. 数据清洗
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """执行数据清洗"""
    print("\n" + "=" * 60)
    print("2. 数据清洗")
    print("=" * 60)

    df = df.copy()

    # 1. 清洗 district 字段: 去掉 "二手房" 后缀
    df["district"] = df["district"].astype(str).str.strip()
    df["district"] = df["district"].str.replace("二手房", "", regex=False)
    print(f"  district 样例 (清洗后): {df['district'].unique()[:10]}")

    # 2. 删除可疑记录: bedrooms==1 AND area_sqm>200
    bedrooms_num = pd.to_numeric(df["bedrooms"], errors="coerce")
    suspicious = (bedrooms_num == 1) & (df["area_sqm"] > 200)
    n_suspicious = suspicious.sum()
    if n_suspicious > 0:
        df = df[~suspicious].reset_index(drop=True)
        print(f"  删除 {n_suspicious} 条可疑记录 (bedrooms==1 & area_sqm>200)")
    else:
        print("  无可疑记录删除")

    print(f"  清洗后数据集形状: {df.shape}")
    return df


# ============================================================
# 3. 特征工程（与 predict.py 保持一致）
# ============================================================

def extract_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 title 字段提取关键词特征（18 个关键词）。
    """
    title_series = df["title"].fillna("").astype(str).str.lower()

    keywords = {
        "has_subway": "地铁",
        "has_decorated": "精装|豪装",
        "has_elevator": "电梯",
        "has_school": "学区|学校",
        "has_movein": "拎包",
        "has_mature": "满五|满二|满",
        "has_garden": "花园|景观",
        "has_quiet": "安静|不临街",
        "has_corner": "边套|全明",
        "has_luxury": "豪华装修",
        "has_simple": "简单装修",
        "has_lianjia": "链家好房",
        "has_full_light": "全明",
        "has_original": "原装",
        "has_brand": "品牌",
        "has_view": "景观",
        "has_traffic": "交通",
        "has_tax_free": "满五|满二",
    }

    result = pd.DataFrame(index=df.index)
    for col_name, pattern in keywords.items():
        result[col_name] = title_series.str.contains(pattern, regex=True).astype(int)

    result["title_len"] = title_series.str.len()
    return result


def extract_orientation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 orientation 字段提取朝向分组特征（5 组）。
    朝向价值: 南北 > 南 > 东西 > 北 > 未知
    """
    ori_series = df["orientation"].fillna("").astype(str).str.lower()

    result = pd.DataFrame(index=df.index)
    result["ori_south_north"] = (
        ori_series.str.contains("南") & ori_series.str.contains("北")
    ).astype(int)
    result["ori_south"] = (
        ori_series.str.contains("南") & ~ori_series.str.contains("北")
    ).astype(int)
    result["ori_east_west"] = (
        ~ori_series.str.contains("南")
        & ~ori_series.str.contains("北")
        & (ori_series.str.contains("东") | ori_series.str.contains("西")
           | ori_series.str.contains("东南") | ori_series.str.contains("西南")
           | ori_series.str.contains("东北") | ori_series.str.contains("西北"))
    ).astype(int)
    result["ori_north"] = (
        ori_series.str.contains("北") & ~ori_series.str.contains("南")
    ).astype(int)
    result["ori_unknown"] = (
        (ori_series == "") | ori_series.isna()
        | ori_series.str.contains("进门", na=False)
        | (~result["ori_south_north"].astype(bool)
           & ~result["ori_south"].astype(bool)
           & ~result["ori_east_west"].astype(bool)
           & ~result["ori_north"].astype(bool))
    ).astype(int)
    return result


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算比例和派生数值特征。
    """
    result = pd.DataFrame(index=df.index)

    bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(2)
    livingrooms = pd.to_numeric(df["livingrooms"], errors="coerce").fillna(1)
    area = pd.to_numeric(df["area_sqm"], errors="coerce").fillna(
        df["area_sqm"].median()
    )
    total_floor = pd.to_numeric(df["total_floor"], errors="coerce").fillna(
        df["total_floor"].median()
    )
    build_year = pd.to_numeric(df["build_year"], errors="coerce").fillna(2000)

    result["total_rooms"] = bedrooms + livingrooms
    result["rooms_per_sqm"] = (bedrooms + livingrooms) / (area + 1e-6)
    result["area_per_room"] = area / (bedrooms + livingrooms + 1e-6)
    result["bedroom_ratio"] = bedrooms / (bedrooms + livingrooms + 1e-6)
    house_age = 2024 - build_year
    result["house_age"] = house_age.clip(0, 100)
    result["is_new_building"] = (house_age <= 5).astype(int)
    result["is_old_building"] = (house_age > 20).astype(int)
    floor_map = {"低": 0, "中": 1, "高": 2, "低区": 0, "中区": 1, "高区": 2}
    result["floor_region_ord"] = (
        df["floor_region"].astype(str).str.strip().map(floor_map).fillna(1)
    )
    result["floor_height_ratio"] = (result["floor_region_ord"] + 1) / (
        total_floor + 1
    )

    # 新增: 每卧室面积
    result["area_per_bedroom"] = area / (bedrooms + 1e-6)

    # 新增: total_floor 分桶与 floor_ratio 交互
    floor_bucket = pd.cut(
        total_floor,
        bins=[0, 6, 18, 200],
        labels=["low_rise", "mid_rise", "high_rise"],
    )
    result["floor_bucket_low"] = (floor_bucket == "low_rise").astype(int)
    result["floor_bucket_mid"] = (floor_bucket == "mid_rise").astype(int)
    result["floor_bucket_high"] = (floor_bucket == "high_rise").astype(int)
    result["floor_ratio_x_total"] = result["floor_height_ratio"] * total_floor

    return result


def detect_villa(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别别墅: floor_region 缺失 且 总楼层 <= 3
    """
    floor_region_missing = df["floor_region"].isna()
    total_floor_low = pd.to_numeric(df["total_floor"], errors="coerce").fillna(99) <= 3
    return pd.DataFrame({
        "is_villa": (floor_region_missing & total_floor_low).astype(int)
    }, index=df.index)


def bucket_build_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    build_year 分桶: pre-1950, 1950-1990, 1990-2000, 2000-2010, 2010+, unknown
    """
    by = pd.to_numeric(df["build_year"], errors="coerce")

    result = pd.DataFrame(index=df.index)
    result["by_pre1950"] = ((by < 1950) & by.notna()).astype(int)
    result["by_1950_1990"] = ((by >= 1950) & (by < 1990)).astype(int)
    result["by_1990_2000"] = ((by >= 1990) & (by < 2000)).astype(int)
    result["by_2000_2010"] = ((by >= 2000) & (by < 2010)).astype(int)
    result["by_2010plus"] = (by >= 2010).astype(int)
    result["by_unknown"] = by.isna().astype(int)
    return result


def frequency_encode(train, test, col):
    """频率编码"""
    freq = train[col].value_counts().to_dict()
    train_enc = train[col].map(freq).fillna(1)
    test_enc = test[col].map(freq).fillna(1)
    return train_enc.values, test_enc.values


def target_encode(train, test, col, target, smooth=1):
    """
    目标编码（含平滑）。
    smooth: 平滑系数，越大越保守
    """
    global_mean = target.mean()
    agg = train.groupby(col)[target.name].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)
    encoding_map = smoothed.to_dict()
    train_enc = train[col].map(encoding_map).fillna(global_mean)
    test_enc = test[col].map(encoding_map).fillna(global_mean)
    return train_enc.values, test_enc.values


def build_features(df, target=None, is_train=True, ref_freq=None, ref_target=None, medians_ref=None):
    """
    构建完整特征矩阵（与 predict.py 保持一致）。
    如果是训练集 (is_train=True)，需要 target 来进行目标编码。
    如果是测试集 (is_train=False)，需要传入 ref_freq 和 ref_target 引用。
    """
    title_feats = extract_title_features(df)
    ori_feats = extract_orientation_features(df)
    ratio_feats = compute_ratio_features(df)
    villa_feats = detect_villa(df)
    by_feats = bucket_build_year(df)

    if is_train:
        community_enc, _ = frequency_encode(df, df, "community")
        subdistrict_enc, _ = frequency_encode(df, df, "subdistrict")
        district_enc, _ = target_encode(df, df, "district", target, smooth=1)
        subdistrict_target_enc, _ = target_encode(df, df, "subdistrict", target, smooth=10)
        ref_freq = {
            "community": df["community"].value_counts().to_dict(),
            "subdistrict": df["subdistrict"].value_counts().to_dict(),
        }
        global_mean = target.mean()
        agg = df.groupby("district")[target.name].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * 1) / (agg["count"] + 1)
        ref_target_district = smoothed.to_dict()
        agg_s = df.groupby("subdistrict")[target.name].agg(["mean", "count"])
        smoothed_s = (agg_s["mean"] * agg_s["count"] + global_mean * 10) / (agg_s["count"] + 10)
        ref_target_subdistrict = smoothed_s.to_dict()
        ref_target = {"district": ref_target_district, "subdistrict": ref_target_subdistrict}
    else:
        freq_c = ref_freq.get("community", {})
        freq_s = ref_freq.get("subdistrict", {})
        community_enc = df["community"].map(freq_c).fillna(1).values
        subdistrict_enc = df["subdistrict"].map(freq_s).fillna(1).values
        district_enc = df["district"].map(ref_target.get("district", {})).fillna(
            target.mean() if target is not None else 1.0
        ).values
        subdistrict_target_enc = df["subdistrict"].map(ref_target.get("subdistrict", {})).fillna(
            target.mean() if target is not None else 1.0
        ).values

    numeric_cols = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
    numeric_data = df[numeric_cols].copy()
    for col in numeric_cols:
        numeric_data[col] = pd.to_numeric(numeric_data[col], errors="coerce")

    if is_train:
        medians_ref = numeric_data.median()
        numeric_data = numeric_data.fillna(medians_ref)
    else:
        numeric_data = numeric_data.fillna(medians_ref if medians_ref is not None else numeric_data.median())

    X = np.column_stack(
        [
            numeric_data.values,
            ratio_feats.values,
            title_feats.values,
            ori_feats.values,
            by_feats.values,
            villa_feats.values,
            community_enc,
            subdistrict_enc,
            district_enc,
            subdistrict_target_enc,
        ]
    )

    # floor_region OneHot
    floor_series = df["floor_region"].fillna("未知").astype(str)
    if is_train:
        all_floor = floor_series.unique()
    else:
        all_floor = ref_freq.get("floor_cats", floor_series.unique())

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[all_floor])
    floor_ohe = ohe.fit_transform(floor_series.values.reshape(-1, 1))
    X = np.column_stack([X, floor_ohe])

    if is_train:
        return X, ref_freq, ref_target, medians_ref, all_floor, ohe

    return X


# ============================================================
# 4. 模型训练与评估
# ============================================================

def evaluate_house_price(model, X_val, y_val_log_pps, val_area_sqm, name):
    """
    用 log(PPS) 模型预测，然后还原为 house_price 进行 RMSE/MAE/R2 评估。
    """
    pred_log_pps = model.predict(X_val)
    pred_price = np.exp(pred_log_pps) * val_area_sqm
    y_val_price = y_val_log_pps  # this was already passed through for area scaling
    rmse = np.sqrt(mean_squared_error(y_val_price, pred_price))
    mae = mean_absolute_error(y_val_price, pred_price)
    r2 = r2_score(y_val_price, pred_price)
    return rmse, mae, r2, pred_price


def train_and_select(X, y_house_price, area_sqm):
    """
    训练多个模型，选择最优。
    所有模型以 log(PPS) 为目标进行训练，但使用 house_price 进行评估。
    """
    print("\n" + "=" * 60)
    print("4. 模型训练与选择 (目标: log(PPS), 评估: house_price)")
    print("=" * 60)

    # ---- 计算 PPS 目标，裁剪 99.5% 分位后取 log ----
    pps = y_house_price / area_sqm
    pps_cap = np.percentile(pps, 99.5)
    pps_clipped = np.clip(pps, None, pps_cap)
    y_pps_log = np.log(pps_clipped)
    print(f"  PPS 裁剪上限 (99.5%): {pps_cap:.2f} 万元/平米")
    print(f"  house_price 偏度: {pd.Series(y_house_price).skew():.2f}")
    print(f"  PPS 偏度: {pd.Series(pps).skew():.2f}")
    print(f"  log(PPS) 偏度: {pd.Series(y_pps_log).skew():.2f}")

    # 分割时使用同一切分，确保一致对比
    X_train, X_val, y_train_l, y_val_l, hp_train, hp_val, area_train, area_val = train_test_split(
        X, y_pps_log, y_house_price, area_sqm,
        test_size=0.2, random_state=RANDOM_STATE
    )

    results = []

    # ---- 线性模型 ----
    models = [
        ("Ridge(alpha=1.0)+PPS", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ("Ridge(alpha=10.0)+PPS", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ("ElasticNet(l1=0.5)+PPS", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE)),
    ]

    # ---- 树集成模型 ----
    models += [
        ("RF(n=300,d=18)+PPS", RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=3,
            min_samples_split=6, n_jobs=-1, random_state=RANDOM_STATE
        )),
        ("RF(n=500,d=20)+PPS", RandomForestRegressor(
            n_estimators=500, max_depth=20, min_samples_leaf=2,
            min_samples_split=4, n_jobs=-1, random_state=RANDOM_STATE
        )),
        ("RF(n=400,d=18)+PPS", RandomForestRegressor(
            n_estimators=400, max_depth=18, min_samples_leaf=3,
            min_samples_split=5, n_jobs=-1, random_state=RANDOM_STATE
        )),
        ("GBR(lr=0.1,n=200)+PPS", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
        )),
        ("GBR(lr=0.05,n=500)+PPS", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
        )),
        ("HistGBR(default)+PPS", HistGradientBoostingRegressor(
            max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE
        )),
    ]

    # ---- 训练与评估 ----
    for name, model_cls in models:
        m = model_cls
        m.fit(X_train, y_train_l)
        pred_log = m.predict(X_val)
        pred_price = np.exp(pred_log) * area_val
        rmse = np.sqrt(mean_squared_error(hp_val, pred_price))
        mae = mean_absolute_error(hp_val, pred_price)
        r2 = r2_score(hp_val, pred_price)
        results.append({"name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": m})
        print(f"  {name:35s} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | R2: {r2:.4f}")

    # 找最优模型
    best = min(results, key=lambda r: r["rmse"])
    print(f"\n{'=' * 60}")
    print(f"最优模型: {best['name']}")
    print(f"  RMSE: {best['rmse']:.2f}")
    print(f"  MAE : {best['mae']:.2f}")
    print(f"  R2  : {best['r2']:.4f}")

    return best


# ============================================================
# 5. 交叉验证
# ============================================================

def cross_validate_best(X, y_house_price, area_sqm, model, name):
    """对最优模型进行 K 折交叉验证（使用 log(PPS) 目标，house_price 评估）"""
    print("\n" + "=" * 60)
    print("5. 交叉验证 (log(PPS) + house_price 评估)")
    print("=" * 60)

    pps = y_house_price / area_sqm
    pps_cap = np.percentile(pps, 99.5)
    y_pps_log = np.log(np.clip(pps, None, pps_cap))
    print(f"  [使用 log(PPS) 目标, 裁剪上限: {pps_cap:.2f}]")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y_pps_log[train_idx], y_pps_log[val_idx]
        hp_vl = y_house_price[val_idx]
        area_vl = area_sqm[val_idx]

        from copy import deepcopy
        m = deepcopy(model)
        m.fit(X_tr, y_tr)

        pred_log = m.predict(X_vl)
        pred_price = np.exp(pred_log) * area_vl

        rmse = np.sqrt(mean_squared_error(hp_vl, pred_price))
        mae = mean_absolute_error(hp_vl, pred_price)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        print(f"  Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}")

    print(f"\n  5-Fold CV 平均:")
    print(f"    RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
    print(f"    MAE : {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")

    return np.mean(rmse_scores)


# ============================================================
# 6. 特征重要性分析
# ============================================================

def analyze_feature_importance(model, feature_names):
    """分析特征重要性并绘图"""
    print("\n" + "=" * 60)
    print("6. 特征重要性分析")
    print("=" * 60)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n特征重要性排名:")
        for i in range(min(20, len(feature_names))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

        plt.figure(figsize=(10, 8))
        plt.title("特征重要性 (Top 20) - PPS 模型")
        plt.barh(range(min(20, len(feature_names))), importances[indices][:20][::-1])
        plt.yticks(
            range(min(20, len(feature_names))),
            [feature_names[i] for i in indices[:20][::-1]],
        )
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()
        print("\n[已保存] feature_importance.png")
    else:
        print("该模型不支持 feature_importances_ 属性")


# ============================================================
# 7. 全量训练与模型保存
# ============================================================

def train_final_model(X, y_house_price, area_sqm, best_model):
    """在全部数据上重新训练 log(PPS) 模型并保存"""
    print("\n" + "=" * 60)
    print("7. 全量训练并保存模型 (目标: log(PPS))")
    print("=" * 60)

    pps = y_house_price / area_sqm
    pps_cap = np.percentile(pps, 99.5)
    y_pps_log = np.log(np.clip(pps, None, pps_cap))
    print(f"  使用 log(PPS) 目标训练, 裁剪上限: {pps_cap:.2f}")

    final_model = best_model.__class__(**best_model.get_params())
    final_model.fit(X, y_pps_log)

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"  模型已保存至: {MODEL_SAVE_PATH}")

    # 验证
    loaded = joblib.load(MODEL_SAVE_PATH)
    sample_pred_log = loaded.predict(X[:5])
    sample_pred_price = np.exp(sample_pred_log) * area_sqm[:5]
    print(f"  加载验证 - 前 5 条预测总价: {np.round(sample_pred_price, 1)}")
    print(f"  加载验证 - 前 5 条实际总价: {np.round(y_house_price[:5], 1)}")

    return final_model


# ============================================================
# 主流程
# ============================================================

def main():
    # Step 1: 加载与 EDA
    df = load_and_explore(DATA_PATH)

    # Step 2: 数据清洗
    df = clean_data(df)

    # Step 3: PPS 目标分析
    analyze_pps(df)

    # Step 4: 特征工程
    print("\n" + "=" * 60)
    print("3. 特征工程")
    print("=" * 60)

    # 计算 PPS 目标（裁剪 + log）
    pps = df["house_price"] / df["area_sqm"]
    pps_cap = np.percentile(pps, 99.5)
    y_pps_log = np.log(np.clip(pps, None, pps_cap))
    # 将 log(PPS) 加入 DataFrame，供目标编码使用
    df["log_pps"] = y_pps_log
    target_series = df["log_pps"]

    # 构建特征（使用 log(PPS) 进行目标编码）
    X, ref_freq, ref_target, medians, all_floor, _ = build_features(
        df, target_series, is_train=True
    )

    # 特征名（用于可视化）
    numeric_names = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
    ratio_names = [
        "total_rooms", "rooms_per_sqm", "area_per_room", "bedroom_ratio",
        "house_age", "is_new_building", "is_old_building",
        "floor_region_ord", "floor_height_ratio",
        "area_per_bedroom", "floor_bucket_low", "floor_bucket_mid",
        "floor_bucket_high", "floor_ratio_x_total",
    ]
    title_names = [
        "has_subway", "has_decorated", "has_elevator", "has_school",
        "has_movein", "has_mature", "has_garden", "has_quiet",
        "has_corner", "has_luxury", "has_simple", "has_lianjia",
        "has_full_light", "has_original", "has_brand", "has_view",
        "has_traffic", "has_tax_free", "title_len",
    ]
    ori_names = [
        "ori_south_north", "ori_south", "ori_east_west", "ori_north", "ori_unknown",
    ]
    by_names = [
        "by_pre1950", "by_1950_1990", "by_1990_2000",
        "by_2000_2010", "by_2010plus", "by_unknown",
    ]
    villa_names = ["is_villa"]
    enc_names = ["community_freq", "subdistrict_freq", "district_target", "subdistrict_target"]
    floor_names = [f"floor_{c}" for c in all_floor]
    feature_names = (
        numeric_names + ratio_names + title_names + ori_names
        + by_names + villa_names + enc_names + floor_names
    )

    print(f"  特征矩阵形状: {X.shape}")
    print(f"  特征数量: {X.shape[1]}")

    # Step 5: 训练与选择
    y_house_price = df["house_price"].values.astype(float)
    area_sqm = pd.to_numeric(df["area_sqm"], errors="coerce").fillna(
        df["area_sqm"].median()
    ).values
    best_result = train_and_select(X, y_house_price, area_sqm)

    # Step 6: 交叉验证
    cv_rmse = cross_validate_best(
        X, y_house_price, area_sqm, best_result["model"], best_result["name"]
    )

    # Step 7: 特征重要性
    pps_full = y_house_price / area_sqm
    pps_cap_fi = np.percentile(pps_full, 99.5)
    best_result["model"].fit(X, np.log(np.clip(pps_full, None, pps_cap_fi)))
    analyze_feature_importance(best_result["model"], feature_names)

    # Step 8: 全量训练并保存
    final_model = train_final_model(X, y_house_price, area_sqm, best_result["model"])

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  模型已保存至: {MODEL_SAVE_PATH}")
    print(f"  5-Fold CV RMSE: {cv_rmse:.2f}")
    print("=" * 60)

    print("\n【后续步骤】")
    print(f"  1. 将 {MODEL_SAVE_PATH} 和 predict.py 一起打包为 zip")
    print("  2. 在 Codabench 提交 zip 文件")
    print("  3. 平台会自动运行: python3 predict.py --train train.csv --test test.csv --output predictions.csv")


if __name__ == "__main__":
    main()
