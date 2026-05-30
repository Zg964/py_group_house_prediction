"""
上海房价预测 —— 本地训练脚本（Phase 2 优化版）
============================================
核心优化:
  - K-Fold 目标编码（防止数据泄露）
  - XGBoost / LightGBM / GBR 多模型对比
  - Stacking 集成（RF + XGBoost + GBR → Ridge meta-learner）
  - 数值特征 log 变换（area_sqm, total_rooms, area_per_bedroom）
  - 交互特征（district × total_rooms, build_year × district）
  - 极端值动态裁剪（PPS 双侧裁剪 + area_sqm winsorization）

功能:
  1. 读取 train.csv，执行探索性数据分析 (EDA)
  2. 特征工程（与 predict.py 保持一致）
  3. 训练多种回归模型并对比（使用 log(PPS) 目标，house_price 评估）
  4. K-Fold 交叉验证评估（内置目标编码，防止泄露）
  5. Stacking 集成评估
  6. 保存模型（用于 predict.py 加载）

运行: python train_model.py
依赖: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
"""

import os
import warnings
warnings.filterwarnings("ignore")

# Set matplotlib backend to Agg to avoid tkinter thread issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from copy import deepcopy
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
STACKING_SAVE_PATH = "model_stacking.pkl"
RANDOM_STATE = 42
N_FOLDS = 5
PPS_LOWER_PERCENTILE = 0.5   # PPS 下侧裁剪
PPS_UPPER_PERCENTILE = 99.5  # PPS 上侧裁剪
AREA_LOWER_PERCENTILE = 0.5  # area_sqm winsorization 下侧
AREA_UPPER_PERCENTILE = 99.5 # area_sqm winsorization 上侧


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
    """分析 PPS 目标变量分布（双侧裁剪版）"""
    print("\n" + "=" * 60)
    print("1b. PPS (price_per_sqm) 分布分析")
    print("=" * 60)

    pps = df["house_price"] / df["area_sqm"]
    lower = np.percentile(pps, PPS_LOWER_PERCENTILE)
    upper = np.percentile(pps, PPS_UPPER_PERCENTILE)
    pps_clipped = np.clip(pps, lower, upper)
    pps_log = np.log(pps_clipped)

    print(f"  PPS 偏度 (原始): {pps.skew():.2f}")
    print(f"  PPS 偏度 (裁剪+log): {pd.Series(pps_log).skew():.2f}")
    print(f"  PPS 裁剪范围: [{lower:.2f}, {upper:.2f}] 万元/平米")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(pps, bins=80, alpha=0.7)
    axes[0].set_title("PPS 原始分布 (skew={:.2f})".format(pps.skew()))

    axes[1].hist(pps_clipped, bins=80, alpha=0.7)
    axes[1].set_title(f"PPS 裁剪 [{lower:.1f}, {upper:.1f}] (skew={pd.Series(pps_clipped).skew():.2f})")

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

    # 3. area_sqm winsorization (双侧裁剪)
    area = pd.to_numeric(df["area_sqm"], errors="coerce")
    area_lower = np.percentile(area.dropna(), AREA_LOWER_PERCENTILE)
    area_upper = np.percentile(area.dropna(), AREA_UPPER_PERCENTILE)
    n_area_clipped = ((area < area_lower) | (area > area_upper)).sum()
    df["area_sqm"] = np.clip(area, area_lower, area_upper)
    print(f"  area_sqm winsorization [{area_lower:.1f}, {area_upper:.1f}]: 裁剪 {n_area_clipped} 条")

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
    新增: log 变换特征 (area_sqm, total_rooms, area_per_bedroom)
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

    # 每卧室面积
    result["area_per_bedroom"] = area / (bedrooms + 1e-6)

    # total_floor 分桶与 floor_ratio 交互
    floor_bucket = pd.cut(
        total_floor,
        bins=[0, 6, 18, 200],
        labels=["low_rise", "mid_rise", "high_rise"],
    )
    result["floor_bucket_low"] = (floor_bucket == "low_rise").astype(int)
    result["floor_bucket_mid"] = (floor_bucket == "mid_rise").astype(int)
    result["floor_bucket_high"] = (floor_bucket == "high_rise").astype(int)
    result["floor_ratio_x_total"] = result["floor_height_ratio"] * total_floor

    # ---- Log 变换特征 (高偏度数值) ----
    result["log_area_sqm"] = np.log(area + 1e-6)
    result["log_total_rooms"] = np.log(result["total_rooms"] + 1e-6)
    result["log_area_per_bedroom"] = np.log(result["area_per_bedroom"] + 1e-6)

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
    目标编码（含平滑）- 全量数据版本（用于最终训练和测试集推理）。
    smooth: 平滑系数，越大越保守
    """
    global_mean = target.mean()
    agg = train.groupby(col)[target.name].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)
    encoding_map = smoothed.to_dict()
    train_enc = train[col].map(encoding_map).fillna(global_mean)
    test_enc = test[col].map(encoding_map).fillna(global_mean)
    return train_enc.values, test_enc.values


def kfold_target_encode(df, col, target, smooth=1, n_folds=N_FOLDS):
    """
    K-Fold 目标编码（防止数据泄露）。
    对训练数据的第 i 折，用其他 n-1 折计算编码映射后再转换第 i 折。
    返回编码后的值（与 df 顺序一致）。
    """
    encoded = np.zeros(len(df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    global_mean = target.mean()

    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        target_fold = target.iloc[train_idx]

        agg = train_fold.groupby(col)[target_fold.name].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)
        encoding_map = smoothed.to_dict()

        encoded[val_idx] = val_fold[col].map(encoding_map).fillna(global_mean).values

    return encoded


def _get_district_rooms_interaction(df):
    """district × total_rooms 交互特征"""
    bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(2)
    livingrooms = pd.to_numeric(df["livingrooms"], errors="coerce").fillna(1)
    total_rooms = bedrooms + livingrooms
    district_clean = df["district"].astype(str).str.strip().str.replace("二手房", "", regex=False)
    # 使用 district 的数值编码与 total_rooms 相乘
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    district_num = le.fit_transform(district_clean)
    return district_num * total_rooms


def _get_build_year_district_interaction(df):
    """build_year × district 交互特征"""
    build_year = pd.to_numeric(df["build_year"], errors="coerce").fillna(2000)
    district_clean = df["district"].astype(str).str.strip().str.replace("二手房", "", regex=False)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    district_num = le.fit_transform(district_clean)
    return district_num * build_year


def build_features(df, target=None, is_train=True, ref_freq=None, ref_target=None, medians_ref=None):
    """
    构建完整特征矩阵（与 predict.py 保持一致）。
    如果是训练集 (is_train=True)，需要 target 来进行目标编码。
    如果是测试集 (is_train=False)，需要传入 ref_freq 和 ref_target 引用。
    注意: is_train=True 时使用全量目标编码（有泄露风险），
          K-Fold 无泄露版本请使用 build_features_with_target_enc() 并传入 fold 参数。
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

    # 交互特征
    district_rooms = _get_district_rooms_interaction(df)
    by_district = _get_build_year_district_interaction(df)

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
            district_rooms,
            by_district,
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


def build_features_leakproof(df, target, train_idx, val_idx, ref_freq=None, medians_ref=None):
    """
    无泄露特征构建：对 train/val 拆分进行目标编码。
    - 目标编码: 仅在 train_idx 上计算编码映射
    - 频率编码: 仅在 train_idx 上计算频率
    - 数值填充: 仅在 train_idx 上计算中位数
    """
    # ---- 静态特征（无信息泄露风险）----
    title_feats = extract_title_features(df)
    ori_feats = extract_orientation_features(df)
    ratio_feats = compute_ratio_features(df)
    villa_feats = detect_villa(df)
    by_feats = bucket_build_year(df)

    # 交互特征
    district_rooms = _get_district_rooms_interaction(df)
    by_district = _get_build_year_district_interaction(df)

    # ---- 数值特征 ----
    numeric_cols = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
    numeric_data = df[numeric_cols].copy()
    for col in numeric_cols:
        numeric_data[col] = pd.to_numeric(numeric_data[col], errors="coerce")
    # 仅在训练集计算中位数
    train_median = numeric_data.iloc[train_idx].median()
    numeric_data = numeric_data.fillna(train_median)

    # ---- 频率编码（仅在训练集上计算频率）----
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    target_train = target.iloc[train_idx]
    target_val = target.iloc[val_idx]

    # community 频率编码
    freq_c = df_train["community"].value_counts().to_dict()
    community_enc = np.zeros(len(df))
    community_enc[train_idx] = df_train["community"].map(freq_c).fillna(1).values
    community_enc[val_idx] = df_val["community"].map(freq_c).fillna(1).values

    # subdistrict 频率编码
    freq_s = df_train["subdistrict"].value_counts().to_dict()
    subdistrict_enc = np.zeros(len(df))
    subdistrict_enc[train_idx] = df_train["subdistrict"].map(freq_s).fillna(1).values
    subdistrict_enc[val_idx] = df_val["subdistrict"].map(freq_s).fillna(1).values

    # ---- 目标编码（仅在训练集上计算）----
    global_mean = target_train.mean()

    # district 目标编码（用临时 DataFrame 做 groupby）
    train_with_target = df_train.copy()
    train_with_target["_target_"] = target_train.values
    agg_d = train_with_target.groupby("district")["_target_"].agg(["mean", "count"])
    smoothed_d = (agg_d["mean"] * agg_d["count"] + global_mean * 1) / (agg_d["count"] + 1)
    map_d = smoothed_d.to_dict()
    district_enc = np.zeros(len(df))
    district_enc[train_idx] = df_train["district"].map(map_d).fillna(global_mean).values
    district_enc[val_idx] = df_val["district"].map(map_d).fillna(global_mean).values

    # subdistrict 目标编码
    agg_s = train_with_target.groupby("subdistrict")["_target_"].agg(["mean", "count"])
    smoothed_s = (agg_s["mean"] * agg_s["count"] + global_mean * 10) / (agg_s["count"] + 10)
    map_s = smoothed_s.to_dict()
    subd_enc = np.zeros(len(df))
    subd_enc[train_idx] = df_train["subdistrict"].map(map_s).fillna(global_mean).values
    subd_enc[val_idx] = df_val["subdistrict"].map(map_s).fillna(global_mean).values

    # ---- floor_region OneHot ----
    floor_series = df["floor_region"].fillna("未知").astype(str)
    all_floor = floor_series.unique()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[all_floor])
    floor_ohe = ohe.fit_transform(floor_series.values.reshape(-1, 1))

    # ---- 组装 ----
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
            subd_enc,
            district_rooms,
            by_district,
        ]
    )
    X = np.column_stack([X, floor_ohe])

    return X[train_idx], X[val_idx], train_median


# ============================================================
# 4. 模型训练与评估（无泄露版本）
# ============================================================

def compute_pps_target(y_house_price, area_sqm):
    """计算 PPS 目标（双侧裁剪 + log）"""
    pps = y_house_price / area_sqm
    lower = np.percentile(pps, PPS_LOWER_PERCENTILE)
    upper = np.percentile(pps, PPS_UPPER_PERCENTILE)
    pps_clipped = np.clip(pps, lower, upper)
    y_pps_log = np.log(pps_clipped)
    return y_pps_log, lower, upper


def train_and_select(df, y_house_price, area_sqm):
    """
    训练多个模型，选择最优（无数据泄露）。
    所有模型以 log(PPS) 为目标进行训练，但使用 house_price 进行评估。
    目标编码仅在训练集内计算，验证集无泄露。
    """
    print("\n" + "=" * 60)
    print("4. 模型训练与选择 (目标: log(PPS), 评估: house_price, 无泄露编码)")
    print("=" * 60)

    # ---- 计算 PPS 目标 ----
    y_pps_log, pps_lower, pps_upper = compute_pps_target(y_house_price, area_sqm)
    print(f"  PPS 裁剪范围: [{pps_lower:.2f}, {pps_upper:.2f}] 万元/平米")
    print(f"  log(PPS) 偏度: {pd.Series(y_pps_log).skew():.2f}")

    # ---- 先做特征拆分 ----
    # 使用 train_test_split 分割 df 索引
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE
    )

    # 构建无泄露特征
    target_series = pd.Series(y_pps_log, name="log_pps")
    X_train, X_val, _ = build_features_leakproof(
        df, target_series, train_idx, val_idx
    )

    # 分割目标
    y_train_l = y_pps_log[train_idx]
    y_val_l = y_pps_log[val_idx]
    hp_train = y_house_price[train_idx]
    hp_val = y_house_price[val_idx]
    area_train = area_sqm[train_idx]
    area_val = area_sqm[val_idx]

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

    # ---- XGBoost 模型 ----
    try:
        import xgboost as xgb
        models += [
            ("XGB(lr=0.05,n=1000,md=6)+PPS", xgb.XGBRegressor(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=50, eval_metric="rmse",
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
            )),
            ("XGB(lr=0.03,n=1500,md=5)+PPS", xgb.XGBRegressor(
                n_estimators=1500, learning_rate=0.03, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=50, eval_metric="rmse",
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
            )),
            ("XGB(lr=0.1,n=500,md=7)+PPS", xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.1, max_depth=7,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
            )),
        ]
        has_xgboost = True
    except ImportError:
        print("  [注意] xgboost 未安装，跳过 XGBoost 模型")
        has_xgboost = False

    # ---- 训练与评估 ----
    for name, model_cls in models:
        m = deepcopy(model_cls)
        if "XGB" in name and has_xgboost and hasattr(model_cls, 'early_stopping_rounds') and model_cls.early_stopping_rounds is not None and model_cls.early_stopping_rounds > 0:
            m.fit(X_train, y_train_l, eval_set=[(X_val, y_val_l)], verbose=False)
        else:
            m.fit(X_train, y_train_l)
        pred_log = m.predict(X_val)
        pred_price = np.exp(pred_log) * area_val
        rmse = np.sqrt(mean_squared_error(hp_val, pred_price))
        mae = mean_absolute_error(hp_val, pred_price)
        r2 = r2_score(hp_val, pred_price)
        results.append({"name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": m})
        print(f"  {name:40s} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | R2: {r2:.4f}")

    # 找最优模型
    best = min(results, key=lambda r: r["rmse"])
    print(f"\n{'=' * 60}")
    print(f"最优模型: {best['name']}")
    print(f"  RMSE: {best['rmse']:.2f}")
    print(f"  MAE : {best['mae']:.2f}")
    print(f"  R2  : {best['r2']:.4f}")

    # 打印 Top-3
    sorted_results = sorted(results, key=lambda r: r["rmse"])
    print(f"\nTop-3 模型:")
    for i, r in enumerate(sorted_results[:3]):
        print(f"  {i+1}. {r['name']:40s} | RMSE: {r['rmse']:8.2f} | MAE: {r['mae']:8.2f} | R2: {r['r2']:.4f}")

    return best, results


# ============================================================
# 5. 交叉验证（无泄露版本）
# ============================================================

def cross_validate_best(df, y_house_price, area_sqm, model, name):
    """对最优模型进行 K 折交叉验证（内置无泄露目标编码）"""
    print("\n" + "=" * 60)
    print("5. 交叉验证 (log(PPS) + house_price 评估, 内置无泄露编码)")
    print("=" * 60)

    y_pps_log, pps_lower, pps_upper = compute_pps_target(y_house_price, area_sqm)
    target_series = pd.Series(y_pps_log, name="log_pps")
    print(f"  PPS 裁剪范围: [{pps_lower:.2f}, {pps_upper:.2f}]")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        # 构建无泄露特征
        X_tr, X_vl, _ = build_features_leakproof(
            df, target_series, train_idx, val_idx
        )
        y_tr = y_pps_log[train_idx]
        y_vl = y_pps_log[val_idx]
        hp_vl = y_house_price[val_idx]
        area_vl = area_sqm[val_idx]

        m = deepcopy(model)
        m.fit(X_tr, y_tr)

        pred_log = m.predict(X_vl)
        pred_price = np.exp(pred_log) * area_vl

        rmse = np.sqrt(mean_squared_error(hp_vl, pred_price))
        mae = mean_absolute_error(hp_vl, pred_price)
        r2 = r2_score(hp_vl, pred_price)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        print(f"  Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

    print(f"\n  5-Fold CV 平均:")
    print(f"    RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
    print(f"    MAE : {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
    print(f"    R2  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    return np.mean(rmse_scores)


# ============================================================
# 5b. Stacking 集成
# ============================================================

def train_stacking_ensemble(df, y_house_price, area_sqm):
    """
    Stacking 集成:
      - Base models: RF(n=500) + XGBoost + GBR
      - 5-fold out-of-fold 预测作为 meta features
      - Meta-learner: Ridge regression
      - 在 house_price 层面评估（还原后评估）
    """
    print("\n" + "=" * 60)
    print("5b. Stacking 集成 (RF + XGBoost + GBR → Ridge)")
    print("=" * 60)

    y_pps_log, pps_lower, pps_upper = compute_pps_target(y_house_price, area_sqm)
    target_series = pd.Series(y_pps_log, name="log_pps")
    print(f"  PPS 裁剪范围: [{pps_lower:.2f}, {pps_upper:.2f}]")

    # Base models
    base_models = {
        "rf": RandomForestRegressor(
            n_estimators=500, max_depth=20, min_samples_leaf=2,
            min_samples_split=4, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
        ),
    }

    try:
        import xgboost as xgb
        base_models["xgb"] = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, eval_metric="rmse",
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
        )
        has_xgb = True
    except ImportError:
        has_xgb = False

    meta_learner = Ridge(alpha=1.0, random_state=RANDOM_STATE)

    # 5-fold OOF 预测
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n = len(df)
    n_models = len(base_models)
    oof_preds = np.zeros((n, n_models))
    model_list = []
    model_names = list(base_models.keys())

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        X_tr, X_vl, _ = build_features_leakproof(
            df, target_series, train_idx, val_idx
        )
        y_tr = y_pps_log[train_idx]
        y_vl = y_pps_log[val_idx]

        fold_models = []
        for i, (m_name, m_cls) in enumerate(base_models.items()):
            m = deepcopy(m_cls)
            if m_name == "xgb" and has_xgb and hasattr(m_cls, 'early_stopping_rounds') and m_cls.early_stopping_rounds is not None:
                m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            else:
                m.fit(X_tr, y_tr)
            oof_preds[val_idx, i] = m.predict(X_vl)
            fold_models.append(m)

        model_list.append(fold_models)
        print(f"  Fold {fold}: OOF 预测完成")

    # 训练 meta-learner（使用 OOF 预测作为特征）
    meta_learner.fit(oof_preds, y_pps_log)

    # 评估 stacking 在训练集上的表现（使用 OOF 预测）
    meta_pred_log = meta_learner.predict(oof_preds)
    meta_pred_price = np.exp(meta_pred_log) * area_sqm

    rmse = np.sqrt(mean_squared_error(y_house_price, meta_pred_price))
    mae = mean_absolute_error(y_house_price, meta_pred_price)
    r2 = r2_score(y_house_price, meta_pred_price)

    print(f"\n  Stacking OOF 评估 (训练集 OOF):")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAE : {mae:.2f}")
    print(f"    R2  : {r2:.4f}")

    # 检查各个 base model 的 OOF 表现
    print(f"\n  Base Model OOF 表现:")
    for i, m_name in enumerate(model_names):
        oof_rmse = np.sqrt(mean_squared_error(y_house_price, np.exp(oof_preds[:, i]) * area_sqm))
        print(f"    {m_name:5s} OOF RMSE: {oof_rmse:.2f}")

    # 打包 stacking 模型
    stacking_model = {
        "base_models": base_models,
        "meta_learner": meta_learner,
        "model_names": model_names,
    }

    return stacking_model, rmse


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

def train_final_model(df, y_house_price, area_sqm, best_estimator, use_stacking=False):
    """在全部数据上重新训练并保存模型"""
    print("\n" + "=" * 60)
    print("7. 全量训练并保存模型 (目标: log(PPS))")
    print("=" * 60)

    y_pps_log, pps_lower, pps_upper = compute_pps_target(y_house_price, area_sqm)
    df = df.copy()
    df["log_pps"] = y_pps_log
    target_series = df["log_pps"]
    print(f"  使用 log(PPS) 目标训练, 裁剪范围: [{pps_lower:.2f}, {pps_upper:.2f}]")

    if use_stacking:
        # Stacking: 在全量数据上训练 base models 和 meta-learner
        stacked = best_estimator
        base_models = stacked["base_models"]
        meta_learner = stacked["meta_learner"]
        model_names = stacked["model_names"]

        # 在全量数据上计算特征（使用全量目标编码）
        X_full, ref_freq, ref_target, medians, all_floor, ohe = build_features(
            df, target_series, is_train=True
        )

        # 训练 base models
        full_base_models = {}
        for m_name, m_cls in base_models.items():
            m = deepcopy(m_cls)
            if m_name == "xgb" and hasattr(m_cls, 'early_stopping_rounds') and m_cls.early_stopping_rounds is not None:
                # 对全量训练不需要 early stopping
                m.n_estimators = min(m.n_estimators, 500)
                m.early_stopping_rounds = None
                import xgboost as xgb
                new_m = xgb.XGBRegressor(
                    n_estimators=500, learning_rate=m.learning_rate, max_depth=m.max_depth,
                    subsample=m.subsample, colsample_bytree=m.colsample_bytree,
                    n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
                )
                new_m.fit(X_full, y_pps_log)
                full_base_models[m_name] = new_m
            else:
                m.fit(X_full, y_pps_log)
                full_base_models[m_name] = m

        # 用 base models 生成 meta features
        meta_features = np.column_stack([m.predict(X_full) for m_name, m in full_base_models.items()])

        # 训练 meta-learner
        meta_learner.fit(meta_features, y_pps_log)

        final_model = {
            "type": "stacking",
            "base_models": full_base_models,
            "meta_learner": meta_learner,
            "model_names": model_names,
        }
        joblib.dump(final_model, STACKING_SAVE_PATH)
        print(f"  Stacking 模型已保存至: {STACKING_SAVE_PATH}")

        # 验证
        meta_pred_log = meta_learner.predict(meta_features)
        pred_price = np.exp(meta_pred_log) * area_sqm
        print(f"  训练集 RMSE: {np.sqrt(mean_squared_error(y_house_price, pred_price)):.2f}")
    else:
        X_full, ref_freq, ref_target, medians, all_floor, ohe = build_features(
            df, target_series, is_train=True
        )

        final_model = deepcopy(best_estimator)
        if isinstance(final_model, dict):
            # Stacking dict was passed directly
            pass
        elif hasattr(final_model, 'early_stopping_rounds'):
            final_model.early_stopping_rounds = None
        final_model.fit(X_full, y_pps_log)

        joblib.dump(final_model, MODEL_SAVE_PATH)
        print(f"  模型已保存至: {MODEL_SAVE_PATH}")

        # 验证
        sample_pred_log = final_model.predict(X_full[:5])
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

    # 构建特征（全量版本，用于特征名获取和后续对比）
    y_pps_log, _, _ = compute_pps_target(df["house_price"].values.astype(float),
                                          pd.to_numeric(df["area_sqm"], errors="coerce").fillna(
                                              df["area_sqm"].median()).values)
    df["log_pps"] = y_pps_log
    target_series = df["log_pps"]
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
        "log_area_sqm", "log_total_rooms", "log_area_per_bedroom",
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
    interaction_names = ["district_rooms_interact", "by_district_interact"]
    floor_names = [f"floor_{c}" for c in all_floor]
    feature_names = (
        numeric_names + ratio_names + title_names + ori_names
        + by_names + villa_names + enc_names + interaction_names + floor_names
    )

    print(f"  特征矩阵形状: {X.shape}")
    print(f"  特征数量: {X.shape[1]}")

    # Step 5: 训练与选择（无泄露版本）
    y_house_price = df["house_price"].values.astype(float)
    area_sqm = pd.to_numeric(df["area_sqm"], errors="coerce").fillna(
        df["area_sqm"].median()
    ).values

    best_result, all_results = train_and_select(df, y_house_price, area_sqm)

    # Step 5b: Stacking 集成
    stacking_model, stacking_rmse = train_stacking_ensemble(df, y_house_price, area_sqm)

    # Step 6: 交叉验证（对最优单模型）
    cv_rmse = cross_validate_best(
        df, y_house_price, area_sqm, best_result["model"], best_result["name"]
    )

    # Step 7: 决定最终模型（比较单模型 vs stacking）
    use_stacking = stacking_rmse < cv_rmse
    print(f"\n  Stacking OOF RMSE: {stacking_rmse:.2f} vs CV RMSE: {cv_rmse:.2f}")
    if use_stacking:
        print("  -> 选择 Stacking 集成作为最终模型")
    else:
        print(f"  -> 选择单模型 {best_result['name']} 作为最终模型")

    # Step 8: 特征重要性（对最优单模型）
    # 用全量数据重训练以获取 feature_importances_
    y_pps_log_full, _, _ = compute_pps_target(y_house_price, area_sqm)
    best_model_full = deepcopy(best_result["model"])
    if hasattr(best_model_full, 'early_stopping_rounds'):
        best_model_full.early_stopping_rounds = None
    best_model_full.fit(X, y_pps_log_full)
    analyze_feature_importance(best_model_full, feature_names)

    # Step 9: 全量训练并保存
    if use_stacking:
        final_model = train_final_model(
            df, y_house_price, area_sqm, stacking_model, use_stacking=True
        )
    else:
        final_model = train_final_model(
            df, y_house_price, area_sqm, best_result["model"], use_stacking=False
        )

    print("\n" + "=" * 60)
    print("训练完成！")
    if use_stacking:
        print(f"  Stacking 模型已保存至: {STACKING_SAVE_PATH}")
    else:
        print(f"  模型已保存至: {MODEL_SAVE_PATH}")
    print(f"  5-Fold CV RMSE (最佳单模型): {cv_rmse:.2f}")
    print("=" * 60)

    print("\n【后续步骤】")
    print(f"  1. 将 model.pkl/model_stacking.pkl 和 predict.py 一起打包为 zip")
    print("  2. 在 Codabench 提交 zip 文件")
    print("  3. 平台会自动运行: python3 predict.py --train train.csv --test test.csv --output predictions.csv")


if __name__ == "__main__":
    main()
